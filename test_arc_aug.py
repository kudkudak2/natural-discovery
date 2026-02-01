import numpy as np
import torch

from arc_aug import (
    AugmentSpec,
    PAD_TOKEN,
    SEP_TOKEN,
    VOCAB_SIZE,
    apply_color_map_np,
    apply_shift_np,
    augment_src_tgt_batch,
    augment_src_tgt_batch_with_params,
    invert_grids_torch,
    sample_color_map_np,
)
from arc_train_utils import (
    PAD_TOKEN as PAD_TOKEN_UTILS,
    TensorizedDataset,
    prepare_batch,
    prompt_seq_len,
)


def _make_prompt(*, grid_size: int, num_demos: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a minimal (src,tgt) batch with a valid -100 mask in tgt.
    Layout matches arc_train_utils._flatten_prompt:
      (x SEP y SEP) repeated `num_demos` times, then (test_x SEP)
    """
    g = int(grid_size)
    nd = int(num_demos)
    grid_tokens = g * g
    # One demo + one test_x.
    x = np.zeros((g, g), dtype=np.int64)
    y = np.zeros((g, g), dtype=np.int64)
    test_x = np.zeros((g, g), dtype=np.int64)

    # Add a few non-zeros to constrain bbox.
    x[0, 0] = 1
    x[g - 1, g - 1] = 2
    y[0, g - 1] = 3
    y[g - 1, 0] = 4
    test_x[g // 2, g // 2] = 5

    seq = []
    for _ in range(nd):
        seq += x.reshape(-1).tolist() + [int(SEP_TOKEN)] + y.reshape(-1).tolist() + [int(SEP_TOKEN)]
    seq += test_x.reshape(-1).tolist() + [int(SEP_TOKEN)]
    src = torch.tensor([seq], dtype=torch.long)

    # Target is full grid with some masked cells.
    tgt = torch.full((1, grid_tokens), -100, dtype=torch.long)
    tgt[0, 0] = 7
    tgt[0, g - 1] = 8
    tgt[0, grid_tokens - 1] = 9
    return src, tgt


def _make_prompt_variable(
    *, grid_size: int, output_grid_size: int, num_demos: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Same as _make_prompt but input grids are grid_size x grid_size and output grids
    are output_grid_size x output_grid_size (variable in/out layout).
    """
    g_in = int(grid_size)
    g_out = int(output_grid_size)
    nd = int(num_demos)
    in_tokens = g_in * g_in
    out_tokens = g_out * g_out
    x = np.zeros((g_in, g_in), dtype=np.int64)
    y = np.zeros((g_out, g_out), dtype=np.int64)
    test_x = np.zeros((g_in, g_in), dtype=np.int64)
    x[0, 0] = 1
    x[g_in - 1, g_in - 1] = 2
    y[0, g_out - 1] = 3
    y[g_out - 1, 0] = 4
    test_x[g_in // 2, g_in // 2] = 5
    seq = []
    for _ in range(nd):
        seq += x.reshape(-1).tolist() + [int(SEP_TOKEN)] + y.reshape(-1).tolist() + [int(SEP_TOKEN)]
    seq += test_x.reshape(-1).tolist() + [int(SEP_TOKEN)]
    src = torch.tensor([seq], dtype=torch.long)
    tgt = torch.full((1, out_tokens), -100, dtype=torch.long)
    tgt[0, 0] = 7
    tgt[0, out_tokens - 1] = 9
    return src, tgt


def test_apply_shift_np_basic():
    grid = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 2],
        ],
        dtype=np.int64,
    )
    out = apply_shift_np(grid, dy=1, dx=1, fill_value=0)
    expected = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(out, expected)


def test_color_map_invariants():
    rng = np.random.default_rng(0)
    m = sample_color_map_np(rng=rng, keep_background=True)
    assert int(m[0]) == 0
    assert int(m[int(SEP_TOKEN)]) == int(SEP_TOKEN)
    assert int(m[int(PAD_TOKEN)]) == int(PAD_TOKEN)
    assert set(int(x) for x in m[1:10].tolist()) == set(range(1, 10))

    rng = np.random.default_rng(1)
    m2 = sample_color_map_np(rng=rng, keep_background=False)
    assert int(m2[int(SEP_TOKEN)]) == int(SEP_TOKEN)
    assert int(m2[int(PAD_TOKEN)]) == int(PAD_TOKEN)
    assert set(int(x) for x in m2[0:10].tolist()) == set(range(0, 10))


def test_apply_color_map_identity_noop():
    tokens = np.array([0, 1, 2, int(SEP_TOKEN), int(PAD_TOKEN), 9], dtype=np.int64)
    ident = np.arange(int(VOCAB_SIZE), dtype=np.int64)
    out = apply_color_map_np(tokens, color_map=ident)
    assert np.array_equal(out, tokens)


def test_augment_identity_noop():
    src, tgt = _make_prompt(grid_size=3, num_demos=1)
    spec = AugmentSpec(enabled=True, geom_prob=0.0, color_prob=0.0, translate_prob=0.0, keep_background=True)
    out_src, out_tgt = augment_src_tgt_batch(
        src=src,
        tgt=tgt,
        grid_size=3,
        num_demos=1,
        generator=torch.Generator().manual_seed(0),
        spec=spec,
    )
    assert torch.equal(out_src, src)
    assert torch.equal(out_tgt, tgt)


def test_augment_mask_preserved_and_target_inverts_roundtrip():
    src, tgt = _make_prompt(grid_size=5, num_demos=1)
    spec = AugmentSpec(
        enabled=True,
        geom_prob=1.0,
        color_prob=1.0,
        translate_prob=1.0,
        translate_max=2,
        keep_background=True,
    )
    gen = torch.Generator().manual_seed(0)
    out_src, out_tgt, params = augment_src_tgt_batch_with_params(
        src=src,
        tgt=tgt,
        grid_size=5,
        num_demos=1,
        generator=gen,
        spec=spec,
    )

    # SEP tokens stay in place.
    assert torch.equal(out_src == int(SEP_TOKEN), src == int(SEP_TOKEN))

    # Mask cardinality preserved.
    assert int((out_tgt == -100).sum().item()) == int((tgt == -100).sum().item())

    # Invert augmented target and compare to original on valid cells.
    g = 5
    valid = (tgt != -100).reshape(1, g, g)
    tgt_filled = torch.where(valid, tgt.reshape(1, g, g), torch.zeros((1, g, g), dtype=torch.long))
    out_valid = (out_tgt != -100).reshape(1, g, g)
    out_filled = torch.where(out_valid, out_tgt.reshape(1, g, g), torch.zeros((1, g, g), dtype=torch.long))

    inv = invert_grids_torch(out_filled.unsqueeze(1), params=params).squeeze(1)  # (1,g,g)
    assert torch.equal(inv[valid], tgt_filled[valid])


def test_augment_different_input_output_grid_sizes():
    """Augment with input grid 3x3 and output grid 2x2; flip/transform must not see or add padding."""
    src, tgt = _make_prompt_variable(grid_size=3, output_grid_size=2, num_demos=1)
    spec = AugmentSpec(enabled=True, geom_prob=0.0, color_prob=0.0, translate_prob=0.0, keep_background=True)
    out_src, out_tgt = augment_src_tgt_batch(
        src=src,
        tgt=tgt,
        grid_size=3,
        num_demos=1,
        output_grid_size=2,
        generator=torch.Generator().manual_seed(0),
        spec=spec,
    )
    assert out_src.shape == src.shape and out_tgt.shape == tgt.shape
    assert torch.equal(out_src, src) and torch.equal(out_tgt, tgt)
    # Content must not contain PAD (augment works on content only, does not introduce padding).
    assert int((out_src == PAD_TOKEN).sum().item()) == 0


def test_augment_variable_length_content_only_no_padding():
    """When augment receives content-only slices (no padding), output has same shape and no PAD in content."""
    src, tgt = _make_prompt_variable(grid_size=2, output_grid_size=3, num_demos=1)
    spec = AugmentSpec(
        enabled=True,
        geom_prob=1.0,
        color_prob=0.0,
        translate_prob=0.0,
        keep_background=True,
    )
    out_src, out_tgt = augment_src_tgt_batch(
        src=src,
        tgt=tgt,
        grid_size=2,
        num_demos=1,
        output_grid_size=3,
        generator=torch.Generator().manual_seed(42),
        spec=spec,
    )
    assert out_src.shape == src.shape and out_tgt.shape == tgt.shape
    assert int((out_src == PAD_TOKEN).sum().item()) == 0


def test_augment_src_tgt_batch_with_params_different_input_output_grid_sizes():
    """
    augment_src_tgt_batch_with_params must support output_grid_size != grid_size (g_in != g_out).
    Regression: previously assumed single g for both x and y, causing reshape size mismatch.
    """
    # Layout that triggered the bug: g_in=3 (9 tokens), g_out=19 (361 tokens), nd=1.
    src, tgt = _make_prompt_variable(grid_size=3, output_grid_size=19, num_demos=1)
    assert src.shape == (1, 382), "expected prompt_seq_len(3, 1, 19)=382"
    assert tgt.shape == (1, 361), "expected 19*19=361"
    spec = AugmentSpec(enabled=True, geom_prob=0.0, color_prob=0.0, translate_prob=0.0, keep_background=True)
    out_src, out_tgt, params = augment_src_tgt_batch_with_params(
        src=src,
        tgt=tgt,
        grid_size=3,
        num_demos=1,
        output_grid_size=19,
        generator=torch.Generator().manual_seed(0),
        spec=spec,
    )
    assert out_src.shape == src.shape and out_tgt.shape == tgt.shape
    assert torch.equal(out_src, src) and torch.equal(out_tgt, tgt)


def test_color_map_applied_consistently_including_background():
    # Disable geom/translation so the only change is color mapping.
    src, tgt = _make_prompt(grid_size=4, num_demos=2)
    spec = AugmentSpec(
        enabled=True,
        geom_prob=0.0,
        color_prob=1.0,
        translate_prob=0.0,
        keep_background=False,
    )
    gen = torch.Generator().manual_seed(0)
    out_src, out_tgt, params = augment_src_tgt_batch_with_params(
        src=src,
        tgt=tgt,
        grid_size=4,
        num_demos=2,
        generator=gen,
        spec=spec,
    )

    # The same per-sample color map must be applied to the entire prompt sequence, including background cells,
    # and SEP/PAD must map to themselves.
    m = params.color_maps[0]  # (VOCAB_SIZE,)
    expected_src = m[src]
    assert torch.equal(expected_src, out_src)

    # Target: map valid cells; keep ignore positions at -100.
    grid_tokens = 4 * 4
    valid = tgt != -100
    tgt_filled = torch.where(valid, tgt, torch.zeros((1, grid_tokens), dtype=torch.long))
    mapped = m[tgt_filled]
    expected_tgt = torch.where(valid, mapped, torch.full_like(mapped, -100))
    assert torch.equal(expected_tgt, out_tgt)


def test_prepare_batch_variable_seq_len_preserves_padding():
    """
    With variable-length examples and T_max/G_max padding, augmentation is applied per-example
    to content only; padding in src (PAD_TOKEN) and tgt (-100) must remain unchanged.
    """
    # Two examples: (g_in=3, g_out=3, nd=1) and (g_in=3, g_out=2, nd=1) -> different src/tgt lengths.
    src0, tgt0 = _make_prompt(grid_size=3, num_demos=1)
    src1, tgt1 = _make_prompt_variable(grid_size=3, output_grid_size=2, num_demos=1)
    pool = TensorizedDataset(
        skill_id=0,
        split="test",
        grid_size=3,
        num_demos=1,
        src_list=[src0[0], src1[0]],
        tgt_list=[tgt0[0], tgt1[0]],
        grid_size_each=torch.tensor([3, 2], dtype=torch.long),
        num_demos_each=torch.tensor([1, 1], dtype=torch.long),
        output_grid_size=3,
    )
    T_max = 30
    G_max = 9
    gen = torch.Generator().manual_seed(123)
    batch = prepare_batch(
        batch_size=2,
        train_pool=pool,
        device=torch.device("cpu"),
        cpu_generator=gen,
        augment=AugmentSpec(enabled=True, geom_prob=0.5, color_prob=0.5, translate_prob=0.0, keep_background=True),
        grid_size=3,
        num_demos=1,
        T_max=T_max,
        G_max=G_max,
    )
    # Batch has 2 rows padded to T_max and G_max.
    assert batch.src.shape == (2, T_max)
    assert batch.tgt.shape == (2, G_max)
    # For each row, padding region must still be PAD (src) and -100 (tgt).
    for j in range(2):
        nd_j = int(batch.num_demos[j].item())
        g_out_j = int(batch.grid_size[j].item())
        content_len = prompt_seq_len(grid_size=3, num_demos=nd_j, output_grid_size=g_out_j)
        g_out_sq = g_out_j * g_out_j
        assert (batch.src[j, content_len:] == PAD_TOKEN).all(), f"row {j}: src padding overwritten"
        assert (batch.tgt[j, g_out_sq:] == -100).all(), f"row {j}: tgt padding overwritten"
    # key_padding_mask: True where padded.
    for j in range(2):
        nd_j = int(batch.num_demos[j].item())
        g_out_j = int(batch.grid_size[j].item())
        content_len = prompt_seq_len(grid_size=3, num_demos=nd_j, output_grid_size=g_out_j)
        assert batch.key_padding_mask[j, :content_len].eq(False).all()
        assert batch.key_padding_mask[j, content_len:].eq(True).all()

