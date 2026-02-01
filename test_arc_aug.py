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

