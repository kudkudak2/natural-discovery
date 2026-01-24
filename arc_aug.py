from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class AugmentSpec:
    """
    ARC prompt augmentation that preserves task semantics by applying the SAME transform
    consistently to:
      - all demo inputs x_i
      - all demo outputs y_i
      - the test input x_test
      - the test output y_test (training target)

    Supported invariances:
      - **Geometry**: dihedral D4 (rotations by 0/90/180/270, optionally mirrored)
      - **Colors**: global permutation of the color palette (optionally keeping background=0 fixed)
    """

    enabled: bool = True

    geom_prob: float = 1.0
    color_prob: float = 1.0

    keep_background: bool = True


def _check_square_grid_np(grid: np.ndarray) -> None:
    if grid.ndim != 2:
        raise ValueError(f"Expected 2D grid, got shape={grid.shape}")
    h, w = int(grid.shape[0]), int(grid.shape[1])
    if h != w:
        raise ValueError(f"Expected square grid, got shape={grid.shape}")


def apply_geom_np(grid: np.ndarray, *, code: int) -> np.ndarray:
    """
    Apply a D4 transform specified by `code` in [0..7].

    Encoding:
      - rot_k = code % 4  (number of 90-degree CCW rotations)
      - mirror = code // 4  (0=no mirror, 1=mirror horizontally after rotation)
    """
    _check_square_grid_np(grid)
    c = int(code) % 8
    rot_k = int(c % 4)
    mirror = int(c // 4)

    out = np.rot90(np.asarray(grid, dtype=np.int64), k=rot_k).astype(np.int64, copy=False)
    if mirror:
        out = np.fliplr(out)
    return out


def _identity_color_map_np() -> np.ndarray:
    return np.arange(6, dtype=np.int64)


def sample_color_map_np(*, rng: np.random.Generator, keep_background: bool) -> np.ndarray:
    """
    Returns a lookup table `m` of shape (6,) mapping tokens 0..5 -> 0..5.
    Token 5 is reserved for SEP and is always mapped to 5.
    """
    kb = bool(keep_background)
    m = _identity_color_map_np()
    if kb:
        perm = rng.permutation(np.arange(1, 5, dtype=np.int64))
        m[1:5] = perm
    else:
        perm = rng.permutation(np.arange(0, 5, dtype=np.int64))
        m[0:5] = perm
    m[5] = 5
    return m


def apply_color_map_np(tokens: np.ndarray, *, color_map: np.ndarray) -> np.ndarray:
    m = np.asarray(color_map, dtype=np.int64).reshape(-1)
    if int(m.shape[0]) != 6:
        raise ValueError(f"Expected color_map shape (6,), got {m.shape}")
    t = np.asarray(tokens, dtype=np.int64)
    if np.any((t < 0) | (t > 5)):
        raise ValueError("apply_color_map_np expects tokens in [0..5]")
    return m[t]


def sample_geom_code_np(*, rng: np.random.Generator) -> int:
    return int(rng.integers(0, 8))


def _prompt_expected_seq_len(*, grid_size: int, num_demos: int) -> int:
    g = int(grid_size)
    nd = int(num_demos)
    grid_tokens = g * g
    return nd * (2 * grid_tokens + 2) + (grid_tokens + 1)


def augment_prompt_np(
    *,
    demos: list[tuple[np.ndarray, np.ndarray]],
    test_in: np.ndarray,
    test_out: np.ndarray,
    rng: np.random.Generator,
    spec: AugmentSpec,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """
    Numpy augmentation for a single ARC task prompt.
    """
    if not bool(spec.enabled):
        return demos, test_in, test_out

    if float(spec.geom_prob) < 0.0 or float(spec.geom_prob) > 1.0:
        raise ValueError(f"geom_prob must be in [0,1], got {spec.geom_prob}")
    if float(spec.color_prob) < 0.0 or float(spec.color_prob) > 1.0:
        raise ValueError(f"color_prob must be in [0,1], got {spec.color_prob}")

    do_geom = float(rng.random()) < float(spec.geom_prob)
    do_color = float(rng.random()) < float(spec.color_prob)

    geom_code = sample_geom_code_np(rng=rng) if do_geom else 0
    color_map = sample_color_map_np(rng=rng, keep_background=bool(spec.keep_background)) if do_color else _identity_color_map_np()

    out_demos: list[tuple[np.ndarray, np.ndarray]] = []
    for x, y in demos:
        xx = apply_geom_np(np.asarray(x, dtype=np.int64), code=int(geom_code))
        yy = apply_geom_np(np.asarray(y, dtype=np.int64), code=int(geom_code))
        xx = apply_color_map_np(xx, color_map=color_map)
        yy = apply_color_map_np(yy, color_map=color_map)
        out_demos.append((xx, yy))

    ti = apply_geom_np(np.asarray(test_in, dtype=np.int64), code=int(geom_code))
    to = apply_geom_np(np.asarray(test_out, dtype=np.int64), code=int(geom_code))
    ti = apply_color_map_np(ti, color_map=color_map)
    to = apply_color_map_np(to, color_map=color_map)
    return out_demos, ti, to


def _apply_geom_torch(grids: torch.Tensor, *, codes: torch.Tensor) -> torch.Tensor:
    """
    Apply per-sample D4 transforms to `grids`.

    Args:
      grids: (B, N, H, W) long
      codes: (B,) long in [0..7]
    """
    if grids.ndim != 4:
        raise ValueError(f"Expected grids shape (B,N,H,W), got {tuple(grids.shape)}")
    b, n, h, w = (int(grids.shape[0]), int(grids.shape[1]), int(grids.shape[2]), int(grids.shape[3]))
    if h != w:
        raise ValueError(f"Expected square grids, got HxW={h}x{w}")
    if codes.ndim != 1 or int(codes.shape[0]) != b:
        raise ValueError(f"Expected codes shape (B,), got {tuple(codes.shape)} for B={b}")

    idx_map = _geom_index_map(size=int(h), device=grids.device)  # (8, g*g)
    cc = (codes.to(dtype=torch.long) % 8).clamp(min=0, max=7)
    gather_idx = idx_map.index_select(0, cc)  # (B, g*g)

    flat = grids.reshape(b, n, h * w)
    gather_idx = gather_idx.unsqueeze(1).expand(b, n, h * w)
    out = torch.gather(flat, dim=2, index=gather_idx)
    return out.reshape(b, n, h, w)


_GEOM_INDEX_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _geom_index_map(*, size: int, device: torch.device) -> torch.Tensor:
    """
    Return an index map for D4 transforms as a tensor of shape (8, g*g).

    For a grid flattened in row-major order, applying code `c` corresponds to:
      out_flat = in_flat[idx_map[c]]
    """
    g = int(size)
    if g <= 0:
        raise ValueError(f"size must be >= 1, got {g}")
    key = (int(g), str(device))
    cached = _GEOM_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    base = torch.arange(g * g, dtype=torch.long).reshape(g, g)
    maps: list[torch.Tensor] = []
    for code in range(8):
        rot_k = int(code % 4)
        mirror = int(code // 4)
        m = base
        if rot_k:
            m = torch.rot90(m, k=rot_k, dims=(0, 1))
        if mirror:
            m = torch.flip(m, dims=(1,))  # horizontal mirror (left-right)
        maps.append(m.reshape(-1))
    out = torch.stack(maps, dim=0).to(device)
    _GEOM_INDEX_CACHE[key] = out
    return out


def _sample_color_maps_torch(
    *,
    batch_size: int,
    generator: Optional[torch.Generator],
    device: torch.device,
    keep_background: bool,
) -> torch.Tensor:
    """
    Returns (B, 6) lookup tables mapping tokens 0..5 -> 0..5.
    """
    b = int(batch_size)
    if b <= 0:
        raise ValueError(f"batch_size must be >= 1, got {b}")
    kb = bool(keep_background)
    maps = torch.empty((b, 6), device=device, dtype=torch.long)
    maps[:, 5] = 5
    if kb:
        maps[:, 0] = 0
        # Permute [1..4] per-sample by sorting random scores.
        scores = torch.rand((b, 4), generator=generator, device=device)
        order = torch.argsort(scores, dim=1)  # (B,4) in [0..3]
        maps[:, 1:5] = (order + 1).to(torch.long)
    else:
        scores = torch.rand((b, 5), generator=generator, device=device)
        order = torch.argsort(scores, dim=1)  # (B,5) in [0..4]
        maps[:, 0:5] = order.to(torch.long)
    return maps


def _apply_color_maps_torch(tokens: torch.Tensor, *, maps: torch.Tensor) -> torch.Tensor:
    """
    Apply per-sample token remapping.

    tokens: (B, ...) long with values in [0..5]
    maps: (B, 6) long
    """
    if maps.ndim != 2 or int(maps.shape[1]) != 6:
        raise ValueError(f"Expected maps shape (B,6), got {tuple(maps.shape)}")
    if tokens.ndim < 1:
        raise ValueError("tokens must have at least 1 dimension")
    b = int(tokens.shape[0])
    if int(maps.shape[0]) != b:
        raise ValueError(f"maps batch mismatch: maps B={int(maps.shape[0])} vs tokens B={b}")
    if tokens.dtype != torch.long:
        tokens = tokens.to(torch.long)
    # Avoid a device sync in the training hot-path: only validate token range on CPU.
    if tokens.device.type == "cpu":
        if int(tokens.min().item()) < 0 or int(tokens.max().item()) > 5:
            raise ValueError("_apply_color_maps_torch expects tokens in [0..5]")
    # Advanced indexing: maps[b, token] for each element.
    idx0 = torch.arange(b, device=tokens.device).view(b, *([1] * (tokens.ndim - 1)))
    return maps[idx0, tokens]


def augment_src_tgt_batch(
    *,
    src: torch.Tensor,
    tgt: torch.Tensor,
    grid_size: int,
    num_demos: int,
    generator: Optional[torch.Generator],
    spec: AugmentSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Token-level augmentation for the Transformer training pipeline.

    `src` format must match arc_train_utils._flatten_prompt:
      (x SEP y SEP) repeated `num_demos` times, then (test_x SEP)

    Args:
      src: (B, T) long tokens in [0..5]
      tgt: (B, grid_tokens) long tokens in [0..4]
    """
    if not bool(spec.enabled):
        return src, tgt

    if float(spec.geom_prob) < 0.0 or float(spec.geom_prob) > 1.0:
        raise ValueError(f"geom_prob must be in [0,1], got {spec.geom_prob}")
    if float(spec.color_prob) < 0.0 or float(spec.color_prob) > 1.0:
        raise ValueError(f"color_prob must be in [0,1], got {spec.color_prob}")

    if src.ndim != 2:
        raise ValueError(f"Expected src shape (B,T), got {tuple(src.shape)}")
    b, t = int(src.shape[0]), int(src.shape[1])
    g = int(grid_size)
    nd = int(num_demos)
    if g <= 0:
        raise ValueError(f"grid_size must be >= 1, got {g}")
    if nd <= 0:
        raise ValueError(f"num_demos must be >= 1, got {nd}")
    grid_tokens = int(g * g)
    expected_t = _prompt_expected_seq_len(grid_size=g, num_demos=nd)
    if int(t) != int(expected_t):
        raise ValueError(f"Unexpected src length={t} (expected {expected_t}) for grid_size={g}, num_demos={nd}")
    if tgt.ndim != 2 or int(tgt.shape[0]) != b or int(tgt.shape[1]) != grid_tokens:
        raise ValueError(f"Expected tgt shape (B,{grid_tokens}), got {tuple(tgt.shape)}")

    device = src.device
    # --- sample which transforms to apply ---
    # geom: per-sample code in [0..7], or 0 for identity when not applied
    p_geom = float(spec.geom_prob)
    if p_geom <= 0.0:
        geom_codes = torch.zeros((b,), device=device, dtype=torch.long)
    else:
        apply_geom = torch.rand((b,), device=device, generator=generator) < float(p_geom)
        codes = torch.randint(0, 8, (b,), device=device, generator=generator, dtype=torch.long)
        geom_codes = torch.where(apply_geom, codes, torch.zeros_like(codes))

    # color: per-sample map, or identity map when not applied
    p_col = float(spec.color_prob)
    if p_col <= 0.0:
        color_maps = torch.arange(6, device=device, dtype=torch.long).unsqueeze(0).repeat(b, 1)
    else:
        apply_col = torch.rand((b,), device=device, generator=generator) < float(p_col)
        rand_maps = _sample_color_maps_torch(
            batch_size=b,
            generator=generator,
            device=device,
            keep_background=bool(spec.keep_background),
        )
        ident = torch.arange(6, device=device, dtype=torch.long).unsqueeze(0).repeat(b, 1)
        color_maps = torch.where(apply_col.unsqueeze(1), rand_maps, ident)

    # --- parse src into 7 grids (3*(x,y) + test_x), apply geom, then stitch back ---
    # Gather grids into (B, 7, g, g)
    grids: list[torch.Tensor] = []
    off = 0
    for _ in range(nd):
        x = src[:, off : off + grid_tokens].reshape(b, g, g)
        off += grid_tokens + 1  # + SEP
        y = src[:, off : off + grid_tokens].reshape(b, g, g)
        off += grid_tokens + 1  # + SEP
        grids.append(x)
        grids.append(y)
    test_x = src[:, off : off + grid_tokens].reshape(b, g, g)
    grids.append(test_x)

    grids_stacked = torch.stack(grids, dim=1).to(torch.long)  # (B, 2*nd+1, g, g)
    tgt_grid = tgt.reshape(b, g, g).unsqueeze(1).to(torch.long)  # (B, 1, g, g)
    all_grids = torch.cat([grids_stacked, tgt_grid], dim=1)  # (B, 2*nd+2, g, g)

    all_grids = _apply_geom_torch(all_grids, codes=geom_codes)
    all_grids = _apply_color_maps_torch(all_grids, maps=color_maps)

    # Split back
    grids_stacked = all_grids[:, : (2 * nd + 1)]
    tgt_grid = all_grids[:, (2 * nd + 1) :, :, :].squeeze(1)  # (B,g,g)

    # Stitch src back, preserving SEP tokens in place.
    out_src = src.clone()
    off = 0
    gi = 0
    for _ in range(nd):
        out_src[:, off : off + grid_tokens] = grids_stacked[:, gi].reshape(b, grid_tokens)
        gi += 1
        off += grid_tokens
        off += 1  # SEP
        out_src[:, off : off + grid_tokens] = grids_stacked[:, gi].reshape(b, grid_tokens)
        gi += 1
        off += grid_tokens
        off += 1  # SEP
    out_src[:, off : off + grid_tokens] = grids_stacked[:, gi].reshape(b, grid_tokens)
    out_tgt = tgt_grid.reshape(b, grid_tokens)
    # Note: SEPs are untouched, but if the original src had bad values outside [0..5],
    # earlier checks would have caught it.
    return out_src, out_tgt

