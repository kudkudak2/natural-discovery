from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# Keep this module standalone (no import from arc_train_utils to avoid cycles).
N_COLORS = 10
SEP_TOKEN = N_COLORS  # 10
PAD_TOKEN = N_COLORS + 1  # 11
VOCAB_SIZE = N_COLORS + 2  # 12


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
    translate_prob: float = 1.0
    # Maximum absolute translation in cells (applied to both axes).
    # -1 means "auto": choose any shift that keeps all non-background (non-zero) pixels in-bounds.
    translate_max: int = -1

    keep_background: bool = True


@dataclass(frozen=True)
class AugmentParams:
    """
    Concrete per-sample augmentation parameters sampled for a batch.
    """

    geom_codes: torch.Tensor  # (B,) long in [0..7]
    color_maps: torch.Tensor  # (B, VOCAB_SIZE) long, token->token
    dy: torch.Tensor  # (B,) long
    dx: torch.Tensor  # (B,) long


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


def apply_shift_np(grid: np.ndarray, *, dy: int, dx: int, fill_value: int = 0) -> np.ndarray:
    """
    Translate a grid by (dy, dx) within the canvas, filling vacated cells with `fill_value`.
    Positive dy shifts content down; positive dx shifts content right.
    """
    _check_square_grid_np(grid)
    g = int(grid.shape[0])
    out = np.full((g, g), int(fill_value), dtype=np.int64)

    dy_i = int(dy)
    dx_i = int(dx)
    src_r0 = max(0, -dy_i)
    src_r1 = min(g, g - dy_i)
    src_c0 = max(0, -dx_i)
    src_c1 = min(g, g - dx_i)
    dst_r0 = src_r0 + dy_i
    dst_r1 = src_r1 + dy_i
    dst_c0 = src_c0 + dx_i
    dst_c1 = src_c1 + dx_i

    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return out
    out[dst_r0:dst_r1, dst_c0:dst_c1] = np.asarray(grid, dtype=np.int64)[src_r0:src_r1, src_c0:src_c1]
    return out


def _bbox_nonzero_np(grids: list[np.ndarray]) -> tuple[int, int, int, int] | None:
    """
    Returns (min_r, max_r, min_c, max_c) over non-zero pixels across all grids, or None if no non-zeros.
    Background is assumed to be color 0.
    """
    mins = []
    maxs = []
    for g in grids:
        gg = np.asarray(g, dtype=np.int64)
        if gg.ndim != 2:
            raise ValueError(f"Expected 2D grid, got shape={gg.shape}")
        rr, cc = np.where(gg != 0)
        if rr.size == 0:
            continue
        mins.append((int(rr.min()), int(cc.min())))
        maxs.append((int(rr.max()), int(cc.max())))
    if len(mins) == 0:
        return None
    min_r = min(r for r, _ in mins)
    min_c = min(c for _, c in mins)
    max_r = max(r for r, _ in maxs)
    max_c = max(c for _, c in maxs)
    return (min_r, max_r, min_c, max_c)


def sample_shift_np(
    *,
    grids: list[np.ndarray],
    rng: np.random.Generator,
    translate_max: int,
) -> tuple[int, int]:
    """
    Sample an in-bounds shift that preserves all non-zero pixels (assumes background=0).
    """
    if len(grids) == 0:
        return 0, 0
    g0 = np.asarray(grids[0])
    _check_square_grid_np(g0)
    g = int(g0.shape[0])
    bbox = _bbox_nonzero_np(grids)
    if bbox is None:
        # All zeros -> any shift is equivalent; keep deterministic/simple.
        return 0, 0
    min_r, max_r, min_c, max_c = bbox
    low_dy = -int(min_r)
    high_dy = int(g - 1 - max_r)
    low_dx = -int(min_c)
    high_dx = int(g - 1 - max_c)
    m = int(translate_max)
    if m >= 0:
        low_dy = max(int(low_dy), -m)
        high_dy = min(int(high_dy), m)
        low_dx = max(int(low_dx), -m)
        high_dx = min(int(high_dx), m)
    if low_dy > high_dy or low_dx > high_dx:
        return 0, 0
    dy = int(rng.integers(int(low_dy), int(high_dy) + 1))
    dx = int(rng.integers(int(low_dx), int(high_dx) + 1))
    return dy, dx


def _identity_color_map_np() -> np.ndarray:
    return np.arange(int(VOCAB_SIZE), dtype=np.int64)


def sample_color_map_np(*, rng: np.random.Generator, keep_background: bool) -> np.ndarray:
    """
    Returns a lookup table `m` of shape (VOCAB_SIZE,) mapping tokens -> tokens.

    - Colors 0..9 are permuted (optionally keeping background=0 fixed).
    - SEP_TOKEN and PAD_TOKEN are always mapped to themselves.
    """
    kb = bool(keep_background)
    m = _identity_color_map_np()
    if kb:
        perm = rng.permutation(np.arange(1, int(N_COLORS), dtype=np.int64))
        m[1:int(N_COLORS)] = perm
    else:
        perm = rng.permutation(np.arange(0, int(N_COLORS), dtype=np.int64))
        m[0:int(N_COLORS)] = perm
    m[int(SEP_TOKEN)] = int(SEP_TOKEN)
    m[int(PAD_TOKEN)] = int(PAD_TOKEN)
    return m


def apply_color_map_np(tokens: np.ndarray, *, color_map: np.ndarray) -> np.ndarray:
    m = np.asarray(color_map, dtype=np.int64).reshape(-1)
    if int(m.shape[0]) != int(VOCAB_SIZE):
        raise ValueError(f"Expected color_map shape ({int(VOCAB_SIZE)},), got {m.shape}")
    t = np.asarray(tokens, dtype=np.int64)
    if np.any((t < 0) | (t > int(PAD_TOKEN))):
        raise ValueError(f"apply_color_map_np expects tokens in [0..{int(PAD_TOKEN)}]")
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
    do_trans = float(rng.random()) < float(spec.translate_prob)

    geom_code = sample_geom_code_np(rng=rng) if do_geom else 0
    color_map = sample_color_map_np(rng=rng, keep_background=bool(spec.keep_background)) if do_color else _identity_color_map_np()
    dy, dx = sample_shift_np(
        grids=[x for xy in demos for x in xy] + [test_in, test_out],
        rng=rng,
        translate_max=int(spec.translate_max),
    ) if do_trans else (0, 0)

    out_demos: list[tuple[np.ndarray, np.ndarray]] = []
    for x, y in demos:
        xx = apply_geom_np(np.asarray(x, dtype=np.int64), code=int(geom_code))
        yy = apply_geom_np(np.asarray(y, dtype=np.int64), code=int(geom_code))
        xx = apply_shift_np(xx, dy=int(dy), dx=int(dx), fill_value=0)
        yy = apply_shift_np(yy, dy=int(dy), dx=int(dx), fill_value=0)
        xx = apply_color_map_np(xx, color_map=color_map)
        yy = apply_color_map_np(yy, color_map=color_map)
        out_demos.append((xx, yy))

    ti = apply_geom_np(np.asarray(test_in, dtype=np.int64), code=int(geom_code))
    to = apply_geom_np(np.asarray(test_out, dtype=np.int64), code=int(geom_code))
    ti = apply_shift_np(ti, dy=int(dy), dx=int(dx), fill_value=0)
    to = apply_shift_np(to, dy=int(dy), dx=int(dx), fill_value=0)
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


_INV_GEOM_CODE_TABLE_CPU: Optional[torch.Tensor] = None


def _inv_geom_code_table_cpu() -> torch.Tensor:
    """
    Returns a tensor `inv` of shape (8,) such that applying code inv[c] inverts code c.
    """
    global _INV_GEOM_CODE_TABLE_CPU
    if _INV_GEOM_CODE_TABLE_CPU is not None:
        return _INV_GEOM_CODE_TABLE_CPU
    device = torch.device("cpu")
    idx = _geom_index_map(size=3, device=device)  # (8, 9)
    inv_codes: list[int] = []
    for c in range(8):
        perm = idx[c]  # out = in[perm]
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(int(perm.numel()), dtype=torch.long)
        found = None
        for c2 in range(8):
            if torch.equal(idx[c2], inv_perm):
                found = c2
                break
        if found is None:
            raise RuntimeError(f"Could not find inverse geom code for code={c}")
        inv_codes.append(int(found))
    _INV_GEOM_CODE_TABLE_CPU = torch.tensor(inv_codes, dtype=torch.long)
    return _INV_GEOM_CODE_TABLE_CPU


def invert_grids_torch(
    grids: torch.Tensor,
    *,
    params: AugmentParams,
) -> torch.Tensor:
    """
    Invert augmentation transforms on grids.

    Forward order is: geom -> shift -> color.
    Inverse order is: inverse color -> inverse shift -> inverse geom.

    grids: (B,N,H,W) long
    """
    if grids.ndim != 4:
        raise ValueError(f"Expected grids shape (B,N,H,W), got {tuple(grids.shape)}")
    b = int(grids.shape[0])
    if params.geom_codes.ndim != 1 or int(params.geom_codes.shape[0]) != b:
        raise ValueError("params.geom_codes must be (B,)")
    if params.color_maps.ndim != 2 or int(params.color_maps.shape[0]) != b or int(params.color_maps.shape[1]) != int(VOCAB_SIZE):
        raise ValueError(f"params.color_maps must be (B,{int(VOCAB_SIZE)})")
    if params.dy.ndim != 1 or params.dx.ndim != 1 or int(params.dy.shape[0]) != b or int(params.dx.shape[0]) != b:
        raise ValueError("params.dy/dx must be (B,)")

    device = grids.device
    inv_maps = torch.empty_like(params.color_maps)
    inv_maps.scatter_(1, params.color_maps.to(device=device), torch.arange(int(VOCAB_SIZE), device=device, dtype=torch.long).unsqueeze(0).expand(b, -1))
    out = _apply_color_maps_torch(grids, maps=inv_maps)
    out = _apply_shifts_torch(out, dy=-params.dy.to(device=device), dx=-params.dx.to(device=device))
    inv_tbl = _inv_geom_code_table_cpu().to(device=device)
    inv_codes = inv_tbl.index_select(0, params.geom_codes.to(device=device).clamp(min=0, max=7))
    out = _apply_geom_torch(out, codes=inv_codes)
    return out


def _sample_color_maps_torch(
    *,
    batch_size: int,
    generator: Optional[torch.Generator],
    device: torch.device,
    keep_background: bool,
) -> torch.Tensor:
    """
    Returns (B, VOCAB_SIZE) lookup tables mapping tokens -> tokens.
    """
    b = int(batch_size)
    if b <= 0:
        raise ValueError(f"batch_size must be >= 1, got {b}")
    kb = bool(keep_background)
    maps = torch.empty((b, int(VOCAB_SIZE)), device=device, dtype=torch.long)
    maps[:, int(SEP_TOKEN)] = int(SEP_TOKEN)
    maps[:, int(PAD_TOKEN)] = int(PAD_TOKEN)
    if kb:
        maps[:, 0] = 0
        # Permute [1..9] per-sample by sorting random scores.
        scores = torch.rand((b, int(N_COLORS - 1)), generator=generator, device=device)
        order = torch.argsort(scores, dim=1)  # (B,9) in [0..8]
        maps[:, 1:int(N_COLORS)] = (order + 1).to(torch.long)
    else:
        scores = torch.rand((b, int(N_COLORS)), generator=generator, device=device)
        order = torch.argsort(scores, dim=1)  # (B,10) in [0..9]
        maps[:, 0:int(N_COLORS)] = order.to(torch.long)
    return maps


def _sample_shifts_torch(
    *,
    grids: torch.Tensor,
    generator: Optional[torch.Generator],
    translate_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample per-sample (dy, dx) shifts that keep all non-background pixels in-bounds.

    Background is assumed to be color 0. Shifts are computed from the union bbox across all grids
    provided for each sample (typically includes demo x/y + test_x + target).
    """
    if grids.ndim != 4:
        raise ValueError(f"Expected grids shape (B,N,H,W), got {tuple(grids.shape)}")
    b, _n, g, w = int(grids.shape[0]), int(grids.shape[1]), int(grids.shape[2]), int(grids.shape[3])
    if g != w:
        raise ValueError(f"Expected square grids, got HxW={g}x{w}")
    device = grids.device

    mask = grids != 0  # (B,N,g,g)
    any_mask = mask.any(dim=(1, 2, 3))
    big = torch.tensor(g, device=device, dtype=torch.long)

    row_idx = torch.arange(g, device=device, dtype=torch.long).view(1, 1, g, 1)
    col_idx = torch.arange(g, device=device, dtype=torch.long).view(1, 1, 1, g)

    min_r = torch.where(mask, row_idx, big).amin(dim=(1, 2, 3))
    max_r = torch.where(mask, row_idx, torch.zeros((), device=device, dtype=torch.long)).amax(dim=(1, 2, 3))
    min_c = torch.where(mask, col_idx, big).amin(dim=(1, 2, 3))
    max_c = torch.where(mask, col_idx, torch.zeros((), device=device, dtype=torch.long)).amax(dim=(1, 2, 3))

    # If empty (all background), force bbox to a single point so we produce dy=dx=0 deterministically.
    min_r = torch.where(any_mask, min_r, torch.zeros_like(min_r))
    max_r = torch.where(any_mask, max_r, torch.zeros_like(max_r))
    min_c = torch.where(any_mask, min_c, torch.zeros_like(min_c))
    max_c = torch.where(any_mask, max_c, torch.zeros_like(max_c))

    low_dy = -min_r
    high_dy = (g - 1) - max_r
    low_dx = -min_c
    high_dx = (g - 1) - max_c

    m = int(translate_max)
    if m >= 0:
        low_dy = torch.maximum(low_dy, torch.full_like(low_dy, -m))
        high_dy = torch.minimum(high_dy, torch.full_like(high_dy, m))
        low_dx = torch.maximum(low_dx, torch.full_like(low_dx, -m))
        high_dx = torch.minimum(high_dx, torch.full_like(high_dx, m))

    # Guard: if bounds are invalid, clamp to 0 shift.
    ok_y = low_dy <= high_dy
    ok_x = low_dx <= high_dx

    def sample_int(low: torch.Tensor, high: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        span = (high - low + 1).clamp(min=1).to(torch.long)
        u = torch.rand((b,), device=device, generator=generator)
        offs = torch.floor(u * span.to(torch.float32)).to(torch.long)
        v = low + offs
        return torch.where(ok, v, torch.zeros_like(v))

    dy = sample_int(low_dy, high_dy, ok_y & any_mask)
    dx = sample_int(low_dx, high_dx, ok_x & any_mask)
    return dy, dx


def _apply_shifts_torch(grids: torch.Tensor, *, dy: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """
    Apply per-sample integer translations to `grids`, filling with background color 0.

    grids: (B,N,H,W) long
    dy/dx: (B,) long
    """
    if grids.ndim != 4:
        raise ValueError(f"Expected grids shape (B,N,H,W), got {tuple(grids.shape)}")
    b, n, g, w = int(grids.shape[0]), int(grids.shape[1]), int(grids.shape[2]), int(grids.shape[3])
    if g != w:
        raise ValueError(f"Expected square grids, got HxW={g}x{w}")
    if dy.ndim != 1 or dx.ndim != 1 or int(dy.shape[0]) != b or int(dx.shape[0]) != b:
        raise ValueError(f"Expected dy/dx shape (B,), got dy={tuple(dy.shape)} dx={tuple(dx.shape)} for B={b}")

    device = grids.device
    rr = torch.arange(g, device=device, dtype=torch.long).view(1, g, 1).expand(b, g, g)
    cc = torch.arange(g, device=device, dtype=torch.long).view(1, 1, g).expand(b, g, g)
    src_r = rr - dy.view(b, 1, 1)
    src_c = cc - dx.view(b, 1, 1)
    valid = (src_r >= 0) & (src_r < g) & (src_c >= 0) & (src_c < g)
    src_lin = (src_r.clamp(0, g - 1) * g + src_c.clamp(0, g - 1)).reshape(b, g * g)

    flat = grids.reshape(b, n, g * g)
    gather_idx = src_lin.unsqueeze(1).expand(b, n, g * g)
    out = torch.gather(flat, dim=2, index=gather_idx)
    out = out * valid.reshape(b, 1, g * g).to(out.dtype)
    return out.reshape(b, n, g, g)


def _apply_color_maps_torch(tokens: torch.Tensor, *, maps: torch.Tensor) -> torch.Tensor:
    """
    Apply per-sample token remapping.

    tokens: (B, ...) long with values in [0..PAD_TOKEN]
    maps: (B, VOCAB_SIZE) long
    """
    if maps.ndim != 2 or int(maps.shape[1]) != int(VOCAB_SIZE):
        raise ValueError(f"Expected maps shape (B,{int(VOCAB_SIZE)}), got {tuple(maps.shape)}")
    if tokens.ndim < 1:
        raise ValueError("tokens must have at least 1 dimension")
    b = int(tokens.shape[0])
    if int(maps.shape[0]) != b:
        raise ValueError(f"maps batch mismatch: maps B={int(maps.shape[0])} vs tokens B={b}")
    if tokens.dtype != torch.long:
        tokens = tokens.to(torch.long)
    # Avoid a device sync in the training hot-path: only validate token range on CPU.
    if tokens.device.type == "cpu":
        if int(tokens.min().item()) < 0 or int(tokens.max().item()) > int(PAD_TOKEN):
            raise ValueError(f"_apply_color_maps_torch expects tokens in [0..{int(PAD_TOKEN)}]")
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
      src: (B, T) long tokens in [0..PAD_TOKEN]
      tgt: (B, grid_tokens) long tokens in [0..9]
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
        color_maps = torch.arange(int(VOCAB_SIZE), device=device, dtype=torch.long).unsqueeze(0).repeat(b, 1)
    else:
        apply_col = torch.rand((b,), device=device, generator=generator) < float(p_col)
        rand_maps = _sample_color_maps_torch(
            batch_size=b,
            generator=generator,
            device=device,
            keep_background=bool(spec.keep_background),
        )
        ident = torch.arange(int(VOCAB_SIZE), device=device, dtype=torch.long).unsqueeze(0).repeat(b, 1)
        color_maps = torch.where(apply_col.unsqueeze(1), rand_maps, ident)

    # translate: per-sample (dy,dx), or (0,0) when not applied
    p_tr = float(spec.translate_prob)
    if p_tr <= 0.0:
        dy = torch.zeros((b,), device=device, dtype=torch.long)
        dx = torch.zeros((b,), device=device, dtype=torch.long)
    else:
        apply_tr = torch.rand((b,), device=device, generator=generator) < float(p_tr)
        # shift bounds are computed from the union bbox across src grids + (masked) tgt.
        # For targets, treat invalid cells (-100) as background 0 when computing bbox.
        # We will restore the invalid mask after transformations.
        # (We'll compute dy/dx after parsing grids below.)
        dy = torch.zeros((b,), device=device, dtype=torch.long)
        dx = torch.zeros((b,), device=device, dtype=torch.long)

    # --- parse src into grids (nd*(x,y) + test_x), apply transforms, then stitch back ---
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
    tgt_valid = (tgt != -100).reshape(b, g, g)  # (B,g,g) bool
    tgt_filled = torch.where(tgt_valid, tgt.reshape(b, g, g), torch.zeros((b, g, g), device=device, dtype=torch.long))
    tgt_grid = tgt_filled.unsqueeze(1).to(torch.long)  # (B, 1, g, g)
    all_grids = torch.cat([grids_stacked, tgt_grid], dim=1)  # (B, 2*nd+2, g, g)

    all_grids = _apply_geom_torch(all_grids, codes=geom_codes)

    if p_tr > 0.0:
        dy_s, dx_s = _sample_shifts_torch(grids=all_grids, generator=generator, translate_max=int(spec.translate_max))
        dy = torch.where(apply_tr, dy_s, torch.zeros_like(dy_s))
        dx = torch.where(apply_tr, dx_s, torch.zeros_like(dx_s))
        all_grids = _apply_shifts_torch(all_grids, dy=dy, dx=dx)

    all_grids = _apply_color_maps_torch(all_grids, maps=color_maps)

    # Split back
    grids_stacked = all_grids[:, : (2 * nd + 1)]
    tgt_grid = all_grids[:, (2 * nd + 1) :, :, :].squeeze(1)  # (B,g,g)

    # Restore target ignore mask after transforms by transforming the mask itself.
    tgt_valid_t = tgt_valid.unsqueeze(1).to(torch.long)  # (B,1,g,g)
    tgt_valid_t = _apply_geom_torch(tgt_valid_t, codes=geom_codes)
    if p_tr > 0.0:
        tgt_valid_t = _apply_shifts_torch(tgt_valid_t, dy=dy, dx=dx)
    tgt_valid_t = tgt_valid_t.squeeze(1).to(torch.bool)
    out_tgt = torch.where(tgt_valid_t.reshape(b, grid_tokens), tgt_grid.reshape(b, grid_tokens), torch.full((b, grid_tokens), -100, device=device, dtype=torch.long))

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
    # Note: SEPs/PADs are untouched, but if the original src had bad values outside [0..PAD_TOKEN],
    # earlier checks would have caught it.
    return out_src, out_tgt


def augment_src_tgt_batch_with_params(
    *,
    src: torch.Tensor,
    tgt: torch.Tensor,
    grid_size: int,
    num_demos: int,
    generator: Optional[torch.Generator],
    spec: AugmentSpec,
) -> tuple[torch.Tensor, torch.Tensor, AugmentParams]:
    """
    Same as `augment_src_tgt_batch`, but also returns the sampled per-sample transform params.
    Intended for test-time augmentation / voting.
    """
    if not bool(spec.enabled):
        zeros = torch.zeros((int(src.shape[0]),), device=src.device, dtype=torch.long)
        ident = torch.arange(int(VOCAB_SIZE), device=src.device, dtype=torch.long).unsqueeze(0).repeat(int(src.shape[0]), 1)
        return src, tgt, AugmentParams(geom_codes=zeros, color_maps=ident, dy=zeros, dx=zeros)

    # Reuse the exact logic by inlining the sampling portions from augment_src_tgt_batch.
    if src.ndim != 2:
        raise ValueError(f"Expected src shape (B,T), got {tuple(src.shape)}")
    b = int(src.shape[0])
    device = src.device
    g = int(grid_size)
    nd = int(num_demos)
    grid_tokens = int(g * g)

    # --- sample which transforms to apply ---
    p_geom = float(spec.geom_prob)
    if p_geom <= 0.0:
        geom_codes = torch.zeros((b,), device=device, dtype=torch.long)
    else:
        apply_geom = torch.rand((b,), device=device, generator=generator) < float(p_geom)
        codes = torch.randint(0, 8, (b,), device=device, generator=generator, dtype=torch.long)
        geom_codes = torch.where(apply_geom, codes, torch.zeros_like(codes))

    p_col = float(spec.color_prob)
    if p_col <= 0.0:
        color_maps = torch.arange(int(VOCAB_SIZE), device=device, dtype=torch.long).unsqueeze(0).repeat(b, 1)
    else:
        apply_col = torch.rand((b,), device=device, generator=generator) < float(p_col)
        rand_maps = _sample_color_maps_torch(
            batch_size=b,
            generator=generator,
            device=device,
            keep_background=bool(spec.keep_background),
        )
        ident = torch.arange(int(VOCAB_SIZE), device=device, dtype=torch.long).unsqueeze(0).repeat(b, 1)
        color_maps = torch.where(apply_col.unsqueeze(1), rand_maps, ident)

    p_tr = float(spec.translate_prob)
    apply_tr = torch.zeros((b,), device=device, dtype=torch.bool)
    if p_tr > 0.0:
        apply_tr = torch.rand((b,), device=device, generator=generator) < float(p_tr)
    dy = torch.zeros((b,), device=device, dtype=torch.long)
    dx = torch.zeros((b,), device=device, dtype=torch.long)

    # --- parse src into grids + tgt (mask-aware) ---
    grids: list[torch.Tensor] = []
    off = 0
    for _ in range(nd):
        x = src[:, off : off + grid_tokens].reshape(b, g, g)
        off += grid_tokens + 1
        y = src[:, off : off + grid_tokens].reshape(b, g, g)
        off += grid_tokens + 1
        grids.append(x)
        grids.append(y)
    test_x = src[:, off : off + grid_tokens].reshape(b, g, g)
    grids.append(test_x)

    grids_stacked = torch.stack(grids, dim=1).to(torch.long)  # (B, 2*nd+1, g, g)
    tgt_valid = (tgt != -100).reshape(b, g, g)
    tgt_filled = torch.where(tgt_valid, tgt.reshape(b, g, g), torch.zeros((b, g, g), device=device, dtype=torch.long))
    tgt_grid = tgt_filled.unsqueeze(1).to(torch.long)
    all_grids = torch.cat([grids_stacked, tgt_grid], dim=1)  # (B, 2*nd+2, g, g)

    all_grids = _apply_geom_torch(all_grids, codes=geom_codes)
    if p_tr > 0.0:
        dy_s, dx_s = _sample_shifts_torch(grids=all_grids, generator=generator, translate_max=int(spec.translate_max))
        dy = torch.where(apply_tr, dy_s, torch.zeros_like(dy_s))
        dx = torch.where(apply_tr, dx_s, torch.zeros_like(dx_s))
        all_grids = _apply_shifts_torch(all_grids, dy=dy, dx=dx)
    all_grids = _apply_color_maps_torch(all_grids, maps=color_maps)

    grids_stacked = all_grids[:, : (2 * nd + 1)]
    tgt_grid = all_grids[:, (2 * nd + 1) :, :, :].squeeze(1)

    tgt_valid_t = tgt_valid.unsqueeze(1).to(torch.long)
    tgt_valid_t = _apply_geom_torch(tgt_valid_t, codes=geom_codes)
    if p_tr > 0.0:
        tgt_valid_t = _apply_shifts_torch(tgt_valid_t, dy=dy, dx=dx)
    tgt_valid_t = tgt_valid_t.squeeze(1).to(torch.bool)
    out_tgt = torch.where(
        tgt_valid_t.reshape(b, grid_tokens),
        tgt_grid.reshape(b, grid_tokens),
        torch.full((b, grid_tokens), -100, device=device, dtype=torch.long),
    )

    out_src = src.clone()
    off = 0
    gi = 0
    for _ in range(nd):
        out_src[:, off : off + grid_tokens] = grids_stacked[:, gi].reshape(b, grid_tokens)
        gi += 1
        off += grid_tokens + 1
        out_src[:, off : off + grid_tokens] = grids_stacked[:, gi].reshape(b, grid_tokens)
        gi += 1
        off += grid_tokens + 1
    out_src[:, off : off + grid_tokens] = grids_stacked[:, gi].reshape(b, grid_tokens)

    params = AugmentParams(geom_codes=geom_codes, color_maps=color_maps, dy=dy, dx=dx)
    return out_src, out_tgt, params

