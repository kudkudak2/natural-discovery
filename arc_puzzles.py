from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class OODSpec:
    """Controls in-distribution vs out-of-distribution input complexity."""

    # NOTE: When we increase the number of "instances" (e.g. scattered pixels) in OOD,
    # we cap it to at most 2x the IID maximum (see Puzzle._num_pixels).
    #
    # Defaults are conservative for 5x5 grids.
    min_pixels_id: int = 1
    max_pixels_id: int = 2
    min_pixels_ood: int = 2
    max_pixels_ood: int = 4


@dataclass(frozen=True)
class ShrinkPerturbSpec:
    """
    Optional input-space augmentation applied *before* computing outputs.

    Rationale: applying shrink/translation to the input grid and then recomputing the
    puzzle output preserves correctness (y = apply(x, rule_color)).
    """

    enabled: bool = False
    prob: float = 1.0
    shrink_min: float = 0.6
    shrink_max: float = 0.9
    shift_max: int = 1


class Puzzle:
    """
    Base ARC-style puzzle.

    Contract:
    - Input grid uses colors in {0..4} where 0 is background.
    - Some puzzles use a hidden `rule_color` that must be inferred from demos.
    """

    skill_id: int
    name: str
    uses_rule_color: bool

    def __init__(
        self,
        *,
        size: int = 5,
        colors: Sequence[int] = (1, 2, 3, 4),
        ood_spec: OODSpec = OODSpec(),
        shrink_perturb: Optional[ShrinkPerturbSpec] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.size = int(size)
        if self.size < 3:
            # Many skills rely on having an interior (e.g. "1 cell away from wall", rectangles with interior, etc.)
            raise ValueError(f"Grid size must be >= 3, got size={self.size}")
        self.colors = tuple(int(c) for c in colors)
        self.ood_spec = ood_spec
        self.rng = rng if rng is not None else np.random.default_rng()
        self.shrink_perturb = shrink_perturb

    def blank(self) -> np.ndarray:
        return np.zeros((self.size, self.size), dtype=np.int64)

    @property
    def variant_id(self) -> Optional[str]:
        """
        Optional per-task latent variant identifier.

        Example: Skills with a per-task explosion kernel return the chosen kernel mode.
        Default: None (no variant).
        """
        return None

    def variant_params(self) -> dict[str, object]:
        """
        Optional per-task latent parameters (JSON-serializable) for analysis/repro.

        Default: {}.
        """
        return {}

    def _num_pixels(self, *, ood: bool) -> int:
        """
        Draw a number of pixels/instances.

        Rule: OOD is allowed to be harder, but we cap the OOD maximum to at most
        2x the IID maximum to prevent extreme distribution shifts.
        """
        lo_id = int(self.ood_spec.min_pixels_id)
        hi_id = int(self.ood_spec.max_pixels_id)
        if hi_id < lo_id:
            hi_id = lo_id

        if not ood:
            return int(self.rng.integers(lo_id, hi_id + 1))

        cap_hi = int(2 * hi_id)
        lo_ood = int(self.ood_spec.min_pixels_ood)
        hi_ood = int(self.ood_spec.max_pixels_ood)
        if hi_ood < lo_ood:
            hi_ood = lo_ood

        # Enforce the 2x cap (also clamps the min if it's above the cap).
        lo = min(max(lo_ood, lo_id), cap_hi)
        hi = min(hi_ood, cap_hi)
        if hi < lo:
            hi = lo
        return int(self.rng.integers(lo, hi + 1))

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        raise NotImplementedError

    def make_input(self, *, ood: bool) -> np.ndarray:
        raise NotImplementedError

    def generate_single_example(self, *, rule_color: Optional[int], ood: bool) -> tuple[np.ndarray, np.ndarray]:
        inp = self.make_input(ood=ood)
        inp = _maybe_shrink_and_perturb(inp, rng=self.rng, spec=self.shrink_perturb)
        out = self.apply(inp, rule_color)
        return inp, out

    def generate_prompt(
        self,
        *,
        num_demos: int = 3,
        ood_test: bool = False,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray], Optional[int]]:
        rule_color: Optional[int] = None
        if self.uses_rule_color:
            rule_color = int(self.rng.choice(self.colors))

        demos: list[tuple[np.ndarray, np.ndarray]] = [
            self.generate_single_example(rule_color=rule_color, ood=False) for _ in range(int(num_demos))
        ]
        test_pair = self.generate_single_example(rule_color=rule_color, ood=bool(ood_test))
        return demos, test_pair, rule_color


def _bbox_nonzero(grid: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    rs, cs = np.nonzero(grid)
    if rs.size == 0:
        return None
    r0 = int(rs.min())
    r1 = int(rs.max())
    c0 = int(cs.min())
    c1 = int(cs.max())
    return r0, r1, c0, c1


def _resize_nearest(patch: np.ndarray, *, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = int(patch.shape[0]), int(patch.shape[1])
    out_h_i, out_w_i = int(out_h), int(out_w)
    if out_h_i <= 0 or out_w_i <= 0:
        raise ValueError(f"Invalid resize target: {(out_h_i, out_w_i)}")
    if in_h <= 0 or in_w <= 0:
        raise ValueError(f"Invalid resize source: {(in_h, in_w)}")
    if in_h == out_h_i and in_w == out_w_i:
        return patch.copy()

    # Nearest-neighbor sampling indices.
    rr = (np.linspace(0, in_h - 1, out_h_i)).round().astype(np.int64)
    cc = (np.linspace(0, in_w - 1, out_w_i)).round().astype(np.int64)
    return patch[rr[:, None], cc[None, :]].astype(np.int64, copy=False)


def _maybe_shrink_and_perturb(
    grid: np.ndarray, *, rng: np.random.Generator, spec: Optional[ShrinkPerturbSpec]
) -> np.ndarray:
    if spec is None or not bool(spec.enabled):
        return grid
    p = float(spec.prob)
    if p <= 0.0:
        return grid
    if p < 1.0 and float(rng.random()) >= p:
        return grid

    size = int(grid.shape[0])
    if grid.ndim != 2 or int(grid.shape[1]) != size:
        raise ValueError(f"Expected square 2D grid, got shape={grid.shape}")

    bbox = _bbox_nonzero(grid)
    if bbox is None:
        return grid
    r0, r1, c0, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]

    smin = float(spec.shrink_min)
    smax = float(spec.shrink_max)
    if not (0.0 < smin <= smax <= 1.0):
        raise ValueError(f"shrink_min/max must satisfy 0 < min <= max <= 1, got {smin}, {smax}")
    scale = float(rng.uniform(smin, smax))

    h, w = int(patch.shape[0]), int(patch.shape[1])
    new_h = max(1, int(np.rint(h * scale)))
    new_w = max(1, int(np.rint(w * scale)))
    resized = _resize_nearest(patch, out_h=new_h, out_w=new_w)

    shift_max = int(spec.shift_max)
    if shift_max < 0:
        raise ValueError(f"shift_max must be >= 0, got {shift_max}")
    dr = int(rng.integers(-shift_max, shift_max + 1)) if shift_max > 0 else 0
    dc = int(rng.integers(-shift_max, shift_max + 1)) if shift_max > 0 else 0

    # Keep the object roughly in its original region; clamp to bounds.
    base_r0 = int(r0) + dr
    base_c0 = int(c0) + dc
    max_r0 = int(size - new_h)
    max_c0 = int(size - new_w)
    rr0 = int(min(max(base_r0, 0), max_r0))
    cc0 = int(min(max(base_c0, 0), max_c0))

    out = np.zeros_like(grid)
    out[rr0 : rr0 + new_h, cc0 : cc0 + new_w] = resized
    return out


def apply_gravity(grid: np.ndarray) -> np.ndarray:
    """Drops non-zero cells in each column to the bottom (stable)."""
    size = grid.shape[0]
    new_grid = np.zeros_like(grid)
    for col in range(size):
        pixels = grid[:, col][grid[:, col] != 0]
        if pixels.size == 0:
            continue
        new_grid[size - pixels.size : size, col] = pixels
    return new_grid


def apply_bottom_color_change(grid: np.ndarray, target_color: int) -> np.ndarray:
    """Changes the bottom-row non-zero cells to `target_color`."""
    new_grid = grid.copy()
    bottom = new_grid[-1, :]
    bottom[bottom != 0] = int(target_color)
    new_grid[-1, :] = bottom
    return new_grid


def apply_right_gravity(grid: np.ndarray) -> np.ndarray:
    """Slides non-zero cells in each row to the right (stable)."""
    size = grid.shape[1]
    new_grid = np.zeros_like(grid)
    for row in range(grid.shape[0]):
        pixels = grid[row, :][grid[row, :] != 0]
        if pixels.size == 0:
            continue
        new_grid[row, size - pixels.size : size] = pixels
    return new_grid


class Skill1Gravity(Puzzle):
    skill_id = 1
    name = "gravity"
    uses_rule_color = False

    def make_input(self, *, ood: bool) -> np.ndarray:
        grid = self.blank()
        n = self._num_pixels(ood=ood)
        # Avoid bottom row so the effect is visible.
        for _ in range(n * 3):
            if n == 0:
                break
            r = int(self.rng.integers(0, self.size - 1))
            c = int(self.rng.integers(0, self.size))
            if grid[r, c] == 0:
                grid[r, c] = 1
                n -= 1
            if n == 0:
                break
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        return apply_gravity(grid)


class Skill2BottomRecolor(Puzzle):
    skill_id = 2
    name = "bottom_recolor"
    uses_rule_color = True

    def make_input(self, *, ood: bool) -> np.ndarray:
        grid = self.blank()
        # Skill 2 should show squares across multiple rows, but the *rule* only recolors
        # the bottom row. We therefore add non-bottom "distractor" pixels that remain unchanged.

        # Ensure at least one bottom-row pixel so the recoloring is observable.
        bottom_k = int(self.rng.integers(1, self.size + 1))
        bottom_cols = self.rng.choice(np.arange(self.size), size=bottom_k, replace=False)
        grid[-1, bottom_cols] = 1

        # Add a few pixels above the bottom row; OOD means "more clutter", not a different rule.
        #
        # Cap OOD clutter count to at most 2x the IID maximum (IID max is size-1, because bottom row is excluded).
        if not ood:
            top_n = int(self.rng.integers(1, self.size))
        else:
            iid_max = max(1, int(self.size) - 1)
            ood_max = min(int(2 * iid_max), int(self.size) * int(self.size))
            ood_min = min(int(self.size), ood_max)
            top_n = int(self.rng.integers(ood_min, ood_max + 1))
        placed = 0
        attempts = 0
        while placed < top_n and attempts < top_n * 10:
            attempts += 1
            r = int(self.rng.integers(0, self.size - 1))  # exclude bottom row
            c = int(self.rng.integers(0, self.size))
            if grid[r, c] == 0:
                grid[r, c] = 1
                placed += 1
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        assert rule_color is not None
        return apply_bottom_color_change(grid, rule_color)


class Skill3GravityThenRecolor(Puzzle):
    skill_id = 3
    name = "gravity_then_recolor"
    uses_rule_color = True

    def make_input(self, *, ood: bool) -> np.ndarray:
        # Same generator as skill1: scattered pixels.
        grid = self.blank()
        n = self._num_pixels(ood=ood)
        for _ in range(n * 3):
            if n == 0:
                break
            r = int(self.rng.integers(0, self.size - 1))
            c = int(self.rng.integers(0, self.size))
            if grid[r, c] == 0:
                grid[r, c] = 1
                n -= 1
            if n == 0:
                break
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        assert rule_color is not None
        return apply_bottom_color_change(apply_gravity(grid), rule_color)


class Skill4PlaceCenterDot(Puzzle):
    skill_id = 4
    name = "place_center_dot"
    uses_rule_color = True

    def make_input(self, *, ood: bool) -> np.ndarray:
        """
        Create a single solid rectangle (color=1). This makes the "center" unambiguous.

        ID: smaller rectangles, OOD: larger rectangles.
        """
        grid = self.blank()

        if ood:
            h = int(self.rng.integers(max(2, self.size - 1), self.size + 1))  # 4-5 when size=5
            w = int(self.rng.integers(max(2, self.size - 1), self.size + 1))
        else:
            h = int(self.rng.integers(2, min(4, self.size) + 1))  # 2-4
            w = int(self.rng.integers(2, min(4, self.size) + 1))

        r0 = int(self.rng.integers(0, self.size - h + 1))
        c0 = int(self.rng.integers(0, self.size - w + 1))
        grid[r0 : r0 + h, c0 : c0 + w] = 1
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        assert rule_color is not None
        """
        Recolor the center cell(s) of the object's bounding box with `rule_color`.

        - If bbox height/width is odd -> single center cell.
        - If even -> 2 center rows and/or 2 center cols, yielding a 2x2 / 2x1 / 1x2 center region.
        """
        out = grid.copy()
        rs, cs = np.nonzero(grid)
        if rs.size == 0:
            return out

        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        if h % 2 == 1:
            center_rows = [r0 + h // 2]
        else:
            center_rows = [r0 + (h // 2) - 1, r0 + (h // 2)]

        if w % 2 == 1:
            center_cols = [c0 + w // 2]
        else:
            center_cols = [c0 + (w // 2) - 1, c0 + (w // 2)]

        for r in center_rows:
            for c in center_cols:
                out[r, c] = int(rule_color)
        return out


class Skill5FillRectangleInterior(Puzzle):
    skill_id = 5
    name = "fill_rectangle_interior"
    uses_rule_color = True

    def make_input(self, *, ood: bool) -> np.ndarray:
        _ = ood
        grid = self.blank()
        # Create a rectangular frame (color 1), min size 3x3 so there's an interior.
        h = int(self.rng.integers(3, self.size + 1))
        w = int(self.rng.integers(3, self.size + 1))
        r0 = int(self.rng.integers(0, self.size - h + 1))
        c0 = int(self.rng.integers(0, self.size - w + 1))
        r1 = r0 + h - 1
        c1 = c0 + w - 1

        grid[r0, c0 : c1 + 1] = 1
        grid[r1, c0 : c1 + 1] = 1
        grid[r0 : r1 + 1, c0] = 1
        grid[r0 : r1 + 1, c1] = 1
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        assert rule_color is not None
        out = grid.copy()
        rs, cs = np.nonzero(grid)
        if rs.size == 0:
            return out
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        if (r1 - r0) < 2 or (c1 - c0) < 2:
            return out
        out[r0 + 1 : r1, c0 + 1 : c1] = int(rule_color)
        return out


class Skill6RightGravity(Puzzle):
    skill_id = 6
    name = "right_gravity"
    uses_rule_color = False

    def make_input(self, *, ood: bool) -> np.ndarray:
        grid = self.blank()
        n = self._num_pixels(ood=ood)
        # Avoid rightmost col so the effect is visible.
        for _ in range(n * 3):
            if n == 0:
                break
            r = int(self.rng.integers(0, self.size))
            c = int(self.rng.integers(0, self.size - 1))
            if grid[r, c] == 0:
                grid[r, c] = 1
                n -= 1
            if n == 0:
                break
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        return apply_right_gravity(grid)


class Skill7DownThenRight(Puzzle):
    skill_id = 7
    name = "down_then_right"
    uses_rule_color = False

    def make_input(self, *, ood: bool) -> np.ndarray:
        # Same style as Skill 1/6: scattered pixels.
        grid = self.blank()
        n = self._num_pixels(ood=ood)
        for _ in range(n * 3):
            if n == 0:
                break
            r = int(self.rng.integers(0, self.size - 1))
            c = int(self.rng.integers(0, self.size - 1))
            if grid[r, c] == 0:
                grid[r, c] = 1
                n -= 1
            if n == 0:
                break
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        return apply_right_gravity(apply_gravity(grid))


def _ray_color_from_source(src_color: int) -> int:
    """
    Deterministic "different color" mapping without a hidden rule:
      1 -> 2 -> 3 -> 4 -> 1
    """
    c = int(src_color)
    if c not in (1, 2, 3, 4):
        return 2
    return 1 + (c % 4)


def _shoot_left_ray(grid: np.ndarray, *, row: int, from_col: int, ray_color: int) -> None:
    """
    Draw a horizontal ray to the left wall on the given row, excluding the cell at `from_col`.
    Only fills background cells (0) to keep objects intact.
    """
    if from_col <= 0:
        return
    for c in range(0, from_col):
        if grid[row, c] == 0:
            grid[row, c] = int(ray_color)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (r0, r1, c0, c1) bbox for a boolean mask; assumes mask has at least one True."""
    rs, cs = np.nonzero(mask)
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    return r0, r1, c0, c1


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Axis-aligned rectangle overlap check for (r0,r1,c0,c1) inclusive coords."""
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    if ar1 < br0 or br1 < ar0:
        return False
    if ac1 < bc0 or bc1 < ac0:
        return False
    return True


def _try_place_solid_rect(
    grid: np.ndarray,
    *,
    color: int,
    h: int,
    w: int,
    rng: np.random.Generator,
    max_tries: int = 200,
) -> Optional[tuple[int, int, int, int]]:
    """
    Try to place a solid h×w rectangle of `color` into background-only cells (0).
    Returns (r0, r1, c0, c1) inclusive coords if placed, else None.
    """
    size_r, size_c = int(grid.shape[0]), int(grid.shape[1])
    h = int(h)
    w = int(w)
    if h <= 0 or w <= 0:
        return None
    if h > size_r or w > size_c:
        return None

    for _ in range(int(max_tries)):
        r0 = int(rng.integers(0, size_r - h + 1))
        c0 = int(rng.integers(0, size_c - w + 1))
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != 0):
            continue
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        return (r0, r1, c0, c1)
    return None


def _count_components_4(mask: np.ndarray) -> int:
    """Count 4-connected components in a boolean mask."""
    return _count_components(mask, connectivity=4)


def _components(mask: np.ndarray, *, connectivity: int) -> list[list[tuple[int, int]]]:
    """
    Return connected components for a boolean mask as lists of (r,c) coords.
    Connectivity ∈ {4,8}.
    """
    conn = int(connectivity)
    if conn not in (4, 8):
        raise ValueError(f"Unsupported connectivity={conn}; expected 4 or 8")

    if conn == 4:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neigh = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    h, w = int(mask.shape[0]), int(mask.shape[1])
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(visited[r, c]):
                continue
            stack = [(int(r), int(c))]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in neigh:
                    nr = cr + int(dr)
                    nc = cc + int(dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(visited[nr, nc]):
                        visited[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _components_sorted(mask: np.ndarray, *, connectivity: int) -> list[list[tuple[int, int]]]:
    """Connected components sorted by (min_row, min_col) reading order."""
    comps = _components(mask, connectivity=int(connectivity))

    def key(comp: list[tuple[int, int]]) -> tuple[int, int]:
        rs = [r for r, _c in comp]
        cs = [c for _r, c in comp]
        return (int(min(rs)), int(min(cs)))

    comps.sort(key=key)
    return comps


def _count_components(mask: np.ndarray, *, connectivity: int) -> int:
    """Count connected components in a boolean mask (connectivity ∈ {4,8})."""
    conn = int(connectivity)
    if conn not in (4, 8):
        raise ValueError(f"Unsupported connectivity={conn}; expected 4 or 8")

    if conn == 4:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neigh = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    h, w = int(mask.shape[0]), int(mask.shape[1])
    visited = np.zeros_like(mask, dtype=bool)
    count = 0

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(visited[r, c]):
                continue
            count += 1
            stack = [(int(r), int(c))]
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for dr, dc in neigh:
                    nr = cr + int(dr)
                    nc = cc + int(dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(visited[nr, nc]):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
    return int(count)


def _component_counts_by_color(grid: np.ndarray, *, connectivity: int = 4) -> dict[int, int]:
    """Return {color: n_components} for each non-zero color in the grid."""
    colors_present = sorted({int(c) for c in grid.flatten().tolist() if int(c) != 0})
    out: dict[int, int] = {}
    conn = int(connectivity)
    for c in colors_present:
        out[int(c)] = int(_count_components(grid == int(c), connectivity=conn))
    return out


def _rect_touches_color(
    grid: np.ndarray,
    *,
    color: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    connectivity: int,
) -> bool:
    """
    Returns True if a candidate solid rect at [r0:r1, c0:c1] (inclusive) would be adjacent
    (per connectivity) to an existing cell of `color` in the grid.
    """
    conn = int(connectivity)
    if conn not in (4, 8):
        raise ValueError(f"Unsupported connectivity={conn}; expected 4 or 8")

    h, w = int(grid.shape[0]), int(grid.shape[1])
    color_i = int(color)
    r0_i, r1_i, c0_i, c1_i = int(r0), int(r1), int(c0), int(c1)

    if conn == 4:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neigh = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    for rr in range(r0_i, r1_i + 1):
        for cc in range(c0_i, c1_i + 1):
            for dr, dc in neigh:
                nr = rr + int(dr)
                nc = cc + int(dc)
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                # Ignore neighbors that are inside the candidate rectangle; those cells are empty pre-placement.
                if r0_i <= nr <= r1_i and c0_i <= nc <= c1_i:
                    continue
                if int(grid[nr, nc]) == color_i:
                    return True
    return False


def _try_place_solid_rect_no_touch(
    grid: np.ndarray,
    *,
    color: int,
    h: int,
    w: int,
    connectivity: int,
    rng: np.random.Generator,
    max_tries: int = 200,
) -> Optional[tuple[int, int, int, int]]:
    """
    Like `_try_place_solid_rect`, but also enforces that the new rectangle does NOT touch
    an existing component of the same color (per connectivity). This keeps components separated.
    """
    size_r, size_c = int(grid.shape[0]), int(grid.shape[1])
    hh = int(h)
    ww = int(w)
    if hh <= 0 or ww <= 0:
        return None
    if hh > size_r or ww > size_c:
        return None

    for _ in range(int(max_tries)):
        r0 = int(rng.integers(0, size_r - hh + 1))
        c0 = int(rng.integers(0, size_c - ww + 1))
        r1 = r0 + hh - 1
        c1 = c0 + ww - 1
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != 0):
            continue
        if _rect_touches_color(grid, color=int(color), r0=r0, r1=r1, c0=c0, c1=c1, connectivity=int(connectivity)):
            continue
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        return (r0, r1, c0, c1)
    return None


def _make_unique_rects_grid(
    *,
    size: int,
    colors: Sequence[int],
    rng: np.random.Generator,
    ood: bool,
    rect_sizes: Optional[Sequence[tuple[int, int]]] = None,
    unique_components: int = 1,
    connectivity: int = 4,
) -> tuple[np.ndarray, int]:
    """
    Create a grid containing solid non-overlapping rectangles with:
    - exactly one "unique" color that appears in exactly `unique_components` components
    - distractor colors that appear in >=2 disconnected components each (and NOT equal to `unique_components`)

    Allowed rectangle sizes are controlled by `rect_sizes`.
    Returns (grid, unique_color).
    """
    size = int(size)
    conn = int(connectivity)
    if conn not in (4, 8):
        raise ValueError(f"Unsupported connectivity={conn}; expected 4 or 8")

    u_k = int(unique_components)
    if u_k < 1:
        u_k = 1

    if rect_sizes is None:
        rect_sizes = [(1, 1), (1, 2), (2, 1), (2, 2)]
    rect_sizes_f = [(int(h), int(w)) for (h, w) in rect_sizes if int(h) > 0 and int(w) > 0 and int(h) <= size and int(w) <= size]
    if not rect_sizes_f:
        rect_sizes_f = [(1, 1)]

    for _ in range(200):
        grid = np.zeros((size, size), dtype=np.int64)
        unique_color = int(rng.choice(np.asarray(colors)))

        # Place the UNIQUE color as exactly u_k disconnected components.
        ok_unique = True
        for _u in range(u_k):
            uh, uw = rect_sizes_f[int(rng.integers(0, len(rect_sizes_f)))]
            if (
                _try_place_solid_rect_no_touch(
                    grid,
                    color=unique_color,
                    h=int(uh),
                    w=int(uw),
                    connectivity=conn,
                    rng=rng,
                    max_tries=500,
                )
                is None
            ):
                ok_unique = False
                break
        if not ok_unique:
            continue

        available_distractors = [int(c) for c in colors if int(c) != unique_color]
        if not available_distractors:
            continue

        # ID: fewer distractor colors and components; OOD: more.
        max_colors = min(len(available_distractors), 3 if not ood else 4)
        n_distractor_colors = int(rng.integers(1, max_colors + 1))
        distractor_colors = rng.choice(np.asarray(available_distractors), size=n_distractor_colors, replace=False).tolist()

        ok = True
        for d_color in distractor_colors:
            # Choose number of components for this distractor, ensuring it differs from u_k.
            if not ood:
                choices = [2, 3]
            else:
                choices = [2, 3, 4, 5]
            choices = [k for k in choices if int(k) != u_k]
            if not choices:
                choices = [max(2, u_k + 1)]
            n_objs = int(rng.choice(np.asarray(choices)))
            placed = 0
            for _obj in range(n_objs):
                h, w = rect_sizes_f[int(rng.integers(0, len(rect_sizes_f)))]
                if (
                    _try_place_solid_rect_no_touch(
                        grid,
                        color=int(d_color),
                        h=int(h),
                        w=int(w),
                        connectivity=conn,
                        rng=rng,
                        max_tries=500,
                    )
                    is None
                ):
                    ok = False
                    break
                placed += 1
            if not ok:
                break
            if placed < 2:
                ok = False
                break

        if not ok:
            continue

        # Validate the uniqueness property by connected components.
        counts = _component_counts_by_color(grid, connectivity=conn)
        if int(counts.get(unique_color, 0)) != u_k:
            continue
        distractor_ok = True
        for d_color in distractor_colors:
            if int(counts.get(int(d_color), 0)) < 2:
                distractor_ok = False
                break
            if int(counts.get(int(d_color), 0)) == u_k:
                distractor_ok = False
                break
        if not distractor_ok:
            continue

        # Also ensure *only one* color has exactly u_k components (for an unambiguous rule).
        targets = [c for c, k in counts.items() if int(k) == u_k]
        if len(targets) != 1:
            continue

        return grid, unique_color

    # Fallback: simple minimal valid construction.
    grid = np.zeros((size, size), dtype=np.int64)
    unique_color = int(colors[0]) if len(colors) > 0 else 1
    # Place unique components as isolated 1x1 pixels.
    placed_u = 0
    for r in range(size):
        for c in range(size):
            if placed_u >= u_k:
                break
            if int(grid[r, c]) != 0:
                continue
            if _rect_touches_color(grid, color=unique_color, r0=r, r1=r, c0=c, c1=c, connectivity=conn):
                continue
            grid[r, c] = unique_color
            placed_u += 1
        if placed_u >= u_k:
            break

    distractor_color = int(colors[1]) if len(colors) > 1 else 2
    # Place 2 components of distractor_color (or 3 if u_k==2).
    need_d = 3 if u_k == 2 else 2
    placed_d = 0
    for r in range(size):
        for c in range(size):
            if placed_d >= need_d:
                break
            if int(grid[r, c]) != 0:
                continue
            if _rect_touches_color(grid, color=distractor_color, r0=r, r1=r, c0=c, c1=c, connectivity=conn):
                continue
            grid[r, c] = distractor_color
            placed_d += 1
        if placed_d >= need_d:
            break
    return grid, unique_color


class Skill8DropThenShootRay(Puzzle):
    """
    Input contains one or more colored 1x1 pixels (colors in 1..4).
    Output:
      1) Apply down-gravity
      2) Each fallen pixel shoots a left ray of a different color (deterministic mapping) to the left wall.

    OOD: more pixels.
    """

    skill_id = 8
    name = "drop_then_shoot_ray"
    uses_rule_color = False

    def make_input(self, *, ood: bool) -> np.ndarray:
        grid = self.blank()
        # ID: 1 pixel. OOD: more pixels, but capped to at most 2x the IID max (=> 2).
        if not ood:
            n = 1
        else:
            n_max = min(2, int(self.size) * int(self.size))
            n_min = min(2, n_max)
            n = int(self.rng.integers(n_min, n_max + 1))
        placed = 0
        attempts = 0
        while placed < n and attempts < n * 20:
            attempts += 1
            # Avoid left wall so the ray is visible; avoid bottom row so gravity does something.
            r = int(self.rng.integers(0, max(1, self.size - 1)))
            c = int(self.rng.integers(1, self.size))
            if grid[r, c] == 0:
                grid[r, c] = int(self.rng.choice(self.colors))
                placed += 1
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        fallen = apply_gravity(grid)
        out = fallen.copy()
        rs, cs = np.nonzero(fallen)
        for r, c in zip(rs.tolist(), cs.tolist()):
            src_color = int(fallen[r, c])
            ray_color = _ray_color_from_source(src_color)
            _shoot_left_ray(out, row=int(r), from_col=int(c), ray_color=ray_color)
        return out


class Skill9ShootRay(Puzzle):
    """
    Input contains one or more colored 1x1 pixels (colors in 1..4).
    Output: each pixel shoots a left ray of a different color (deterministic mapping) to the left wall.

    OOD: more pixels.
    """

    skill_id = 9
    name = "shoot_ray"
    uses_rule_color = False

    def make_input(self, *, ood: bool) -> np.ndarray:
        grid = self.blank()
        # ID: 1 pixel. OOD: more pixels, but capped to at most 2x the IID max (=> 2).
        if not ood:
            n = 1
        else:
            n_max = min(2, int(self.size) * int(self.size))
            n_min = min(2, n_max)
            n = int(self.rng.integers(n_min, n_max + 1))
        placed = 0
        attempts = 0
        while placed < n and attempts < n * 20:
            attempts += 1
            r = int(self.rng.integers(0, self.size))
            c = int(self.rng.integers(1, self.size))  # avoid left wall so ray is visible
            if grid[r, c] == 0:
                grid[r, c] = int(self.rng.choice(self.colors))
                placed += 1
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        out = grid.copy()
        rs, cs = np.nonzero(grid)
        for r, c in zip(rs.tolist(), cs.tolist()):
            src_color = int(grid[r, c])
            ray_color = _ray_color_from_source(src_color)
            _shoot_left_ray(out, row=int(r), from_col=int(c), ray_color=ray_color)
        return out


class Skill10BoxesToNearestWall(Puzzle):
    """
    Input contains multiple solid rectangles ("boxes"), each with a distinct color.

    Output: move each box horizontally so that it is exactly 1 cell away from its nearest wall
    (left or right), without changing its shape or vertical position.

    "Exclude middle column" (for odd grid sizes): we generate boxes that do NOT occupy the middle
    column so that the nearest-wall choice is unambiguous.
    """

    skill_id = 10
    name = "boxes_to_nearest_wall"
    uses_rule_color = False

    def make_input(self, *, ood: bool) -> np.ndarray:
        grid = self.blank()

        # IID: 2 boxes. OOD: 3-4 boxes (capped to 2x IID => 4).
        n_boxes = 2 if not ood else int(self.rng.integers(3, 5))
        n_boxes = int(min(n_boxes, len(self.colors)))

        size = int(self.size)
        if size < 4:
            # Need at least: wall + 1-cell margin + box width>=2
            raise ValueError(f"Skill10 requires grid size >= 4, got size={size}")

        # "Exclude middle column": always exclude a single column index.
        # For odd sizes this is the true middle; for even sizes this picks the right-middle column.
        mid_col = size // 2

        # Box dimensions scale with grid size (no 5x5 assumptions):
        # - width at least 2 so it's a "box", and at most size-2 so it can be 1 cell away from a wall.
        # - height at least 1; keep small to allow multiple boxes without overlap.
        w_min = 2
        w_max_global = min(4, size - 2)

        used_row_ranges: list[tuple[int, int]] = []
        placed_rects: list[tuple[int, int, int, int]] = []

        box_colors = list(self.rng.choice(np.asarray(self.colors), size=n_boxes, replace=False).tolist())
        for color in box_colors:
            placed = False
            for _ in range(200):
                h = int(self.rng.integers(1, min(3, size) + 1))  # 1..3 (or smaller if size<3, but we guard)
                r0 = int(self.rng.integers(0, size - h + 1))
                r1 = r0 + h - 1

                # Keep boxes vertically disjoint (with a 1-row gap) so horizontal shifts cannot overlap them.
                ok_rows = True
                for ur0, ur1 in used_row_ranges:
                    if not (r1 < ur0 - 1 or ur1 < r0 - 1):
                        ok_rows = False
                        break
                if not ok_rows:
                    continue

                # Place a box fully on either side of the excluded middle column.
                # left side columns: [0 .. mid_col-1] has width mid_col
                # right side columns: [mid_col+1 .. size-1] has width (size-mid_col-1)
                left_width = int(mid_col)
                right_width = int(size - mid_col - 1)

                side = "left" if bool(self.rng.integers(0, 2)) == 0 else "right"
                # If the chosen side can't fit width>=2, flip sides.
                if side == "left" and left_width < w_min:
                    side = "right"
                if side == "right" and right_width < w_min:
                    side = "left"

                side_width = left_width if side == "left" else right_width
                w_max = min(w_max_global, side_width)
                if w_max < w_min:
                    continue
                w = int(self.rng.integers(w_min, w_max + 1))

                if side == "left":
                    max_c0 = (mid_col - 1) - (w - 1)
                    if max_c0 < 0:
                        continue
                    c0 = int(self.rng.integers(0, max_c0 + 1))
                else:
                    min_c0 = mid_col + 1
                    max_c0 = (size - 1) - (w - 1)
                    if max_c0 < min_c0:
                        continue
                    c0 = int(self.rng.integers(min_c0, max_c0 + 1))

                c1 = c0 + w - 1
                if c0 <= mid_col <= c1:
                    continue

                rect = (r0, r1, c0, c1)
                if any(_rects_overlap(rect, r) for r in placed_rects):
                    continue

                grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
                placed_rects.append(rect)
                used_row_ranges.append((r0, r1))
                placed = True
                break

            if not placed:
                break

        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        size = int(grid.shape[1])
        out = np.zeros_like(grid)

        # Each box is monochrome; track boxes by color.
        colors_present = sorted({int(c) for c in grid.flatten().tolist() if int(c) != 0})
        for color in colors_present:
            mask = grid == int(color)
            if not np.any(mask):
                continue
            _, _, c0, c1 = _bbox(mask)

            dist_left = int(c0)
            dist_right = int((size - 1) - c1)
            if dist_left <= dist_right:
                dx = int(1 - c0)  # 1 cell away from left wall
            else:
                dx = int((size - 2) - c1)  # 1 cell away from right wall

            rs, cs = np.nonzero(mask)
            out[rs, (cs + dx).astype(np.int64)] = int(color)

        return out

class Skill11FilterUnique(Puzzle):
    """
    Dependent Skill 1: Uniqueness Detection.
    Input: A grid with noise (distractors) and exactly one pixel of a 'unique' color.
    Output: Only the unique pixel remains; all distractors are removed (black).
    """
    skill_id = 11
    name = "filter_unique"
    uses_rule_color = False

    def make_input(self, *, ood: bool) -> np.ndarray:
        grid = self.blank()
        
        # 1. Pick a color to be the unique one
        unique_color = int(self.rng.choice(self.colors))
        
        # 2. Pick distractor colors (at least 1, max all others)
        available_distractors = [c for c in self.colors if c != unique_color]
        if not available_distractors:
            # Fallback for edge case with few colors
            available_distractors = [c for c in range(1, 5) if c != unique_color]
            
        # 3. Place the Unique Pixel
        # OOD: Place it anywhere. ID: Avoid edges to make it "easier" to see? 
        # Actually, let's keep placement uniform, but vary the amount of clutter.
        r = int(self.rng.integers(0, self.size))
        c = int(self.rng.integers(0, self.size))
        grid[r, c] = unique_color
        
        # 4. Place Distractors
        # Constraint: Distractors must appear at least 2 times each to be "not unique"
        # ID: Low density. OOD: High density.
        num_distractor_colors = int(self.rng.integers(1, len(available_distractors) + 1))
        chosen_distractors = self.rng.choice(available_distractors, size=num_distractor_colors, replace=False)
        
        for d_color in chosen_distractors:
            # Min 2 pixels for distractors
            if ood:
                count = int(self.rng.integers(3, 7)) # Higher clutter
            else:
                count = int(self.rng.integers(2, 4)) # Lower clutter
                
            for _ in range(count):
                # Try to place distractor
                for _try in range(20):
                    dr = int(self.rng.integers(0, self.size))
                    dc = int(self.rng.integers(0, self.size))
                    if grid[dr, dc] == 0:
                        grid[dr, dc] = d_color
                        break
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        # Count color frequencies
        vals, counts = np.unique(grid, return_counts=True)
        unique_c = -1
        
        # Find the color that appears exactly once (excluding background 0)
        for v, count in zip(vals, counts):
            if v != 0 and count == 1:
                unique_c = int(v)
                break
        
        out = np.zeros_like(grid)
        if unique_c != -1:
            # Mask acts as a filter
            mask = (grid == unique_c)
            out[mask] = unique_c
            
        return out


class Skill12PixelExplosion(Puzzle):
    """
    Dependent Skill 2: Spatial Expansion.
    Input: Scattered single pixels.
    Output: Each pixel becomes a 3x3 square of that color centered on the original.
    Overlap handling: Later writes overwrite earlier ones (or simple max).
    """
    skill_id = 12
    name = "pixel_explosion"
    uses_rule_color = False

    def make_input(self, *, ood: bool) -> np.ndarray:
        grid = self.blank()

        # Standard 3x3 kernel offsets.
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]

        # ID: 1-2 well-separated pixels. OOD: 3-4 pixels.
        n_pixels = int(self.rng.integers(3, 5)) if ood else int(self.rng.integers(1, 3))

        placed_locs: list[tuple[int, int]] = []
        for _ in range(n_pixels):
            mask = _get_safe_placement_mask(self.size, offsets, placed_locs, separation_buffer=1)
            mask[grid != 0] = False
            valid = np.argwhere(mask)
            if valid.size == 0:
                break
            idx = int(self.rng.choice(len(valid)))
            r, c = valid[idx]
            grid[int(r), int(c)] = int(self.rng.choice(self.colors))
            placed_locs.append((int(r), int(c)))
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        out = np.zeros_like(grid)
        size = grid.shape[0]
        
        rs, cs = np.nonzero(grid)
        for r, c in zip(rs, cs):
            color = grid[r, c]
            
            # Define 3x3 bounds
            r_min = max(0, r - 1)
            r_max = min(size, r + 2) # Slice exclusive
            c_min = max(0, c - 1)
            c_max = min(size, c + 2)
            
            out[r_min:r_max, c_min:c_max] = color
            
        return out


class Skill13ExplodeUnique(Puzzle):
    """
    Target Skill: Composition of S11 and S12.
    Input: Cluttered grid with one unique color pixel.
    Output: Find the unique pixel, remove everything else, and expand the unique one to 3x3.
    """
    skill_id = 13
    name = "explode_unique"
    uses_rule_color = False

    def make_input(self, *, ood: bool) -> np.ndarray:
        # Use the logic from Skill 11 to generate the input
        # We can instantiate S11 purely for generation if we want, or copy logic.
        # Copying logic to avoid instantiation overhead/context issues.
        
        grid = self.blank()
        unique_color = int(self.rng.choice(self.colors))
        
        available_distractors = [c for c in self.colors if c != unique_color]
        if not available_distractors:
             available_distractors = [c for c in range(1, 5) if c != unique_color]

        # Place Unique
        r = int(self.rng.integers(0, self.size))
        c = int(self.rng.integers(0, self.size))
        grid[r, c] = unique_color
        
        # Place Distractors
        # ID: Low clutter. OOD: High clutter.
        num_distractor_colors = int(self.rng.integers(1, len(available_distractors) + 1))
        chosen_distractors = self.rng.choice(available_distractors, size=num_distractor_colors, replace=False)
        
        for d_color in chosen_distractors:
            if ood:
                count = int(self.rng.integers(3, 7))
            else:
                count = int(self.rng.integers(2, 4))
                
            for _ in range(count):
                for _try in range(20):
                    dr = int(self.rng.integers(0, self.size))
                    dc = int(self.rng.integers(0, self.size))
                    if grid[dr, dc] == 0:
                        grid[dr, dc] = d_color
                        break
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        
        # Step 1: Filter Unique (Logic from S11)
        vals, counts = np.unique(grid, return_counts=True)
        unique_c = -1
        unique_pos = None
        
        for v, count in zip(vals, counts):
            if v != 0 and count == 1:
                unique_c = int(v)
                break
        
        if unique_c == -1:
            return np.zeros_like(grid) # Should not happen based on generator
            
        # Find position of unique color
        rs, cs = np.where(grid == unique_c)
        if len(rs) == 0:
             return np.zeros_like(grid)
             
        r, c = rs[0], cs[0]
        
        # Step 2: Explode (Logic from S12)
        out = np.zeros_like(grid)
        size = grid.shape[0]
        
        r_min = max(0, r - 1)
        r_max = min(size, r + 2)
        c_min = max(0, c - 1)
        c_max = min(size, c + 2)
        
        out[r_min:r_max, c_min:c_max] = unique_c
        
        return out


def _explosion_offsets(mode: str) -> list[tuple[int, int]]:
    """
    Return relative offsets (dr, dc) for a single-pixel "explosion" kernel.
    Modes are designed to increase combinatorial variety while remaining easy to infer from demos.
    """
    if mode == "square3":
        return [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    if mode == "line3_h":
        return [(0, -1), (0, 0), (0, 1)]
    if mode == "line3_v":
        return [(-1, 0), (0, 0), (1, 0)]
    if mode == "ray3_right":
        return [(0, 0), (0, 1), (0, 2)]
    if mode == "ray3_left":
        return [(0, 0), (0, -1), (0, -2)]
    if mode == "ray3_down":
        return [(0, 0), (1, 0), (2, 0)]
    if mode == "ray3_up":
        return [(0, 0), (-1, 0), (-2, 0)]
    raise ValueError(f"Unknown explosion mode: {mode}")


def _explosion_offsets_extended(mode: str) -> list[tuple[int, int]]:
    """
    Extended explosion kernels used by some variant skills.

    Includes the base modes from `_explosion_offsets` plus:
    - diag3_main: main diagonal centered on the pixel
    - diag3_anti: anti-diagonal centered on the pixel
    - cross3: center + 4-neighbors (up/down/left/right)
    """
    if mode == "diag3_main":
        return [(-1, -1), (0, 0), (1, 1)]
    if mode == "diag3_anti":
        return [(-1, 1), (0, 0), (1, -1)]
    if mode == "cross3":
        return [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
    if mode == "square5":
        return [(dr, dc) for dr in (-2, -1, 0, 1, 2) for dc in (-2, -1, 0, 1, 2)]
    if mode == "line5_h":
        return [(0, -2), (0, -1), (0, 0), (0, 1), (0, 2)]
    if mode == "line5_v":
        return [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)]
    if mode == "ray5_right":
        return [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    if mode == "ray5_left":
        return [(0, 0), (0, -1), (0, -2), (0, -3), (0, -4)]
    if mode == "ray5_down":
        return [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    if mode == "ray5_up":
        return [(0, 0), (-1, 0), (-2, 0), (-3, 0), (-4, 0)]
    if mode == "diag5_main":
        return [(-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2)]
    if mode == "diag5_anti":
        return [(-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2)]
    if mode == "cross5":
        return [(-2, 0), (-1, 0), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]
    if mode == "diamond5":
        return [(dr, dc) for dr in (-2, -1, 0, 1, 2) for dc in (-2, -1, 0, 1, 2) if abs(dr) + abs(dc) <= 2]
    if mode == "ring5_square":
        return [
            (dr, dc)
            for dr in (-2, -1, 0, 1, 2)
            for dc in (-2, -1, 0, 1, 2)
            if max(abs(dr), abs(dc)) == 2
        ]
    return _explosion_offsets(mode)


def _shift_bool_grid(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """
    Shift a boolean grid by (dr, dc). Out-of-bounds cells are filled with False.

    dr > 0 shifts content down; dc > 0 shifts content right.
    """
    h, w = int(a.shape[0]), int(a.shape[1])
    out = np.zeros((h, w), dtype=bool)

    if dr >= 0:
        src_r0, dst_r0 = 0, dr
        src_r1, dst_r1 = h - dr, h
    else:
        src_r0, dst_r0 = -dr, 0
        src_r1, dst_r1 = h, h + dr

    if dc >= 0:
        src_c0, dst_c0 = 0, dc
        src_c1, dst_c1 = w - dc, w
    else:
        src_c0, dst_c0 = -dc, 0
        src_c1, dst_c1 = w, w + dc

    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return out

    out[dst_r0:dst_r1, dst_c0:dst_c1] = a[src_r0:src_r1, src_c0:src_c1]
    return out


def _get_safe_placement_mask(
    size: int,
    offsets: Sequence[tuple[int, int]],
    existing_pixels: list[tuple[int, int]],
    separation_buffer: int = 1,
) -> np.ndarray:
    """
    Returns a boolean mask of valid (r, c) locations where a pixel can be placed such that:
    1) Margin check: its explosion (defined by offsets) does not clip outside the grid.
    2) Separation check: its explosion footprint does not overlap or touch existing explosions.

    Separation is enforced at the *explosion footprint* level (not just center distance), and
    uses 4-neighbor adjacency expanded by `separation_buffer` in Chebyshev distance.
    """
    size = int(size)
    mask = np.ones((size, size), dtype=bool)
    if size <= 0:
        return mask

    # 1) Margin check: eliminate edges where the kernel would clip.
    rows = [int(dr) for dr, _dc in offsets]
    cols = [int(dc) for _dr, dc in offsets]
    min_dr, max_dr = int(min(rows)), int(max(rows))
    min_dc, max_dc = int(min(cols)), int(max(cols))

    if min_dr < 0:
        mask[0 : -min_dr, :] = False
    if max_dr > 0:
        mask[size - max_dr : size, :] = False
    if min_dc < 0:
        mask[:, 0 : -min_dc] = False
    if max_dc > 0:
        mask[:, size - max_dc : size] = False

    # 2) Separation check: new explosion must not overlap/touch old explosions.
    if existing_pixels:
        existing_footprint = np.zeros((size, size), dtype=bool)
        for r0, c0 in existing_pixels:
            rr0, cc0 = int(r0), int(c0)
            for dr, dc in offsets:
                rr = rr0 + int(dr)
                cc = cc0 + int(dc)
                if 0 <= rr < size and 0 <= cc < size:
                    existing_footprint[rr, cc] = True

        # Expand footprint by separation_buffer in Chebyshev distance.
        forbidden = existing_footprint.copy()
        b = int(separation_buffer)
        if b > 0:
            for dr in range(-b, b + 1):
                for dc in range(-b, b + 1):
                    if dr == 0 and dc == 0:
                        continue
                    forbidden |= _shift_bool_grid(existing_footprint, dr, dc)

        # Block any center whose explosion footprint would land on a forbidden cell.
        blocked_centers = np.zeros((size, size), dtype=bool)
        for dr, dc in offsets:
            blocked_centers |= _shift_bool_grid(forbidden, -int(dr), -int(dc))
        mask[blocked_centers] = False

        # Also disallow re-using an existing center location.
        for rr, cc in existing_pixels:
            r1, c1 = int(rr), int(cc)
            if 0 <= r1 < size and 0 <= c1 < size:
                mask[r1, c1] = False

    return mask


def _apply_explosion(
    *,
    out: np.ndarray,
    row: int,
    col: int,
    color: int,
    offsets: Sequence[tuple[int, int]],
) -> None:
    """Write an explosion at (row,col) onto out, clipping to bounds."""
    h, w = int(out.shape[0]), int(out.shape[1])
    for dr, dc in offsets:
        rr = int(row) + int(dr)
        cc = int(col) + int(dc)
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = int(color)


class Skill14FilterUniqueVariants(Puzzle):
    """
    Variant of Skill11 (filter_unique), but objects are solid rectangles, not just single pixels.

    Input: non-overlapping solid rectangles of sizes in {1x1,1x2,2x1,2x2}.
           Exactly one color appears in exactly one connected component; all other colors appear
           in >=2 disconnected components.
    Output: keep the entire unique-color object; remove everything else.
    """

    skill_id = 14
    name = "filter_unique_variants"
    uses_rule_color = False

    def __init__(
        self,
        *,
        size: int = 5,
        colors: Sequence[int] = (1, 2, 3, 4),
        ood_spec: OODSpec = OODSpec(),
        shrink_perturb: Optional[ShrinkPerturbSpec] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(size=size, colors=colors, ood_spec=ood_spec, shrink_perturb=shrink_perturb, rng=rng)
        size_i = int(self.size)

        # Variant knobs:
        # - connectivity (4 vs 8) affects component counting when same-color objects touch diagonally
        # - unique_components: keep the color that has exactly K components (K is task-specific)
        # - shape_set controls allowed rectangle sizes
        self._connectivity = int(self.rng.choice(np.asarray([4, 8])))

        k_choices = [1, 2, 3] if size_i >= 5 else [1, 2]
        self._unique_components = int(self.rng.choice(np.asarray(k_choices)))

        shape_sets: list[tuple[str, list[tuple[int, int]]]] = [
            ("tiny", [(1, 1), (1, 2), (2, 1), (2, 2)]),
            ("bars3", [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]),
            ("mix3", [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2)]),
            ("big3", [(1, 1), (2, 2), (3, 3), (1, 3), (3, 1), (2, 3), (3, 2)]),
        ]
        if size_i >= 6:
            shape_sets.append(("bars4", [(1, 4), (4, 1), (1, 3), (3, 1), (2, 2), (1, 2), (2, 1), (1, 1)]))
        self._shape_set_name, self._rect_sizes = shape_sets[int(self.rng.integers(0, len(shape_sets)))]

        # Filter sizes to be valid for the current grid.
        self._rect_sizes = [(int(h), int(w)) for (h, w) in self._rect_sizes if int(h) <= size_i and int(w) <= size_i]
        if not self._rect_sizes:
            self._rect_sizes = [(1, 1)]

    @property
    def variant_id(self) -> Optional[str]:
        return f"conn{int(self._connectivity)}_k{int(self._unique_components)}_{self._shape_set_name}"

    def variant_params(self) -> dict[str, object]:
        return {
            "connectivity": int(self._connectivity),
            "unique_components": int(self._unique_components),
            "shape_set": str(self._shape_set_name),
            "rect_sizes": [(int(h), int(w)) for (h, w) in self._rect_sizes],
        }

    def make_input(self, *, ood: bool) -> np.ndarray:
        grid, _unique_color = _make_unique_rects_grid(
            size=self.size,
            colors=self.colors,
            rng=self.rng,
            ood=ood,
            rect_sizes=self._rect_sizes,
            unique_components=int(self._unique_components),
            connectivity=int(self._connectivity),
        )
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        counts = _component_counts_by_color(grid, connectivity=int(self._connectivity))
        target_colors = [c for c, k in counts.items() if int(k) == int(self._unique_components)]
        if not target_colors:
            return np.zeros_like(grid)
        unique_c = int(sorted(target_colors)[0])
        out = np.zeros_like(grid)
        out[grid == unique_c] = unique_c
        return out


class Skill15PixelExplosionVariants(Puzzle):
    """
    Variant of Skill12 (pixel_explosion) with a per-task explosion kernel:
    - 3x3 square
    - 3-long horizontal / vertical line centered
    - 3-long directional "ray" from the pixel (up/down/left/right)
    - 3-long diagonals
    - 5-cell cross (center + 4-neighbors)
    """

    skill_id = 15
    name = "pixel_explosion_variants"
    uses_rule_color = False

    def __init__(
        self,
        *,
        size: int = 5,
        colors: Sequence[int] = (1, 2, 3, 4),
        ood_spec: OODSpec = OODSpec(),
        shrink_perturb: Optional[ShrinkPerturbSpec] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(size=size, colors=colors, ood_spec=ood_spec, shrink_perturb=shrink_perturb, rng=rng)
        modes = [
            "square3",
            "line3_h",
            "line3_v",
            "ray3_right",
            "ray3_left",
            "ray3_down",
            "ray3_up",
            "diag3_main",
            "diag3_anti",
            "cross3",
            # Larger kernels / extra shapes (still per-task and demo-inferable)
            "square5",
            "line5_h",
            "line5_v",
            "diag5_main",
            "diag5_anti",
            "cross5",
            "diamond5",
            "ring5_square",
        ]
        self._explosion_mode = str(self.rng.choice(np.asarray(modes)))
        self._offsets = _explosion_offsets_extended(self._explosion_mode)

    @property
    def variant_id(self) -> Optional[str]:
        return str(self._explosion_mode)

    def variant_params(self) -> dict[str, object]:
        return {
            "explosion_mode": str(self._explosion_mode),
            "offsets": [(int(dr), int(dc)) for (dr, dc) in self._offsets],
        }

    def make_input(self, *, ood: bool) -> np.ndarray:
        grid = self.blank()

        # ID: 1-2 pixels. OOD: 2-4 pixels.
        n_pixels = int(self.rng.integers(2, 5)) if ood else int(self.rng.integers(1, 3))

        placed_locs: list[tuple[int, int]] = []
        for _ in range(n_pixels):
            mask = _get_safe_placement_mask(self.size, self._offsets, placed_locs, separation_buffer=1)
            mask[grid != 0] = False
            valid = np.argwhere(mask)
            if valid.size == 0:
                break
            idx = int(self.rng.choice(len(valid)))
            r, c = valid[idx]
            grid[int(r), int(c)] = int(self.rng.choice(self.colors))
            placed_locs.append((int(r), int(c)))
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        out = np.zeros_like(grid)
        rs, cs = np.nonzero(grid)
        for r, c in zip(rs.tolist(), cs.tolist()):
            color = int(grid[int(r), int(c)])
            _apply_explosion(out=out, row=int(r), col=int(c), color=color, offsets=self._offsets)
        return out


class Skill16ExplodeUniqueVariants(Puzzle):
    """
    Variant of Skill13 (explode_unique) that composes:
    - Skill14-style uniqueness detection (by connected components)
    - Skill15-style explosion kernel

    Input: scattered pixels where exactly one color forms exactly one 4-connected component,
           and all other present colors form >=2 disconnected components each.
    Output: explode every pixel of the unique-color component using the per-task kernel.
    """

    skill_id = 16
    name = "explode_unique_variants"
    uses_rule_color = False

    def __init__(
        self,
        *,
        size: int = 5,
        colors: Sequence[int] = (1, 2, 3, 4),
        ood_spec: OODSpec = OODSpec(),
        shrink_perturb: Optional[ShrinkPerturbSpec] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(size=size, colors=colors, ood_spec=ood_spec, shrink_perturb=shrink_perturb, rng=rng)
        modes = [
            "square3",
            "line3_h",
            "line3_v",
            "ray3_right",
            "ray3_left",
            "ray3_down",
            "ray3_up",
            "diag3_main",
            "diag3_anti",
            "cross3",
            "square5",
            "line5_h",
            "line5_v",
            "diag5_main",
            "diag5_anti",
            "cross5",
            "diamond5",
            "ring5_square",
        ]
        self._explosion_mode = str(self.rng.choice(np.asarray(modes)))
        self._offsets = _explosion_offsets_extended(self._explosion_mode)

    @property
    def variant_id(self) -> Optional[str]:
        return str(self._explosion_mode)

    def variant_params(self) -> dict[str, object]:
        return {
            "explosion_mode": str(self._explosion_mode),
            "offsets": [(int(dr), int(dc)) for (dr, dc) in self._offsets],
        }

    def make_input(self, *, ood: bool) -> np.ndarray:
        size = int(self.size)

        for _attempt in range(200):
            grid = self.blank()

            # 1) Pick the unique color and distractor colors.
            unique_color = int(self.rng.choice(self.colors))
            distractors = [int(c) for c in self.colors if int(c) != unique_color]
            if not distractors:
                distractors = [c for c in range(1, 5) if int(c) != unique_color]
            if not distractors:
                continue

            max_d = min(len(distractors), 3 if not ood else 4)
            n_distractor_colors = int(self.rng.integers(1, max_d + 1))
            distractor_colors = (
                self.rng.choice(np.asarray(distractors), size=n_distractor_colors, replace=False).tolist()
            )

            # 2) Place UNIQUE object (strictly 1x1) in a safe spot for the explosion.
            mask = _get_safe_placement_mask(size, self._offsets, [], separation_buffer=1)
            valid = np.argwhere(mask)
            if valid.size == 0:
                continue
            idx = int(self.rng.choice(len(valid)))
            ur, uc = valid[idx]
            ur_i, uc_i = int(ur), int(uc)
            grid[ur_i, uc_i] = unique_color

            # 3) Place DISTRACTORS such that each distractor color has >=2 disconnected components.
            # We do this by placing isolated (4-disconnected) pixels for each distractor color.
            for d_color in distractor_colors:
                if ood:
                    n_pix = int(self.rng.integers(3, 7))
                else:
                    n_pix = int(self.rng.integers(2, 5))

                placed = 0
                tries = 0
                while placed < n_pix and tries < 400:
                    tries += 1
                    r = int(self.rng.integers(0, size))
                    c = int(self.rng.integers(0, size))
                    if grid[r, c] != 0:
                        continue
                    # Ensure this pixel is not 4-adjacent to an existing pixel of the same color,
                    # so components stay separated.
                    if r > 0 and int(grid[r - 1, c]) == int(d_color):
                        continue
                    if r + 1 < size and int(grid[r + 1, c]) == int(d_color):
                        continue
                    if c > 0 and int(grid[r, c - 1]) == int(d_color):
                        continue
                    if c + 1 < size and int(grid[r, c + 1]) == int(d_color):
                        continue
                    grid[r, c] = int(d_color)
                    placed += 1

                if placed < 2:
                    break

            # Validate uniqueness-by-components (same contract as Skill14/16 expect).
            counts = _component_counts_by_color(grid)
            if int(counts.get(unique_color, 0)) != 1:
                continue
            if any(int(counts.get(int(dc), 0)) < 2 for dc in distractor_colors):
                continue
            singles = [c for c, k in counts.items() if int(k) == 1]
            if len(singles) != 1:
                continue

            return grid

        # Fallback: minimal, valid construction (still enforces unique 1x1).
        grid = self.blank()
        unique_color = int(self.colors[0]) if len(self.colors) > 0 else 1
        mask = _get_safe_placement_mask(size, self._offsets, [], separation_buffer=1)
        valid = np.argwhere(mask)
        if valid.size == 0:
            ur_i, uc_i = size // 2, size // 2
        else:
            r, c = valid[int(self.rng.choice(len(valid)))]
            ur_i, uc_i = int(r), int(c)
        grid[ur_i, uc_i] = unique_color

        distractor_color = int(self.colors[1]) if len(self.colors) > 1 else 2
        # Place two isolated pixels of the distractor color.
        placed = 0
        for r in range(size):
            for c in range(size):
                if placed >= 2:
                    break
                if grid[r, c] != 0:
                    continue
                if r > 0 and int(grid[r - 1, c]) == distractor_color:
                    continue
                if r + 1 < size and int(grid[r + 1, c]) == distractor_color:
                    continue
                if c > 0 and int(grid[r, c - 1]) == distractor_color:
                    continue
                if c + 1 < size and int(grid[r, c + 1]) == distractor_color:
                    continue
                grid[r, c] = distractor_color
                placed += 1
            if placed >= 2:
                break
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        counts = _component_counts_by_color(grid)
        unique_colors = [c for c, k in counts.items() if int(k) == 1]
        if not unique_colors:
            return np.zeros_like(grid)
        unique_c = int(sorted(unique_colors)[0])
        mask = grid == unique_c

        out = np.zeros_like(grid)
        rs, cs = np.nonzero(mask)
        for r, c in zip(rs.tolist(), cs.tolist()):
            _apply_explosion(out=out, row=int(r), col=int(c), color=unique_c, offsets=self._offsets)
        return out


class Skill17ComponentLabeling(Puzzle):
    """
    Component labeling skill (simplified): recolor each blob by its CENTER pixel color.

    Input: multiple disconnected solid rectangles, but with RANDOM colors inside each rectangle.
           Each rectangle has odd height/width ∈ {1,3} so it has a well-defined center pixel.
    Output: for each connected component (4-connected over non-zero), recolor ALL pixels in that
            component to match the component's center pixel color from the input.
    """

    skill_id = 17
    name = "component_labeling"
    uses_rule_color = False

    def __init__(
        self,
        *,
        size: int = 5,
        colors: Sequence[int] = (1, 2, 3, 4),
        ood_spec: OODSpec = OODSpec(),
        shrink_perturb: Optional[ShrinkPerturbSpec] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(size=size, colors=colors, ood_spec=ood_spec, shrink_perturb=shrink_perturb, rng=rng)
        # Keep this skill focused: fixed 4-connectivity.
        self._connectivity = 4

    @property
    def variant_id(self) -> Optional[str]:
        return f"conn{int(self._connectivity)}"

    def variant_params(self) -> dict[str, object]:
        return {"connectivity": int(self._connectivity)}

    def make_input(self, *, ood: bool) -> np.ndarray:
        grid = self.blank()
        max_components = min(4, int(self.size))  # keep sparse + always placeable
        if not ood:
            n_comp = int(self.rng.integers(2, min(3, max_components) + 1))
        else:
            n_comp = int(self.rng.integers(min(3, max_components), max_components + 1))

        # Only odd dimensions so the center pixel is unambiguous.
        rect_sizes: list[tuple[int, int]] = [(1, 1), (1, 3), (3, 1), (3, 3)]
        size_r, size_c = int(grid.shape[0]), int(grid.shape[1])

        def can_place(r0: int, c0: int, h: int, w: int) -> bool:
            r1 = r0 + int(h) - 1
            c1 = c0 + int(w) - 1
            if r0 < 0 or c0 < 0 or r1 >= size_r or c1 >= size_c:
                return False
            # Enforce a 1-cell buffer around the rectangle (Chebyshev), so blobs stay disconnected.
            rr0 = max(0, r0 - 1)
            rr1 = min(size_r - 1, r1 + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(size_c - 1, c1 + 1)
            return not bool(np.any(grid[rr0 : rr1 + 1, cc0 : cc1 + 1] != 0))

        placed = 0
        tries = 0
        while placed < n_comp and tries < 800:
            tries += 1
            h, w = rect_sizes[int(self.rng.integers(0, len(rect_sizes)))]
            h_i, w_i = int(h), int(w)
            if h_i > size_r or w_i > size_c:
                continue
            r0 = int(self.rng.integers(0, size_r - h_i + 1))
            c0 = int(self.rng.integers(0, size_c - w_i + 1))
            if not can_place(r0, c0, h_i, w_i):
                continue
            # Fill the rectangle with random colors (non-zero).
            patch = self.rng.choice(np.asarray(self.colors), size=(h_i, w_i), replace=True).astype(np.int64)
            grid[r0 : r0 + h_i, c0 : c0 + w_i] = patch
            placed += 1
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        mask = grid != 0
        comps = _components_sorted(mask, connectivity=int(self._connectivity))
        out = np.zeros_like(grid)

        for comp in comps:
            rs = [int(r) for r, _c in comp]
            cs = [int(c) for _r, c in comp]
            r0, r1 = int(min(rs)), int(max(rs))
            c0, c1 = int(min(cs)), int(max(cs))
            cr = (r0 + r1) // 2
            cc = (c0 + c1) // 2
            color = int(grid[cr, cc])
            for r, c in comp:
                out[int(r), int(c)] = color
        return out


class Skill18PerComponentPixelCount(Puzzle):
    """
    Per-component feature extraction skill (pixel count) -- simplified to a single blob.

    Input: exactly ONE connected component (4-connected over non-zero).
    Output: a single vertical bar in column 0 whose height equals the blob's pixel count,
            drawn from the bottom up.
    """

    skill_id = 18
    name = "per_component_pixel_count"
    uses_rule_color = False

    def __init__(
        self,
        *,
        size: int = 5,
        colors: Sequence[int] = (1, 2, 3, 4),
        ood_spec: OODSpec = OODSpec(),
        shrink_perturb: Optional[ShrinkPerturbSpec] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(size=size, colors=colors, ood_spec=ood_spec, shrink_perturb=shrink_perturb, rng=rng)
        self._connectivity = 4

        # Restrict shapes so pixel-count fits in the grid height (bar height <= size).
        size_i = int(self.size)
        shapes: list[tuple[int, int]] = [(1, 1), (1, 2), (2, 1), (2, 2)]
        if size_i >= 4:
            shapes += [(1, 3), (3, 1)]
        if size_i >= 5:
            shapes += [(1, 4), (4, 1)]
        # Filter by area <= size (so bar height fits).
        self._rect_sizes = [(int(h), int(w)) for (h, w) in shapes if int(h) * int(w) <= size_i]
        if not self._rect_sizes:
            self._rect_sizes = [(1, 1)]

        # Per-task input coloring mode (still one connected component: mask is "non-zero").
        self._fill_mode = str(self.rng.choice(np.asarray(["solid_1", "solid_random", "random_cells", "stripes"])))

    @property
    def variant_id(self) -> Optional[str]:
        return f"conn{int(self._connectivity)}_{self._fill_mode}"

    def variant_params(self) -> dict[str, object]:
        return {
            "connectivity": int(self._connectivity),
            "fill_mode": str(self._fill_mode),
            "rect_sizes": [(int(h), int(w)) for (h, w) in self._rect_sizes],
        }

    def make_input(self, *, ood: bool) -> np.ndarray:
        _ = ood
        grid = self.blank()
        # Always place exactly ONE rectangle, with area <= size so the output bar fits.
        size_i = int(self.size)
        for _ in range(600):
            h, w = self._rect_sizes[int(self.rng.integers(0, len(self._rect_sizes)))]
            h_i, w_i = int(h), int(w)
            if h_i <= 0 or w_i <= 0 or (h_i * w_i) > size_i:
                continue
            if h_i > size_i or w_i > size_i:
                continue
            r0 = int(self.rng.integers(0, size_i - h_i + 1))
            c0 = int(self.rng.integers(0, size_i - w_i + 1))
            if bool(np.any(grid[r0 : r0 + h_i, c0 : c0 + w_i] != 0)):
                continue
            if self._fill_mode == "solid_1":
                patch = np.ones((h_i, w_i), dtype=np.int64)
            elif self._fill_mode == "solid_random":
                c = int(self.rng.choice(np.asarray(self.colors))) if len(self.colors) > 0 else 1
                patch = np.full((h_i, w_i), int(c), dtype=np.int64)
            elif self._fill_mode == "stripes":
                # Alternating row stripes of two colors (both non-zero).
                if len(self.colors) >= 2:
                    c1, c2 = (int(self.colors[0]), int(self.colors[1]))
                elif len(self.colors) == 1:
                    c1, c2 = (int(self.colors[0]), int(self.colors[0]))
                else:
                    c1, c2 = (1, 2)
                patch = np.zeros((h_i, w_i), dtype=np.int64)
                for rr in range(h_i):
                    patch[rr, :] = c1 if (rr % 2 == 0) else c2
            else:
                # random_cells
                palette = np.asarray(self.colors, dtype=np.int64)
                if palette.size == 0:
                    palette = np.asarray([1, 2, 3, 4], dtype=np.int64)
                patch = self.rng.choice(palette, size=(h_i, w_i), replace=True).astype(np.int64)
            grid[r0 : r0 + h_i, c0 : c0 + w_i] = patch
            break
        return grid

    def apply(self, grid: np.ndarray, rule_color: Optional[int]) -> np.ndarray:
        _ = rule_color
        size_i = int(grid.shape[0])
        mask = grid != 0
        comps = _components_sorted(mask, connectivity=int(self._connectivity))
        if not comps:
            return np.zeros_like(grid)
        comp = comps[0]
        height = int(len(comp))
        if height > size_i:
            height = size_i
        out = np.zeros_like(grid)
        bar_color = int(self.colors[0]) if len(self.colors) > 0 else 1
        out[size_i - height : size_i, 0] = int(bar_color)
        return out

# --- Update the Factory ---

def build_puzzle(
    skill_id: int,
    *,
    size: int,
    rng: Optional[np.random.Generator] = None,
    shrink_perturb: Optional[ShrinkPerturbSpec] = None,
) -> Puzzle:
    if skill_id == 1:
        return Skill1Gravity(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 2:
        return Skill2BottomRecolor(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 3:
        return Skill3GravityThenRecolor(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 4:
        return Skill4PlaceCenterDot(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 5:
        return Skill5FillRectangleInterior(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 6:
        return Skill6RightGravity(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 7:
        return Skill7DownThenRight(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 8:
        return Skill8DropThenShootRay(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 9:
        return Skill9ShootRay(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 10:
        return Skill10BoxesToNearestWall(size=size, shrink_perturb=shrink_perturb, rng=rng)
    # New Skills
    if skill_id == 11:
        return Skill11FilterUnique(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 12:
        return Skill12PixelExplosion(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 13:
        return Skill13ExplodeUnique(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 14:
        return Skill14FilterUniqueVariants(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 15:
        return Skill15PixelExplosionVariants(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 16:
        return Skill16ExplodeUniqueVariants(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 17:
        return Skill17ComponentLabeling(size=size, shrink_perturb=shrink_perturb, rng=rng)
    if skill_id == 18:
        return Skill18PerComponentPixelCount(size=size, shrink_perturb=shrink_perturb, rng=rng)
        
    raise ValueError(f"Unknown skill_id={skill_id}")

