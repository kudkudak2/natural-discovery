from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from arc_dataset_models import ARCDataset
from arc_dataset_models import ARCExamplePair, ARCTask, ARCTestCase
from arc_aug import AugmentSpec, augment_src_tgt_batch, augment_src_tgt_batch_with_params, invert_grids_torch

import hashlib

def run_unit_tests(*, test_paths: list[Path]) -> None:
    """
    Run pytest on the given paths. Raises on failure.
    """
    paths = [Path(p).expanduser().resolve() for p in test_paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise ValueError(f"Missing test files: {[str(p) for p in missing]}")
    if importlib.util.find_spec("pytest") is None:
        raise RuntimeError("pytest is not installed. Install it (e.g. `pip install pytest`) or run with --no-run_tests.")
    import pytest  # type: ignore

    code = int(pytest.main(["-q", *[str(p) for p in paths]]))
    if code != 0:
        raise RuntimeError(f"Unit tests failed with exit code {code}.")


# --- Token vocabulary ---
# ARC colors are 0..9. We reserve two special tokens:
# - SEP_TOKEN: separator between grids in the prompt
# - PAD_TOKEN: reserved for potential variable-length prompts (kept fixed by augmentations)
N_COLORS = 10
SEP_TOKEN = N_COLORS  # 10
PAD_TOKEN = N_COLORS + 1  # 11
VOCAB_SIZE = N_COLORS + 2  # 12

# Visualization palette for colors 0..9 (SEP/PAD are not colors).
ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 light blue
    "#870C25",  # 9 maroon
]


def prompt_seq_len(
    *,
    grid_size: int,
    num_demos: int = 3,
    output_grid_size: Optional[int] = None,
) -> int:
    """
    Prompt layout (fixed):
      (x SEP y SEP) repeated `num_demos` times, then (test_x SEP)
    where x/test_x are input grid_size*grid_size tokens and y is output_grid_size*output_grid_size.
    When output_grid_size is None, it equals grid_size (input and output same size).
    """
    g_in = int(grid_size) * int(grid_size)
    out_g = int(output_grid_size if output_grid_size is not None else grid_size)
    g_out = out_g * out_g
    return int(num_demos) * (g_in + 1 + g_out + 1) + (g_in + 1)


def infer_input_grid_size(
    seq_len: int,
    num_demos: int,
    output_grid_size: int,
) -> Optional[int]:
    """
    Infer input grid side from prompt sequence length when layout is (g_in, g_out, num_demos).
    Returns g_in if seq_len == prompt_seq_len(g_in, num_demos, output_grid_size) for some g_in, else None.
    """
    nd = int(num_demos)
    g_out = int(output_grid_size)
    if nd <= 0 or g_out <= 0:
        return None
    out_sq = g_out * g_out
    remainder = int(seq_len) - nd * (2 + out_sq) - 1
    if remainder <= 0:
        return None
    denom = nd + 1
    if remainder % denom != 0:
        return None
    g_in_sq = remainder // denom
    g_in = int(round(math.sqrt(g_in_sq)))
    if g_in * g_in != g_in_sq or g_in <= 0:
        return None
    if prompt_seq_len(grid_size=g_in, num_demos=nd, output_grid_size=g_out) != int(seq_len):
        return None
    return g_in


def _max_grid_size_for_seq_len(*, max_seq_len: int, num_demos: int) -> int:
    """
    Maximum grid_size such that prompt_seq_len(grid_size, num_demos) <= max_seq_len.

    prompt_seq_len = (2*num_demos+1) * (grid_size^2) + (2*num_demos+1).
    """
    cap = int(max_seq_len)
    nd = int(num_demos)
    if cap <= 0:
        raise ValueError(f"max_seq_len must be >= 1, got {cap}")
    if nd <= 0:
        raise ValueError(f"num_demos must be >= 1, got {nd}")
    denom = int(2 * nd + 1)
    # Need grid_size^2 <= cap/denom - 1.
    base = (cap // denom) - 1
    if base <= 0:
        return 0
    return int(math.isqrt(int(base)))


def _has_tqdm() -> bool:
    return importlib.util.find_spec("tqdm") is not None


def _has_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def progress(iterable, *, total: int, desc: str, enabled: bool):
    if not enabled:
        return iterable
    if _has_tqdm():
        from tqdm import tqdm  # type: ignore

        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit="it",
            unit_scale=False,
            smoothing=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
    return iterable


def render_ascii(grid: np.ndarray) -> str:
    # Keep it compact for debugging: 0..9 as digits, unknown as '?'
    chars = {i: str(i) for i in range(10)}
    return "\n".join(" ".join(chars.get(int(c), "?") for c in row) for row in grid)


def _decode_prompt_src(
    *,
    src_tokens: np.ndarray,
    grid_size: int,
    num_demos: int = 3,
    output_grid_size: Optional[int] = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Inverse of `_flatten_prompt` for visualization/debug.
    Layout: (x SEP y SEP) repeated `num_demos` times, then (test_x SEP).
    x/test_x use grid_size (input); y uses output_grid_size (default grid_size).
    """
    if src_tokens.ndim != 1:
        raise ValueError(f"Expected 1D src_tokens, got shape={src_tokens.shape}")
    g_in = int(grid_size)
    g_out = int(output_grid_size if output_grid_size is not None else grid_size)
    if g_in <= 0:
        raise ValueError(f"grid_size must be >= 1, got {g_in}")
    if g_out <= 0:
        raise ValueError(f"output_grid_size must be >= 1, got {g_out}")
    in_tokens = g_in * g_in
    out_tokens = g_out * g_out
    expected = prompt_seq_len(grid_size=g_in, num_demos=int(num_demos), output_grid_size=g_out)
    if int(src_tokens.shape[0]) != int(expected):
        raise ValueError(f"Unexpected src length={int(src_tokens.shape[0])} (expected {expected})")

    def unflatten_in(block: np.ndarray) -> np.ndarray:
        return np.asarray(block, dtype=np.int64).reshape(g_in, g_in)

    def unflatten_out(block: np.ndarray) -> np.ndarray:
        return np.asarray(block, dtype=np.int64).reshape(g_out, g_out)

    demos: list[tuple[np.ndarray, np.ndarray]] = []
    off = 0
    for _ in range(int(num_demos)):
        x = unflatten_in(src_tokens[off : off + in_tokens])
        off += in_tokens
        if int(src_tokens[off]) != int(SEP_TOKEN):
            raise ValueError(f"Expected SEP after demo x at off={off}, got {int(src_tokens[off])}")
        off += 1

        y = unflatten_out(src_tokens[off : off + out_tokens])
        off += out_tokens
        if int(src_tokens[off]) != int(SEP_TOKEN):
            raise ValueError(f"Expected SEP after demo y at off={off}, got {int(src_tokens[off])}")
        off += 1
        demos.append((x, y))

    test_x = unflatten_in(src_tokens[off : off + in_tokens])
    off += in_tokens
    if int(src_tokens[off]) != int(SEP_TOKEN):
        raise ValueError(f"Expected trailing SEP after test_x at off={off}, got {int(src_tokens[off])}")
    return demos, test_x


def decode_prompt_src(
    *,
    src_tokens: np.ndarray,
    grid_size: int,
    num_demos: int = 3,
    output_grid_size: Optional[int] = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Public wrapper for `_decode_prompt_src` (used by other scripts for plotting/debug)."""
    return _decode_prompt_src(
        src_tokens=src_tokens,
        grid_size=grid_size,
        num_demos=num_demos,
        output_grid_size=output_grid_size,
    )


def _save_arc_failure_png(
    *,
    demos: list[tuple[np.ndarray, np.ndarray]],
    test_x: np.ndarray,
    pred_y: np.ndarray,
    true_y: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    if not _has_matplotlib():
        return
    from matplotlib import pyplot as plt  # type: ignore
    from matplotlib.colors import BoundaryNorm, ListedColormap  # type: ignore

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = ListedColormap(list(ARC_COLORS), name="arc10")
    norm = BoundaryNorm(boundaries=[-0.5] + [i + 0.5 for i in range(int(N_COLORS))], ncolors=int(N_COLORS))

    def clip_colors(a: np.ndarray) -> np.ndarray:
        aa = np.asarray(a, dtype=np.int64)
        return np.clip(aa, 0, int(N_COLORS - 1))

    panels: list[tuple[str, np.ndarray]] = []
    for i, (dx, dy) in enumerate(demos):
        panels.append((f"demo{i+1} x", clip_colors(dx)))
        panels.append((f"demo{i+1} y", clip_colors(dy)))
    panels.append(("test x", clip_colors(test_x)))
    panels.append(("pred y", clip_colors(pred_y)))
    panels.append(("true y", clip_colors(true_y)))

    n = len(panels)
    fig_w = max(8.0, 2.2 * float(n))
    fig_h = 2.8
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]
    for ax, (lab, grid) in zip(axes, panels):
        ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(lab, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # light cell grid
        g = int(grid.shape[0])
        ax.set_xticks(np.arange(-0.5, g, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, g, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0, alpha=0.4)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_arc_prompt_prediction_png(
    *,
    demos: list[tuple[np.ndarray, np.ndarray]],
    test_x: np.ndarray,
    pred_y: np.ndarray,
    true_y: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Save a compact multi-panel ARC prompt/prediction visualization as a PNG (no-op if matplotlib missing)."""
    _save_arc_failure_png(
        demos=demos,
        test_x=test_x,
        pred_y=pred_y,
        true_y=true_y,
        out_path=out_path,
        title=title,
    )


@dataclass(frozen=True)
class Batch:
    src: torch.Tensor  # (B, T) padded with PAD_TOKEN
    tgt: torch.Tensor  # (B, Gmax) padded with ignore_index (-100)
    grid_size: torch.Tensor  # (B,) long (actual per-example grid size)
    num_demos: torch.Tensor  # (B,) long (actual per-example num_demos)
    pred_pos: torch.Tensor  # (B, Gmax) long indices into T
    pred_mask: torch.Tensor  # (B, Gmax) bool
    key_padding_mask: torch.Tensor  # (B, T) bool (True for PAD positions)


def _flatten_prompt(demos: list[tuple[np.ndarray, np.ndarray]], test_in: np.ndarray) -> list[int]:
    seq: list[int] = []
    for x, y in demos:
        seq += x.flatten().tolist() + [SEP_TOKEN] + y.flatten().tolist() + [SEP_TOKEN]
    seq += test_in.flatten().tolist() + [SEP_TOKEN]
    return [int(t) for t in seq]


def _parse_dataset_json(path: Path) -> ARCDataset:
    raw = path.read_text(encoding="utf-8")
    if hasattr(ARCDataset, "model_validate_json"):
        return ARCDataset.model_validate_json(raw)  # pydantic v2
    return ARCDataset.parse_raw(raw)  # pydantic v1


def _as_data_dirs(data_dir: Path | list[Path]) -> list[Path]:
    if isinstance(data_dir, list):
        return [Path(p) for p in data_dir]
    return [Path(data_dir)]


def _iter_json_files(dir_path: Path) -> list[Path]:
    d = Path(dir_path)
    if not d.exists() or (not d.is_dir()):
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".json"])


def _is_arc_agi_root(dir_path: Path) -> bool:
    d = Path(dir_path)
    tr = d / "training"
    ev = d / "evaluation"
    return tr.is_dir() and ev.is_dir() and (len(_iter_json_files(tr)) > 0) and (len(_iter_json_files(ev)) > 0)


def _is_synthetic_skill_root(dir_path: Path) -> bool:
    """
    Heuristic for the synthetic dataset layout produced by `arc_dataset_generator.py`:
      <root>/skill_<id>/{train,ood}.json

    We keep this check lightweight and purely structural (no JSON parsing).
    """
    d = Path(dir_path)
    if not d.is_dir():
        return False
    for child in d.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("skill_"):
            continue
        if (child / "train.json").exists() or (child / "ood.json").exists():
            return True
    return False


def _grid_dims(grid: list[list[int]]) -> tuple[int, int]:
    if not isinstance(grid, list) or len(grid) == 0:
        raise ValueError("Grid must be a non-empty list of rows")
    if not all(isinstance(r, list) for r in grid):
        raise ValueError("Grid rows must be lists")
    h = int(len(grid))
    w = int(len(grid[0]))
    if w <= 0:
        raise ValueError("Grid must have at least 1 column")
    for r in grid:
        if int(len(r)) != int(w):
            raise ValueError("Grid rows must all have the same length")
    return h, w


def _validate_color_range(*, grid: list[list[int]], path: Path) -> None:
    mn = 10**9
    mx = -(10**9)
    for row in grid:
        for v in row:
            if not isinstance(v, int):
                raise ValueError(f"Grid contains non-int value {v!r} in {path}")
            mn = min(mn, int(v))
            mx = max(mx, int(v))
    if mn < 0 or mx >= int(N_COLORS):
        raise ValueError(f"Grid has values outside [0..{int(N_COLORS - 1)}] in {path}: min={mn} max={mx}")


def _pad_to_square(grid: list[list[int]], *, size: int) -> list[list[int]]:
    h, w = _grid_dims(grid)
    g = int(size)
    if g <= 0:
        raise ValueError(f"size must be >= 1, got {g}")
    if h > g or w > g:
        raise ValueError(f"Grid {h}x{w} does not fit into requested square size={g}")
    # IMPORTANT: within-grid padding uses background color 0 (not PAD_TOKEN).
    # This is required for geometry/translation augmentations to preserve semantics
    # when mixing variable-sized grids inside a fixed max canvas.
    out = [[0 for _ in range(g)] for _ in range(g)]
    for r in range(h):
        row = grid[r]
        for c in range(w):
            out[r][c] = int(row[c])
    return out


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:24]


def _read_arc_agi_json(path: Path) -> dict:
    raw = Path(path).read_text(encoding="utf-8")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected ARC-AGI JSON object at {path}, got {type(obj).__name__}")
    return obj


def _extract_arc_agi_tasks_from_file(
    path: Path,
) -> list[tuple[list[list[list[int]]], list[list[list[int]]], list[list[int]], list[list[int]], str]]:
    """
    Parse one ARC-AGI JSON file into raw tasks (before padding/num_demos normalization).

    Returns list of tuples:
      (demo_xs, demo_ys, test_x, test_y, task_id_suffix)
    """
    obj = _read_arc_agi_json(path)
    tr = obj.get("train", None)
    te = obj.get("test", None)
    if not isinstance(tr, list) or len(tr) == 0:
        raise ValueError(f"ARC-AGI file missing non-empty 'train' list: {path}")
    if not isinstance(te, list) or len(te) == 0:
        raise ValueError(f"ARC-AGI file missing non-empty 'test' list: {path}")

    demo_xs: list[list[list[int]]] = []
    demo_ys: list[list[list[int]]] = []
    for i, pair in enumerate(tr):
        if not isinstance(pair, dict):
            raise ValueError(f"train[{i}] must be an object in {path}")
        x = pair.get("input", None)
        y = pair.get("output", None)
        if not isinstance(x, list) or not isinstance(y, list):
            raise ValueError(f"train[{i}] must have 'input'/'output' grids in {path}")
        _validate_color_range(grid=x, path=path)
        _validate_color_range(grid=y, path=path)
        _ = _grid_dims(x), _grid_dims(y)
        demo_xs.append(x)
        demo_ys.append(y)

    out: list[tuple[list[list[list[int]]], list[list[list[int]]], list[list[int]], list[list[int]], str]] = []
    for j, case in enumerate(te):
        if not isinstance(case, dict):
            raise ValueError(f"test[{j}] must be an object in {path}")
        x = case.get("input", None)
        y = case.get("output", None)
        if not isinstance(x, list):
            raise ValueError(f"test[{j}] must have 'input' grid in {path}")
        if not isinstance(y, list):
            # Some ARC evaluation sources omit output; for training/eval we require it.
            raise ValueError(f"test[{j}] must have 'output' grid (ground truth) in {path}")
        _validate_color_range(grid=x, path=path)
        _validate_color_range(grid=y, path=path)
        _ = _grid_dims(x), _grid_dims(y)
        out.append((demo_xs, demo_ys, x, y, f"{path.stem}_test{j}"))
    return out


def _infer_external_grid_size(
    raw_tasks: list[tuple[list[list[list[int]]], list[list[list[int]]], list[list[int]], list[list[int]], str]],
) -> int:
    mx = 0
    for demo_xs, demo_ys, test_x, test_y, _tid in raw_tasks:
        for g in demo_xs + demo_ys + [test_x, test_y]:
            h, w = _grid_dims(g)
            mx = max(mx, int(h), int(w))
    if mx <= 0:
        raise ValueError("Could not infer grid_size from external dataset (no tasks/grids).")
    return int(mx)


def _build_external_arcdataset(
    *,
    raw_tasks: list[tuple[list[list[list[int]]], list[list[list[int]]], list[list[int]], list[list[int]], str]],
    split: str,
    grid_size: int,
    num_demos: int,
    dataset_id: str,
) -> ARCDataset:
    g = int(grid_size)
    nd = int(num_demos)
    # g <= 0: per-task padding (each task padded to its own max dimension). Variable grid size per task.
    # g > 0: fixed grid; skip tasks with any grid > g, pad all to g.
    use_per_task_padding = g <= 0
    # nd <= 0 means "use all available demos per task".

    tasks: list[ARCTask] = []
    skipped_too_few_demos = 0
    skipped_too_large = 0
    for demo_xs, demo_ys, test_x, test_y, tid_suf in raw_tasks:
        if int(len(demo_xs)) <= 0 or int(len(demo_ys)) <= 0:
            skipped_too_few_demos += 1
            continue
        use_nd = int(nd) if int(nd) > 0 else min(int(len(demo_xs)), int(len(demo_ys)))
        if int(use_nd) <= 0:
            skipped_too_few_demos += 1
            continue
        mx = 0
        for gg in demo_xs[:use_nd] + demo_ys[:use_nd] + [test_x, test_y]:
            h, w = _grid_dims(gg)
            mx = max(int(mx), int(h), int(w))
        if not use_per_task_padding and int(mx) > int(g):
            skipped_too_large += 1
            continue
        pad_size = int(mx) if use_per_task_padding else int(g)

        demos = []
        for i in range(int(use_nd)):
            demos.append(
                ARCExamplePair(x=_pad_to_square(demo_xs[i], size=pad_size), y=_pad_to_square(demo_ys[i], size=pad_size))
            )
        test = ARCTestCase(x=_pad_to_square(test_x, size=pad_size), y=_pad_to_square(test_y, size=pad_size))

        task_id = _stable_id(dataset_id, str(split), tid_suf)
        tasks.append(
            ARCTask(
                task_id=str(task_id),
                skill_id=0,
                skill_name="external_arc",
                grid_size=int(pad_size),
                demos=demos,
                test=test,
            )
        )

    if len(tasks) == 0:
        raise ValueError(f"External dataset split {split!r} produced no usable tasks (num_demos={nd}).")
    # When per-task padding, dataset-level grid_size is max over tasks (for downstream compat).
    ds_grid_size = max(int(t.grid_size) for t in tasks) if use_per_task_padding else int(g)
    return ARCDataset(
        dataset_id=str(dataset_id),
        created_at=ARCDataset.now_iso(),
        split=str(split),
        ood=False,
        skills=[0],
        grid_size=int(ds_grid_size),
        tasks=tasks,
        extra={"skipped_too_few_demos": int(skipped_too_few_demos), "skipped_too_large": int(skipped_too_large)},
    )


def maybe_load_external_arc_splits(
    *,
    data_dirs: list[Path],
    grid_size: int,
    num_demos: int,
    rng: np.random.Generator,
    train_frac_for_unsplit: float,
    max_seq_len: Optional[int] = None,
) -> Optional[list[tuple[str, "TensorizedDataset", "TensorizedDataset"]]]:
    """
    Auto-detect and load external ARC datasets.

    Supported layouts:
    - ARC-AGI: <root>/{training,evaluation}/*.json
      Uses training for training and evaluation for evaluation.
    - Generic: <root>/<any_subdir>/*.json (where subdir names are NOT exactly training/evaluation)
      Loads all jsons and performs a deterministic split into (train, evaluation) using train_frac_for_unsplit.
    """
    roots = [Path(p).expanduser().resolve() for p in data_dirs]
    roots = [r for r in roots if r.exists()]

    # Per-root raw tasks, so we can report metrics per dataset while still merging training pools.
    raw_by_root: dict[Path, dict[str, list[tuple[list[list[list[int]]], list[list[list[int]]], list[list[int]], list[list[int]], str]]]] = {}

    for root in roots:
        if _is_synthetic_skill_root(root):
            continue
        if _is_arc_agi_root(root):
            raw_tr: list[tuple[list[list[list[int]]], list[list[list[int]]], list[list[int]], list[list[int]], str]] = []
            raw_ev: list[tuple[list[list[list[int]]], list[list[list[int]]], list[list[int]], list[list[int]], str]] = []
            for p in _iter_json_files(root / "training"):
                raw_tr.extend(_extract_arc_agi_tasks_from_file(p))
            for p in _iter_json_files(root / "evaluation"):
                raw_ev.extend(_extract_arc_agi_tasks_from_file(p))
            if len(raw_tr) > 0 or len(raw_ev) > 0:
                raw_by_root[root] = {"train": raw_tr, "evaluation": raw_ev}
            continue

        # Generic: immediate subdirs with jsons, ignore files directly under root.
        if root.is_dir():
            raw_all: list[tuple[list[list[list[int]]], list[list[list[int]]], list[list[int]], list[list[int]], str]] = []
            kids = [p for p in root.iterdir() if p.is_dir()]
            for sub in sorted(kids):
                js = _iter_json_files(sub)
                for p in js:
                    raw_all.extend(_extract_arc_agi_tasks_from_file(p))
            if len(raw_all) > 0:
                raw_by_root[root] = {"all": raw_all}

    if len(raw_by_root) == 0:
        return None

    raw_for_infer: list[tuple[list[list[list[int]]], list[list[list[int]]], list[list[int]], list[list[int]], str]] = []
    for m in raw_by_root.values():
        for v in m.values():
            raw_for_infer.extend(v)

    # For external datasets, `num_demos` is treated as:
    # - >0: cap each task to the first N demos
    # - <=0: use all available demos per task (variable demo count supported downstream)
    nd = int(num_demos)
    nd_cap = max(int(len(dx)) for (dx, _dy, _tx, _ty, _tid) in raw_for_infer)
    nd_cap = max(1, int(nd_cap))

    g_infer = int(grid_size) if int(grid_size) > 0 else _infer_external_grid_size(raw_for_infer)
    if max_seq_len is not None and int(max_seq_len) > 0 and int(nd) > 0:
        # Use the worst-case demo count for the cap; tasks with fewer demos will be shorter.
        # This keeps the resulting fixed token budget within max_seq_len without silently reducing demos.
        g_cap = _max_grid_size_for_seq_len(max_seq_len=int(max_seq_len), num_demos=int(nd))
        if int(g_cap) <= 0:
            raise ValueError(
                f"max_seq_len={int(max_seq_len)} is too small for num_demos={int(nd)} "
                f"(minimum is prompt_seq_len(grid_size=1,num_demos={int(nd)})="
                f"{int(prompt_seq_len(grid_size=1, num_demos=int(nd)))})."
            )
        if int(grid_size) > 0 and int(g_infer) > int(g_cap):
            raise ValueError(
                f"Requested grid_size={int(g_infer)} exceeds what max_seq_len={int(max_seq_len)} allows for num_demos={int(nd)} "
                f"(max grid_size={int(g_cap)})."
            )
        g_infer = min(int(g_infer), int(g_cap))
    g = int(g_infer)

    out: list[tuple[str, TensorizedDataset, TensorizedDataset]] = []
    for root, splits in sorted(raw_by_root.items(), key=lambda kv: str(kv[0])):
        name = root.name or str(root)
        dataset_id = _stable_id("external_arc", str(root))

        if "train" in splits or "evaluation" in splits:
            raw_tr = splits.get("train", [])
            raw_ev = splits.get("evaluation", [])
            if len(raw_tr) == 0 or len(raw_ev) == 0:
                raise ValueError(
                    f"Detected ARC-AGI layout for {root} but one split is empty. "
                    "Ensure both <root>/training and <root>/evaluation contain .json files."
                )
            ds_train = _build_external_arcdataset(raw_tasks=raw_tr, split="train", grid_size=int(grid_size), num_demos=nd, dataset_id=dataset_id)
            ds_eval = _build_external_arcdataset(
                raw_tasks=raw_ev, split="evaluation", grid_size=int(grid_size), num_demos=nd, dataset_id=dataset_id
            )
            if max_seq_len is not None and int(max_seq_len) > 0:
                tr_sk = int(getattr(ds_train, "extra", {}).get("skipped_too_large", 0))
                ev_sk = int(getattr(ds_eval, "extra", {}).get("skipped_too_large", 0))
                if tr_sk > 0 or ev_sk > 0:
                    # For nd<=0 ("use all demos"), this is only informational.
                    demos_for_cap = int(nd) if int(nd) > 0 else int(nd_cap)
                    print(
                        f"[max_seq_len={int(max_seq_len)}] external_arc[{name}] skipped_too_large: "
                        f"train={tr_sk} evaluation={ev_sk} (grid_size_cap={int(g)}, num_demos={int(demos_for_cap)})",
                        flush=True,
                    )
            t_train = _tensorize_dataset(ds_train, max_seq_len=max_seq_len)
            t_eval = _tensorize_dataset(ds_eval, max_seq_len=max_seq_len)
            t_eval = _dedupe_against(
                keep=t_train,
                drop=t_eval,
                label=f"external_arc_agi[{name}]: train vs evaluation",
                split_suffix="dedup_vs_train",
            )
            assert_disjoint_datasets(a=t_train, b=t_eval, label=f"external_arc_agi[{name}]: train vs evaluation (post-dedup)")
            out.append((name, t_train, t_eval))
            continue

        raw_all = splits.get("all", [])
        ds_all = _build_external_arcdataset(raw_tasks=raw_all, split="all", grid_size=int(grid_size), num_demos=nd, dataset_id=dataset_id)
        if max_seq_len is not None and int(max_seq_len) > 0:
            sk = int(getattr(ds_all, "extra", {}).get("skipped_too_large", 0))
            if sk > 0:
                # For nd<=0 ("use all demos"), this is only informational.
                demos_for_cap = int(nd) if int(nd) > 0 else int(nd_cap)
                print(
                    f"[max_seq_len={int(max_seq_len)}] external_arc[{name}] skipped_too_large: "
                    f"all={sk} (grid_size_cap={int(g)}, num_demos={int(demos_for_cap)})",
                    flush=True,
                )
        t_all = _tensorize_dataset(ds_all, max_seq_len=max_seq_len)
        tr, ev = split_dataset(t_all, train_frac=float(train_frac_for_unsplit), rng=rng)
        out_g_tr = tr.effective_output_grid_size()
        out_g_ev = ev.effective_output_grid_size()
        tr = TensorizedDataset(
            skill_id=0,
            split="train",
            grid_size=tr.grid_size,
            num_demos=tr.num_demos,
            src_list=tr.src_list,
            tgt_list=tr.tgt_list,
            grid_size_each=tr.grid_size_each,
            num_demos_each=tr.num_demos_each,
            output_grid_size=out_g_tr,
        )
        ev = TensorizedDataset(
            skill_id=0,
            split="evaluation",
            grid_size=ev.grid_size,
            num_demos=ev.num_demos,
            src_list=ev.src_list,
            tgt_list=ev.tgt_list,
            grid_size_each=ev.grid_size_each,
            num_demos_each=ev.num_demos_each,
            output_grid_size=out_g_ev,
        )
        assert_disjoint_datasets(a=tr, b=ev, label=f"external_generic[{name}]: train vs evaluation")
        out.append((name, tr, ev))

    return out


@dataclass(frozen=True)
class TensorizedDataset:
    """
    Variable-length storage: each example has its own seq length and target size.
    Padding to batch max is done in prepare_batch (and eval batching).
    """
    skill_id: int
    split: str
    grid_size: int  # max input grid size (demos x, test x)
    num_demos: int  # max num_demos_each in this dataset
    src_list: list[torch.Tensor]  # length N; each (T_i,) source sequence
    tgt_list: list[torch.Tensor]  # length N; each (G_i,) target grid flattened; G_i = output_side_i^2
    grid_size_each: torch.Tensor  # (N,) long — output grid side we predict
    num_demos_each: torch.Tensor  # (N,) long
    output_grid_size: int = 0  # max output grid size (demos y, target); 0 means same as grid_size

    @property
    def n(self) -> int:
        return len(self.src_list)

    def effective_output_grid_size(self) -> int:
        """Output grid size (same as grid_size when output_grid_size is 0)."""
        return int(self.output_grid_size) if int(self.output_grid_size) > 0 else int(self.grid_size)


def _subset_dataset(ds: TensorizedDataset, idx: np.ndarray, *, split_suffix: str) -> TensorizedDataset:
    idx_list = idx.tolist()
    out_g = int(ds.output_grid_size) if int(ds.output_grid_size) > 0 else int(ds.grid_size)
    return TensorizedDataset(
        skill_id=ds.skill_id,
        split=f"{ds.split}_{split_suffix}",
        grid_size=int(ds.grid_size_each[idx].max().item()) if len(idx_list) > 0 else ds.grid_size,
        num_demos=int(ds.num_demos_each[idx].max().item()) if len(idx_list) > 0 else ds.num_demos,
        src_list=[ds.src_list[i] for i in idx_list],
        tgt_list=[ds.tgt_list[i] for i in idx_list],
        grid_size_each=ds.grid_size_each[idx],
        num_demos_each=ds.num_demos_each[idx],
        output_grid_size=out_g,
    )


def _row_digests(ds: TensorizedDataset) -> set[bytes]:
    """Per-example digests for disjointness checks (variable-length rows)."""
    if ds.n <= 0:
        return set()
    out: set[bytes] = set()
    for i in range(ds.n):
        s = ds.src_list[i].detach().cpu().numpy().tobytes() + ds.tgt_list[i].detach().cpu().numpy().tobytes()
        out.add(hashlib.blake2b(s, digest_size=16).digest())
    return out


def _row_digest_list(ds: TensorizedDataset) -> list[bytes]:
    """Per-row digests aligned with dataset row indices."""
    if ds.n <= 0:
        return []
    return [
        hashlib.blake2b(
            ds.src_list[i].detach().cpu().numpy().tobytes() + ds.tgt_list[i].detach().cpu().numpy().tobytes(),
            digest_size=16,
        ).digest()
        for i in range(ds.n)
    ]


def _dedupe_against(*, keep: TensorizedDataset, drop: TensorizedDataset, label: str, split_suffix: str) -> TensorizedDataset:
    """
    Remove any examples from `drop` that exactly match an example in `keep` (based on src+tgt digest).

    This is used to enforce disjointness when upstream datasets occasionally contain duplicates.
    """
    if keep.n == 0 or drop.n == 0:
        return drop
    keep_set = _row_digests(keep)
    drop_digests = _row_digest_list(drop)
    mask = np.asarray([d not in keep_set for d in drop_digests], dtype=bool)
    idx = np.nonzero(mask)[0].astype(np.int64)
    removed = int(drop.n) - int(idx.shape[0])
    if removed == 0:
        return drop
    if int(idx.shape[0]) == 0:
        raise ValueError(f"After deduplication, dataset became empty ({label}).")
    return _subset_dataset(drop, idx, split_suffix=f"{split_suffix}_rm{removed}")


def assert_disjoint_datasets(*, a: TensorizedDataset, b: TensorizedDataset, label: str) -> None:
    """
    Raise if `a` and `b` share any exact example (src+tgt).

    This is a guardrail against accidental train/test leakage when changing splitting,
    pooling, augmentation, or dataset loading code.
    """
    if a.n == 0 or b.n == 0:
        return
    da = _row_digests(a)
    db = _row_digests(b)
    overlap = da & db
    if len(overlap) != 0:
        raise ValueError(
            f"Train/test leakage detected: datasets are not disjoint ({label}). "
            f"a=({a.skill_id},{a.split},n={a.n}) b=({b.skill_id},{b.split},n={b.n}) "
            f"overlap={len(overlap)}"
        )


def split_dataset(
    ds: TensorizedDataset, *, train_frac: float, rng: np.random.Generator
) -> tuple[TensorizedDataset, TensorizedDataset]:
    """
    Deterministically split a dataset into (train_part, eval_part).
    Ensures both are non-empty when ds.n >= 2.

    Important: this split is performed over *unique examples* (exact src+tgt).
    If the dataset contains duplicates, we keep all identical rows on the same side
    to prevent train/test leakage.
    """
    frac = float(train_frac)
    if not (0.0 <= frac <= 1.0):
        raise ValueError(f"train_frac must be in [0,1], got {frac}")
    if ds.n == 1:
        # Can't split; keep it for eval to avoid leaking the only sample.
        return _subset_dataset(ds, np.asarray([], dtype=np.int64), split_suffix="train0"), ds

    n_train = int(frac * ds.n)
    n_train = max(1, n_train)
    n_train = min(ds.n - 1, n_train)

    # If there are duplicates, a naive row-level split can put identical examples
    # on both sides, which will trip `assert_disjoint_datasets`. We instead split
    # by digest-groups.
    digests: list[bytes] = _row_digest_list(ds)

    groups: dict[bytes, list[int]] = {}
    for i, d in enumerate(digests):
        groups.setdefault(d, []).append(int(i))

    # Fast path: no duplicates.
    if len(groups) == ds.n:
        perm = rng.permutation(ds.n)
        train_idx = perm[:n_train]
        eval_idx = perm[n_train:]
        return _subset_dataset(ds, train_idx, split_suffix=f"train{n_train}"), _subset_dataset(ds, eval_idx, split_suffix="heldout")

    keys = list(groups.keys())
    order = rng.permutation(len(keys))

    train_rows = 0
    train_keys: list[bytes] = []
    for oi, k_i in enumerate(order):
        k = keys[int(k_i)]
        sz = len(groups[k])
        # Never consume all rows into train; eval must remain non-empty.
        if train_rows + sz >= ds.n:
            continue
        # Greedily add groups until we reach (or slightly exceed) the target.
        if train_rows < n_train:
            train_keys.append(k)
            train_rows += sz

        # If we've already hit the target, we can stop early.
        if train_rows >= n_train and oi + 1 < len(order):
            # Still leave remaining groups for eval.
            continue

    # If the greedy pass somehow left train empty (pathological), force one group into train.
    if len(train_keys) == 0:
        k0 = keys[int(order[0])]
        if len(groups[k0]) >= ds.n:
            raise ValueError(
                f"Cannot split into disjoint train/eval: dataset has only one unique example (n={ds.n})."
            )
        train_keys = [k0]

    train_idx_list: list[int] = []
    for k in train_keys:
        train_idx_list.extend(groups[k])
    train_idx = np.asarray(train_idx_list, dtype=np.int64)

    train_mask = np.zeros(ds.n, dtype=bool)
    train_mask[train_idx] = True
    eval_idx = np.nonzero(~train_mask)[0].astype(np.int64)

    # Final guard: ensure non-empty.
    if int(train_idx.shape[0]) == 0 or int(eval_idx.shape[0]) == 0:
        raise ValueError(
            f"Grouped split failed to produce non-empty partitions: train_n={int(train_idx.shape[0])} eval_n={int(eval_idx.shape[0])}."
        )

    return _subset_dataset(ds, train_idx, split_suffix=f"train{int(train_idx.shape[0])}"), _subset_dataset(ds, eval_idx, split_suffix="heldout")


def concat_datasets(datasets: list[TensorizedDataset], *, skill_id: int, split: str, grid_size: int) -> TensorizedDataset:
    non_empty = [ds for ds in datasets if ds.n > 0]
    if len(non_empty) == 0:
        raise ValueError("No datasets to concatenate (all empty).")
    src_list: list[torch.Tensor] = []
    tgt_list: list[torch.Tensor] = []
    for ds in non_empty:
        src_list.extend(ds.src_list)
        tgt_list.extend(ds.tgt_list)
    grid_each = torch.cat([ds.grid_size_each for ds in non_empty], dim=0)
    demos_each = torch.cat([ds.num_demos_each for ds in non_empty], dim=0)
    out_g = max(int(d.output_grid_size) if int(d.output_grid_size) > 0 else int(d.grid_size) for d in non_empty)
    return TensorizedDataset(
        skill_id=skill_id,
        split=split,
        grid_size=int(grid_size),
        num_demos=int(max(int(d.num_demos) for d in non_empty)),
        src_list=src_list,
        tgt_list=tgt_list,
        grid_size_each=grid_each,
        num_demos_each=demos_each,
        output_grid_size=out_g,
    )


def cap_dataset(ds: TensorizedDataset, *, cap: Optional[int], rng: np.random.Generator) -> TensorizedDataset:
    if cap is None:
        return ds
    cap_i = int(cap)
    if cap_i <= 0:
        raise ValueError(f"cap must be >= 1, got {cap_i}")
    if ds.n <= cap_i:
        return ds
    idx = rng.permutation(ds.n)[:cap_i]
    idx_list = idx.tolist()
    out_g = int(ds.output_grid_size) if int(ds.output_grid_size) > 0 else int(ds.grid_size)
    return TensorizedDataset(
        skill_id=ds.skill_id,
        split=f"{ds.split}_cap{cap_i}",
        grid_size=int(ds.grid_size_each[idx].max().item()),
        num_demos=int(ds.num_demos_each[idx].max().item()),
        src_list=[ds.src_list[i] for i in idx_list],
        tgt_list=[ds.tgt_list[i] for i in idx_list],
        grid_size_each=ds.grid_size_each[idx],
        num_demos_each=ds.num_demos_each[idx],
        output_grid_size=out_g,
    )


def _tensorize_dataset(ds: ARCDataset, *, max_seq_len: Optional[int] = None) -> TensorizedDataset:
    # Per-task variability is supported. We encode every task into a *fixed* prompt layout
    # sized by the maxima across tasks in this dataset: (max_grid_size, max_num_demos).
    tasks = list(ds.tasks)
    if len(tasks) == 0:
        raise ValueError("Dataset has no tasks.")

    parsed: list[tuple[int, int, int, list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]] = []
    all_seq_lens: list[int] = []
    max_g = 0
    max_nd = 0
    dropped = 0
    total = 0

    for task in tasks:
        total += 1
        demos: list[tuple[np.ndarray, np.ndarray]] = []
        for demo in task.demos:
            demos.append((np.asarray(demo.x, dtype=np.int64), np.asarray(demo.y, dtype=np.int64)))
        nd = int(len(demos))
        if nd <= 0:
            raise ValueError(f"Task has no demonstrations (task_id={getattr(task,'task_id','?')})")

        test_in = np.asarray(task.test.x, dtype=np.int64)
        test_out = np.asarray(task.test.y, dtype=np.int64)
        if test_in.ndim != 2:
            raise ValueError(f"Expected 2D test input grid, got ndim={test_in.ndim}")
        if test_out.ndim != 2:
            raise ValueError(f"Expected 2D test output grid, got ndim={test_out.ndim}")

        # Sequence length uses the max dimension over ALL grids in this task (demos + test).
        # ARC-AGI has variable-sized demos; padding is to this task's max, so seq_len = prompt_seq_len(task_max_g, nd).
        task_max_g = 0
        for x, y in demos:
            task_max_g = max(int(task_max_g), int(x.shape[0]), int(x.shape[1]), int(y.shape[0]), int(y.shape[1]))
        task_max_g = max(int(task_max_g), int(test_in.shape[0]), int(test_in.shape[1]), int(test_out.shape[0]), int(test_out.shape[1]))
        if int(task_max_g) <= 0:
            raise ValueError(f"Task has no grid content (task_id={getattr(task,'task_id','?')})")
        test_g = max(int(test_out.shape[0]), int(test_out.shape[1]))  # target grid size we predict

        tlen = int(prompt_seq_len(grid_size=int(task_max_g), num_demos=int(nd)))
        all_seq_lens.append(tlen)
        if max_seq_len is not None and int(max_seq_len) > 0:
            if int(tlen) > int(max_seq_len):
                dropped += 1
                continue

        parsed.append((task_max_g, test_g, nd, demos, test_in, test_out))
        max_g = max(int(max_g), int(task_max_g))
        max_nd = max(int(max_nd), int(nd))

    # Always print quantiles of all tasks (including dropped) for debugging.
    if len(all_seq_lens) > 0:
        seq_arr = np.array(all_seq_lens)
        q0, q25, q50, q75, q100 = np.percentile(seq_arr, [0, 25, 50, 75, 100])
        dsid = getattr(ds, "dataset_id", "?")
        split_s = getattr(ds, "split", "?")
        print(
            f"[seq_len quantiles] dataset_id={dsid} split={split_s} n_all={len(all_seq_lens)} kept={len(parsed)} "
            f"min={int(q0)} p25={int(q25)} p50={int(q50)} p75={int(q75)} max={int(q100)}",
            flush=True,
        )

    if len(parsed) == 0:
        if max_seq_len is not None and int(max_seq_len) > 0:
            raise ValueError(
                f"After applying max_seq_len={int(max_seq_len)}, dataset became empty (dropped {int(dropped)}/{int(total)} tasks)."
            )
        raise ValueError("Dataset became empty after parsing tasks.")

    def embed_grid(grid: np.ndarray, *, out_size: int) -> np.ndarray:
        out = np.full((int(out_size), int(out_size)), 0, dtype=np.int64)
        r, c = int(grid.shape[0]), int(grid.shape[1])
        out[:r, :c] = np.asarray(grid, dtype=np.int64)
        return out

    src_list: list[torch.Tensor] = []
    tgt_list: list[torch.Tensor] = []
    grid_each = torch.empty((len(parsed),), dtype=torch.long)
    demos_each = torch.empty((len(parsed),), dtype=torch.long)

    for i, (task_max_g, test_g, nd, demos, test_in, test_out) in enumerate(parsed):
        grid_each[i] = int(test_g)
        demos_each[i] = int(nd)
        g = int(task_max_g)
        nd_i = int(nd)

        demos_fixed = []
        for di in range(nd_i):
            x, y = demos[di]
            demos_fixed.append((embed_grid(x, out_size=g), embed_grid(y, out_size=g)))
        test_in_big = embed_grid(test_in, out_size=g)
        seq = _flatten_prompt(demos_fixed, test_in_big)
        src_list.append(torch.tensor(seq, dtype=torch.long))

        tgt_flat = torch.full((g * g,), -100, dtype=torch.long)
        to = np.asarray(test_out, dtype=np.int64)
        rows, cols = int(to.shape[0]), int(to.shape[1])
        for r in range(rows):
            for c in range(cols):
                tgt_flat[int(r * g + c)] = int(to[r, c])
        tgt_list.append(tgt_flat)

    if max_seq_len is not None and int(max_seq_len) > 0:
        kept = int(len(parsed))
        cap = int(max_seq_len)
        dsid = getattr(ds, "dataset_id", "?")
        split_s = getattr(ds, "split", "?")
        print(
            f"[max_seq_len={cap}] filtered tasks for dataset_id={dsid} split={split_s}: "
            f"dropped={int(dropped)}/{int(total)} kept={kept} (padding per batch)",
            flush=True,
        )

    return TensorizedDataset(
        skill_id=int(ds.skills[0]) if ds.skills else -1,
        split=str(ds.split),
        grid_size=int(max_g),
        num_demos=int(max_nd),
        src_list=src_list,
        tgt_list=tgt_list,
        grid_size_each=grid_each,
        num_demos_each=demos_each,
    )


def pad_dataset_to(ds: TensorizedDataset, *, grid_size: int, num_demos: int, output_grid_size: int) -> TensorizedDataset:
    """
    Retokenize each example to a larger (grid_size, num_demos) budget.
    """
    g = int(grid_size)
    nd = int(num_demos)
    if g <= 0:
        raise ValueError(f"grid_size must be >= 1, got {g}")
    if nd <= 0:
        raise ValueError(f"num_demos must be >= 1, got {nd}")
    if int(ds.grid_size) > int(g) or int(ds.num_demos) > int(nd):
        raise ValueError("pad_dataset_to can only increase (grid_size, num_demos).")
    if int(ds.grid_size) == int(g) and int(ds.num_demos) == int(nd):
        return ds

    dev = ds.src_list[0].device if ds.n > 0 else torch.device("cpu")
    out_src_list = []
    out_tgt_list = []

    for i in range(ds.n):
        tokens = ds.src_list[i]
        old_nd_i = int(ds.num_demos_each[i].item())
        old_g_i = int(round(ds.tgt_list[i].shape[0] ** 0.5))
        old_grid_tokens = old_g_i * old_g_i

        off = 0
        seq = []
        for di in range(nd):
            if di < old_nd_i:
                x_flat = tokens[off : off + old_grid_tokens]
                off += old_grid_tokens + 1
                y_flat = tokens[off : off + old_grid_tokens]
                off += old_grid_tokens + 1
                x_old = x_flat.reshape(old_g_i, old_g_i)
                y_old = y_flat.reshape(old_g_i, old_g_i)
            else:
                x_old = torch.zeros((old_g_i, old_g_i), dtype=torch.long, device=tokens.device)
                y_old = torch.zeros((old_g_i, old_g_i), dtype=torch.long, device=tokens.device)
            x_big = torch.zeros((g, g), dtype=torch.long, device=tokens.device)
            y_big = torch.zeros((g, g), dtype=torch.long, device=tokens.device)
            x_big[:old_g_i, :old_g_i] = x_old
            y_big[:old_g_i, :old_g_i] = y_old
            seq += x_big.reshape(-1).tolist() + [int(SEP_TOKEN)] + y_big.reshape(-1).tolist() + [int(SEP_TOKEN)]

        test_x_flat = tokens[off : off + old_grid_tokens]
        test_x_old = test_x_flat.reshape(old_g_i, old_g_i)
        test_x_big = torch.zeros((g, g), dtype=torch.long, device=tokens.device)
        test_x_big[:old_g_i, :old_g_i] = test_x_old
        seq += test_x_big.reshape(-1).tolist() + [int(SEP_TOKEN)]
        out_src_list.append(torch.tensor(seq, dtype=torch.long, device=dev))

        t_old = ds.tgt_list[i].reshape(old_g_i, old_g_i)
        t_big = torch.full((g, g), -100, dtype=torch.long, device=dev)
        t_big[:old_g_i, :old_g_i] = t_old
        out_tgt_list.append(t_big.reshape(-1))

    return TensorizedDataset(
        skill_id=ds.skill_id,
        split=ds.split,
        grid_size=g,
        num_demos=nd,
        src_list=out_src_list,
        tgt_list=out_tgt_list,
        grid_size_each=ds.grid_size_each.to(device=dev),
        num_demos_each=ds.num_demos_each.to(device=dev),
        output_grid_size=int(output_grid_size),
    )


def load_skill_split(
    *, data_dir: Path | list[Path], skill_id: int, split: str, max_seq_len: Optional[int] = None
) -> TensorizedDataset:
    """
    Load a synthetic (skill-based) dataset split.

    `data_dir` can be:
    - a single directory (back-compat)
    - a list of directories: all matching splits found across roots are concatenated.
    """
    roots = [Path(p).expanduser().resolve() for p in _as_data_dirs(data_dir)]
    parts: list[TensorizedDataset] = []
    for root in roots:
        path = root / f"skill_{int(skill_id)}" / f"{split}.json"
        if not path.exists():
            continue
        ds = _parse_dataset_json(path)
        tds = _tensorize_dataset(ds, max_seq_len=max_seq_len)
        if int(tds.grid_size) <= 0:
            raise ValueError("Invalid grid_size in dataset.")
        if int(tds.num_demos) <= 0:
            raise ValueError("Invalid num_demos in dataset.")
        parts.append(tds)

    if len(parts) == 0:
        raise ValueError(f"Missing dataset: skill_{int(skill_id)}/{split}.json in any of: {[str(p) for p in roots]}")
    if len(parts) == 1:
        return parts[0]

    max_g = max(int(p.grid_size) for p in parts)
    max_out_g = max(int(p.output_grid_size) if int(p.output_grid_size) > 0 else int(p.grid_size) for p in parts)
    max_nd = max(int(p.num_demos) for p in parts)
    padded = [
        pad_dataset_to(p, grid_size=int(max_g), num_demos=int(max_nd), output_grid_size=int(max_out_g))
        for p in parts
    ]
    return concat_datasets(padded, skill_id=int(skill_id), split=split, grid_size=int(max_g))


def maybe_load_skill_split(
    *, data_dir: Path | list[Path], skill_id: int, split: str, max_seq_len: Optional[int] = None
) -> Optional[TensorizedDataset]:
    roots = [Path(p).expanduser().resolve() for p in _as_data_dirs(data_dir)]
    for root in roots:
        path = root / f"skill_{int(skill_id)}" / f"{split}.json"
        if path.exists():
            return load_skill_split(data_dir=data_dir, skill_id=skill_id, split=split, max_seq_len=max_seq_len)
    return None


def _pad_batch_variable(
    src_list: list[torch.Tensor],
    tgt_list: list[torch.Tensor],
    idx: torch.Tensor,
    device: torch.device,
    *,
    pad_token: int,
    tgt_ignore_index: int = -100,
    T_max: Optional[int] = None,
    G_max: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length src/tgt to batch max or to T_max/G_max when provided; return (src, tgt, key_padding_mask, src_lengths)."""
    idx_list = idx.tolist()
    src_tensors = [src_list[i] for i in idx_list]
    tgt_tensors = [tgt_list[i] for i in idx_list]
    T_batch = max(int(s.shape[0]) for s in src_tensors)
    G_batch = max(int(t.shape[0]) for t in tgt_tensors)
    T = int(T_max) if T_max is not None else T_batch
    G = int(G_max) if G_max is not None else G_batch
    if T < T_batch or G < G_batch:
        raise ValueError(f"T_max={T_max} and G_max={G_max} must be >= batch max (T_batch={T_batch}, G_batch={G_batch})")
    bsz = len(idx_list)
    src = torch.full((bsz, T), pad_token, dtype=torch.long, device=device)
    tgt = torch.full((bsz, G), tgt_ignore_index, dtype=torch.long, device=device)
    key_padding_mask = torch.ones((bsz, T), device=device, dtype=torch.bool)
    for i, (s, t) in enumerate(zip(src_tensors, tgt_tensors)):
        ti = int(s.shape[0])
        gi = int(t.shape[0])
        src[i, :ti] = s.to(device, non_blocking=True)
        tgt[i, :gi] = t.to(device, non_blocking=True)
        key_padding_mask[i, :ti] = False
    return src, tgt, key_padding_mask, torch.tensor([int(s.shape[0]) for s in src_tensors], device=device, dtype=torch.long)


_SORTED_BY_LEN_CACHE: dict[int, np.ndarray] = {}


def _indices_sorted_by_src_len(pool: TensorizedDataset) -> np.ndarray:
    """Return indices into pool sorted by source sequence length (cached per pool)."""
    key = id(pool)
    if key not in _SORTED_BY_LEN_CACHE:
        n = pool.n
        lengths = np.array([int(pool.src_list[i].shape[0]) for i in range(n)], dtype=np.int64)
        _SORTED_BY_LEN_CACHE[key] = np.argsort(lengths)
    return _SORTED_BY_LEN_CACHE[key]


def prepare_batch(
    *,
    batch_size: int,
    train_pool: TensorizedDataset,
    device: torch.device,
    cpu_generator: torch.Generator,
    augment: Optional[AugmentSpec] = None,
    grid_size: Optional[int] = None,
    num_demos: Optional[int] = None,
    T_max: Optional[int] = None,
    G_max: Optional[int] = None,
    group_by_length: bool = True,
) -> Batch:
    """
    Sample a batch and pad to T_max/G_max when provided (so model sees fixed layout), else to batch max.
    When group_by_length is True, samples a contiguous segment from indices sorted by sequence length
    so that batches have similar lengths and less padding (faster training).
    """
    bsz = int(batch_size)
    n = int(train_pool.n)
    pool_device = train_pool.src_list[0].device if n > 0 else torch.device("cpu")

    if group_by_length and n >= bsz:
        sorted_idx = _indices_sorted_by_src_len(train_pool)
        start = int(torch.randint(0, n - bsz + 1, (1,), generator=cpu_generator, device=torch.device("cpu")).item())
        batch_idx = sorted_idx[start : start + bsz]
        idx = torch.from_numpy(batch_idx).to(device=pool_device, dtype=torch.long)
    elif pool_device.type == "cpu":
        idx = torch.randint(
            low=0,
            high=n,
            size=(bsz,),
            device=torch.device("cpu"),
            generator=cpu_generator,
            dtype=torch.long,
        )
    else:
        idx = torch.randint(
            low=0,
            high=n,
            size=(bsz,),
            device=pool_device,
            dtype=torch.long,
        )
    src, tgt, key_padding_mask, src_lengths = _pad_batch_variable(
        train_pool.src_list,
        train_pool.tgt_list,
        idx,
        device,
        pad_token=int(PAD_TOKEN),
        T_max=T_max,
        G_max=G_max,
    )
    g_each = train_pool.grid_size_each[idx].to(device, non_blocking=True)
    nd_each = train_pool.num_demos_each[idx].to(device, non_blocking=True)
    g_in = int(train_pool.grid_size)
    # Augmentation is applied per-example to content slices only (no padding passed to augment).
    # Flip/transform therefore never see PAD; padding in the batch is left unchanged.
    if augment is not None and bool(augment.enabled):
        gen = cpu_generator if device.type == "cpu" else None
        for i in range(bsz):
            nd_i = int(nd_each[i].item())
            g_out_i = int(g_each[i].item())
            len_i = int(src_lengths[i].item())
            g_out_sq = g_out_i * g_out_i
            expected_len = prompt_seq_len(grid_size=g_in, num_demos=nd_i, output_grid_size=g_out_i)
            if len_i != expected_len or int(tgt.shape[1]) < g_out_sq:
                continue
            aug_src, aug_tgt = augment_src_tgt_batch(
                src=src[i : i + 1, :len_i],
                tgt=tgt[i : i + 1, :g_out_sq],
                grid_size=g_in,
                num_demos=nd_i,
                output_grid_size=g_out_i,
                generator=gen,
                spec=augment,
            )
            src[i, :len_i].copy_(aug_src[0])
            tgt[i, :g_out_sq].copy_(aug_tgt[0])
    T = int(src.shape[1])
    G_max_batch = int(tgt.shape[1])
    pred_pos = torch.arange(G_max_batch, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, G_max_batch)
    pred_mask = tgt != -100
    return Batch(
        src=src,
        tgt=tgt,
        grid_size=g_each,
        num_demos=nd_each,
        pred_pos=pred_pos,
        pred_mask=pred_mask,
        key_padding_mask=key_padding_mask,
    )


def _pin_if_cuda(t: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if device.type != "cuda":
        return t
    if t.device.type != "cpu":
        return t
    return t.pin_memory()


def maybe_move_train_pool(
    ds: TensorizedDataset,
    *,
    device: torch.device,
    dataset_device: str,
) -> TensorizedDataset:
    """
    Move or pin training pools to avoid CPU bottlenecks.
    """
    mode = str(dataset_device).lower()
    if mode not in {"cpu", "gpu"}:
        raise ValueError(f"dataset_device must be 'cpu' or 'gpu', got {dataset_device!r}")
    if mode == "gpu":
        if device.type != "cuda":
            return ds
        out_g = int(ds.output_grid_size) if int(ds.output_grid_size) > 0 else int(ds.grid_size)
        return TensorizedDataset(
            skill_id=ds.skill_id,
            split=ds.split,
            grid_size=ds.grid_size,
            num_demos=ds.num_demos,
            src_list=[t.to(device) for t in ds.src_list],
            tgt_list=[t.to(device) for t in ds.tgt_list],
            grid_size_each=ds.grid_size_each.to(device),
            num_demos_each=ds.num_demos_each.to(device),
            output_grid_size=out_g,
        )
    out_g = int(ds.output_grid_size) if int(ds.output_grid_size) > 0 else int(ds.grid_size)
    return TensorizedDataset(
        skill_id=ds.skill_id,
        split=ds.split,
        grid_size=ds.grid_size,
        num_demos=ds.num_demos,
        src_list=[_pin_if_cuda(t, device=device) for t in ds.src_list],
        tgt_list=[_pin_if_cuda(t, device=device) for t in ds.tgt_list],
        grid_size_each=_pin_if_cuda(ds.grid_size_each, device=device),
        num_demos_each=_pin_if_cuda(ds.num_demos_each, device=device),
        output_grid_size=out_g,
    )


@dataclass
class LearningCurves:
    steps: list[int]
    loss: list[float]
    acc_train: dict[int, list[float]]
    acc_id: dict[int, list[float]]
    acc_ood: dict[int, list[float]]
    probe_train_ood: list[float]
    probe_fully_heldout_ood: list[float]

    def ensure_skill(self, sid: int) -> None:
        if sid not in self.acc_train:
            self.acc_train[sid] = []
        if sid not in self.acc_id:
            self.acc_id[sid] = []
        if sid not in self.acc_ood:
            self.acc_ood[sid] = []


def write_learning_curves_csv(
    *,
    curves: LearningCurves,
    skills: list[int],
    out_path: Path,
) -> None:
    """
    Save a "wide" CSV of all tracked metrics at each eval step.

    Columns:
      - step, loss
      - probe_train_ood, probe_fully_heldout_ood
      - train_acc_s{sid}, id_acc_s{sid}, ood_acc_s{sid} for sid in `skills`
    """
    import csv

    if len(curves.steps) != len(curves.loss):
        raise ValueError("LearningCurves has inconsistent steps/loss lengths.")
    if len(curves.probe_train_ood) != len(curves.steps):
        raise ValueError("LearningCurves has inconsistent probe_train_ood length.")
    if len(curves.probe_fully_heldout_ood) != len(curves.steps):
        raise ValueError("LearningCurves has inconsistent probe_fully_heldout_ood length.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols: list[str] = ["step", "loss", "probe_train_ood", "probe_fully_heldout_ood"]
    for sid in skills:
        cols.append(f"train_acc_s{int(sid)}")
        cols.append(f"id_acc_s{int(sid)}")
        cols.append(f"ood_acc_s{int(sid)}")

    rows: list[dict[str, float | int]] = []
    for i, step in enumerate(curves.steps):
        r: dict[str, float | int] = {
            "step": int(step),
            "loss": float(curves.loss[i]),
            "probe_train_ood": float(curves.probe_train_ood[i]),
            "probe_fully_heldout_ood": float(curves.probe_fully_heldout_ood[i]),
        }
        for sid in skills:
            sid_i = int(sid)
            tr = curves.acc_train.get(sid_i, [])
            idd = curves.acc_id.get(sid_i, [])
            ood = curves.acc_ood.get(sid_i, [])
            if len(tr) != len(curves.steps) or len(idd) != len(curves.steps) or len(ood) != len(curves.steps):
                raise ValueError(f"LearningCurves has inconsistent metric lengths for skill {sid_i}.")
            r[f"train_acc_s{sid_i}"] = float(tr[i])
            r[f"id_acc_s{sid_i}"] = float(idd[i])
            r[f"ood_acc_s{sid_i}"] = float(ood[i])
        rows.append(r)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def count_params(model: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def plot_learning_curves(
    *,
    curves: LearningCurves,
    skills: list[int],
    out_path: Path,
    title: str,
) -> None:
    # Keep training usable in minimal environments: CSV saving is the source of truth,
    # and plotting is best-effort.
    if not _has_matplotlib():
        return

    import matplotlib.pyplot as plt  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Matplotlib style names vary across versions; choose the best available without throwing.
    style_candidates = ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]
    avail = set(getattr(plt.style, "available", []))
    for s in style_candidates:
        if s in avail:
            plt.style.use(s)
            break
    fig = plt.figure(figsize=(12, 7), dpi=140)
    ax = fig.add_subplot(1, 1, 1)

    # Distinct colors per skill; solid=id, dashed=ood.
    palette = {
        1: "#1f77b4",
        2: "#ff7f0e",
        3: "#2ca02c",
        4: "#d62728",
        5: "#9467bd",
    }

    x = curves.steps
    for sid in skills:
        color = palette.get(sid, None)
        ax.plot(x, curves.acc_id.get(sid, []), label=f"s{sid} id", color=color, linewidth=2.0)
        ax.plot(
            x,
            curves.acc_ood.get(sid, []),
            label=f"s{sid} ood",
            color=color,
            linewidth=2.0,
            linestyle="--",
            alpha=0.9,
        )

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.minorticks_on()

    # Legend outside the plot.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10)

    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(out_path)
    plt.close(fig)


@torch.no_grad()
def evaluate_accuracy(
    *,
    model: nn.Module,
    rng: np.random.Generator,
    n_tasks: int,
    device: torch.device,
    grid_tokens: int,
    dataset: TensorizedDataset,
    eval_batch_size: int,
    vote_augs: int = 0,
    vote_spec: Optional[AugmentSpec] = None,
    save_unsolved_dir: Optional[Path] = None,
    save_unsolved_max: int = 0,
    save_unsolved_step: Optional[int] = None,
    save_unsolved_tag: str = "test",
    save_solved_dir: Optional[Path] = None,
    save_solved_max: int = 0,
    save_solved_step: Optional[int] = None,
    save_solved_tag: str = "test",
    save_augmented_dir: Optional[Path] = None,
    save_augmented_max: int = 0,
    save_augmented_step: Optional[int] = None,
    save_augmented_tag: str = "test",
    save_augmented_spec: Optional[AugmentSpec] = None,
    print_solved_max: int = 0,
    print_solved_step: Optional[int] = None,
    print_solved_tag: str = "test",
    show_progress: bool = False,
) -> float:
    """
    Evaluate accuracy on a random subset of the dataset (at most n_tasks examples).
    """
    model.eval()
    k = min(int(n_tasks), dataset.n)
    if k <= 0:
        return 0.0

    idx_np = rng.choice(dataset.n, size=k, replace=False)
    idx_full = torch.as_tensor(idx_np, dtype=torch.long)
    pool_dev = dataset.src_list[0].device if dataset.n > 0 else torch.device("cpu")
    grid_each = dataset.grid_size_each.index_select(0, idx_full).to(device)

    bs = max(1, int(eval_batch_size))
    num_batches = (k + bs - 1) // bs
    batch_offsets = range(0, k, bs)
    if show_progress and _has_tqdm():
        from tqdm import tqdm  # type: ignore

        batch_offsets = tqdm(
            batch_offsets,
            total=num_batches,
            desc="eval",
            unit="it",
            unit_scale=False,
            smoothing=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
    correct = 0
    saved_unsolved = 0
    saved_solved = 0
    saved_augmented = 0
    printed = 0
    for off in batch_offsets:
        batch_idx = idx_full[off : off + bs]
        xb, yb, key_padding_mask, _ = _pad_batch_variable(
            dataset.src_list,
            dataset.tgt_list,
            batch_idx,
            device,
            pad_token=int(PAD_TOKEN),
        )
        valid = yb != -100
        grid_tokens_batch = int(yb.shape[1])
        g_batch = int(round(grid_tokens_batch**0.5))
        nd_batch = int(dataset.num_demos)

        if int(vote_augs) > 0 and (vote_spec is None or not bool(vote_spec.enabled)):
            raise ValueError("vote_augs > 0 requires vote_spec.enabled=True")
        # Only run vote_augs when batch has a single layout (same content length, inferrable g_in).
        do_vote = (
            int(vote_augs) > 0
            and vote_spec is not None
            and bool(vote_spec.enabled)
        )
        if do_vote:
            content_lens = (key_padding_mask.logical_not()).sum(dim=1).cpu().tolist()
            if len(set(content_lens)) != 1:
                do_vote = False
            else:
                g_in = infer_input_grid_size(content_lens[0], nd_batch, g_batch)
                if g_in is None:
                    do_vote = False

        if do_vote:
            bsz = int(xb.shape[0])
            best_pred = torch.empty((bsz, grid_tokens_batch), dtype=torch.long, device="cpu")
            best_count = [0 for _ in range(bsz)]
            counts: list[dict[bytes, int]] = [dict() for _ in range(bsz)]
            g_in = infer_input_grid_size(content_lens[0], nd_batch, g_batch)

            for _ in range(int(vote_augs)):
                xb_aug, yb_aug, params = augment_src_tgt_batch_with_params(
                    src=xb,
                    tgt=yb,
                    grid_size=g_in,
                    num_demos=nd_batch,
                    output_grid_size=g_batch,
                    generator=None,
                    spec=vote_spec,
                )
                logits = model(xb_aug, key_padding_mask=key_padding_mask)
                pred_logits = logits[:, -(grid_tokens_batch + 1) : -1, :]
                pred = torch.argmax(pred_logits, dim=-1).reshape(bsz, g_batch, g_batch).unsqueeze(1)
                pred_inv = invert_grids_torch(pred, params=params).squeeze(1).reshape(bsz, grid_tokens_batch)
                pred_inv = torch.where(valid, pred_inv, torch.zeros_like(pred_inv))
                pred_cpu = pred_inv.detach().to("cpu")

                for bi in range(bsz):
                    key = pred_cpu[bi].numpy().tobytes()
                    c = counts[bi].get(key, 0) + 1
                    counts[bi][key] = c
                    if c > best_count[bi]:
                        best_count[bi] = c
                        best_pred[bi] = pred_cpu[bi]

            pred_final = best_pred.to(device=device)
        else:
            logits = model(xb, key_padding_mask=key_padding_mask)
            pred_logits = logits[:, -(grid_tokens_batch + 1) : -1, :]
            pred_final = torch.argmax(pred_logits, dim=-1)

        # Exact match on valid cells only.
        eq = torch.where(valid, pred_final == yb, torch.ones_like(valid, dtype=torch.bool)).all(dim=1)
        correct += int(eq.sum().item())

        # Print a few solved examples (small grids only).
        if int(print_solved_max) > 0 and printed < int(print_solved_max):
            bi_solved = (eq).nonzero(as_tuple=False).reshape(-1).tolist()
            if len(bi_solved) > 0:
                # Decode + print on CPU.
                g_each_b = grid_each[off : off + bs].detach().cpu().numpy()
                xb_cpu = xb.detach().cpu().numpy()
                yb_cpu = yb.detach().cpu().numpy()
                pred_cpu = pred_final.detach().cpu().numpy()

                def crop(a: np.ndarray, g: int) -> np.ndarray:
                    gg = int(g)
                    return a[:gg, :gg]

                step_s = "na" if print_solved_step is None else f"{int(print_solved_step):07d}"
                nd_each_b = dataset.num_demos_each[batch_idx].detach().cpu().numpy()
                for bi in bi_solved:
                    if printed >= int(print_solved_max):
                        break
                    g_i = int(g_each_b[int(bi)])
                    if g_i <= 0:
                        continue
                    ds_idx_i = int(idx_np[int(off + bi)])
                    g_out_i = int(round(dataset.tgt_list[ds_idx_i].shape[0] ** 0.5))
                    nd_i = int(nd_each_b[int(bi)])
                    content_len = int((key_padding_mask[int(bi)].logical_not()).sum().item())
                    g_in = infer_input_grid_size(content_len, nd_i, g_out_i)
                    if g_in is None:
                        continue
                    _demos, test_x = _decode_prompt_src(
                        src_tokens=xb_cpu[int(bi)][:content_len],
                        grid_size=g_in,
                        num_demos=nd_i,
                        output_grid_size=g_out_i,
                    )
                    test_x = crop(test_x, g_i)
                    true_y = crop(yb_cpu[int(bi)].reshape(g_batch, g_batch), g_i)
                    pred_y = crop(pred_cpu[int(bi)].reshape(g_batch, g_batch), g_i)

                    print(
                        f"\n=== solved example | tag={print_solved_tag} | s{int(dataset.skill_id)} | split={dataset.split} | "
                        f"step={step_s} | idx={ds_idx_i} | grid={g_i}x{g_i} ===",
                        flush=True,
                    )
                    print("test_x:", flush=True)
                    print(render_ascii(test_x), flush=True)
                    print("\npred_y:", flush=True)
                    print(render_ascii(pred_y), flush=True)
                    print("\ntrue_y:", flush=True)
                    print(render_ascii(true_y), flush=True)
                    printed += 1

        # Save a fixed set of "latest" example images (slot00..slotN) so evals overwrite in-place
        # instead of creating new files every time.
        if save_unsolved_dir is not None and int(save_unsolved_max) > 0 and saved_unsolved < int(save_unsolved_max):
            step_s = "na" if save_unsolved_step is None else f"{int(save_unsolved_step):07d}"
            base_dir = (
                Path(save_unsolved_dir) / f"{save_unsolved_tag}" / f"s{int(dataset.skill_id)}" / f"{dataset.split}"
            )
            bad = (~eq).nonzero(as_tuple=False).reshape(-1).tolist()
            if len(bad) > 0:
                while saved_unsolved < int(save_unsolved_max):
                    bi = int(bad[saved_unsolved % len(bad)])
                    # Decode + save on CPU.
                    ds_idx = int(idx_np[int(off + bi)])
                    g_out_i = int(round(dataset.tgt_list[ds_idx].shape[0] ** 0.5))
                    nd_i = int(dataset.num_demos_each[ds_idx].item())
                    content_len = int((key_padding_mask[bi].logical_not()).sum().item())
                    g_in = infer_input_grid_size(content_len, nd_i, g_out_i)
                    if g_in is None:
                        saved_unsolved += 1
                        if saved_unsolved >= int(save_unsolved_max):
                            break
                        continue
                    src_i = xb[bi].detach().cpu().numpy()
                    demos, test_x = _decode_prompt_src(
                        src_tokens=src_i[:content_len],
                        grid_size=g_in,
                        num_demos=nd_i,
                        output_grid_size=g_out_i,
                    )
                    g_i = int(grid_each[off + bi].item())
                    true_y = yb[bi].detach().cpu().numpy().reshape(g_batch, g_batch)[:g_i, :g_i]
                    pred_y = pred_final[bi].detach().cpu().numpy().reshape(g_batch, g_batch)[:g_i, :g_i]

                    out_path = base_dir / f"slot{int(saved_unsolved):02d}.png"
                    title = (
                        f"{save_unsolved_tag} latest (unsolved) | s{int(dataset.skill_id)} | split={dataset.split} | "
                        f"step={step_s} | idx={ds_idx} | slot={int(saved_unsolved):02d}"
                    )
                    save_arc_prompt_prediction_png(
                        demos=demos, test_x=test_x, pred_y=pred_y, true_y=true_y, out_path=out_path, title=title
                    )
                    saved_unsolved += 1
                    if saved_unsolved >= int(save_unsolved_max):
                        break

        if save_solved_dir is not None and int(save_solved_max) > 0 and saved_solved < int(save_solved_max):
            step_s = "na" if save_solved_step is None else f"{int(save_solved_step):07d}"
            base_dir = Path(save_solved_dir) / f"{save_solved_tag}" / f"s{int(dataset.skill_id)}" / f"{dataset.split}"
            good = (eq).nonzero(as_tuple=False).reshape(-1).tolist()
            if len(good) > 0:
                while saved_solved < int(save_solved_max):
                    bi = int(good[saved_solved % len(good)])
                    ds_idx = int(idx_np[int(off + bi)])
                    g_out_i = int(round(dataset.tgt_list[ds_idx].shape[0] ** 0.5))
                    nd_i = int(dataset.num_demos_each[ds_idx].item())
                    content_len = int((key_padding_mask[bi].logical_not()).sum().item())
                    g_in = infer_input_grid_size(content_len, nd_i, g_out_i)
                    if g_in is None:
                        saved_solved += 1
                        if saved_solved >= int(save_solved_max):
                            break
                        continue
                    src_i = xb[bi].detach().cpu().numpy()
                    demos, test_x = _decode_prompt_src(
                        src_tokens=src_i[:content_len],
                        grid_size=g_in,
                        num_demos=nd_i,
                        output_grid_size=g_out_i,
                    )
                    g_i = int(grid_each[off + bi].item())
                    true_y = yb[bi].detach().cpu().numpy().reshape(g_batch, g_batch)[:g_i, :g_i]
                    pred_y = pred_final[bi].detach().cpu().numpy().reshape(g_batch, g_batch)[:g_i, :g_i]
                    out_path = base_dir / f"slot{int(saved_solved):02d}.png"
                    title = (
                        f"{save_solved_tag} latest (solved) | s{int(dataset.skill_id)} | split={dataset.split} | "
                        f"step={step_s} | idx={ds_idx} | slot={int(saved_solved):02d}"
                    )
                    save_arc_prompt_prediction_png(
                        demos=demos, test_x=test_x, pred_y=pred_y, true_y=true_y, out_path=out_path, title=title
                    )
                    saved_solved += 1
                    if saved_solved >= int(save_solved_max):
                        break

        if (
            save_augmented_dir is not None
            and int(save_augmented_max) > 0
            and saved_augmented < int(save_augmented_max)
            and save_augmented_spec is not None
            and bool(save_augmented_spec.enabled)
        ):
            step_s = "na" if save_augmented_step is None else f"{int(save_augmented_step):07d}"
            base_dir = (
                Path(save_augmented_dir) / f"{save_augmented_tag}" / f"s{int(dataset.skill_id)}" / f"{dataset.split}"
            )
            bsz = int(xb.shape[0])
            while saved_augmented < int(save_augmented_max):
                bi = int(saved_augmented % max(1, bsz))
                ds_idx = int(idx_np[int(off + bi)])
                g_out_i = int(round(dataset.tgt_list[ds_idx].shape[0] ** 0.5))
                nd_i = int(dataset.num_demos_each[ds_idx].item())
                content_len = int((key_padding_mask[bi].logical_not()).sum().item())
                g_in = infer_input_grid_size(content_len, nd_i, g_out_i)
                if g_in is None:
                    saved_augmented += 1
                    if saved_augmented >= int(save_augmented_max):
                        break
                    continue
                g_out_sq = g_out_i * g_out_i
                xb1 = xb[bi : bi + 1, :content_len]
                yb1 = yb[bi : bi + 1, :g_out_sq]
                xb_aug, yb_aug = augment_src_tgt_batch(
                    src=xb1,
                    tgt=yb1,
                    grid_size=g_in,
                    num_demos=nd_i,
                    output_grid_size=g_out_i,
                    generator=None,
                    spec=save_augmented_spec,
                )
                # Augmented output has same content length; pad to batch dims for model.
                xb_aug_pad = torch.full(
                    (1, int(xb.shape[1])),
                    int(PAD_TOKEN),
                    dtype=xb_aug.dtype,
                    device=xb_aug.device,
                )
                xb_aug_pad[:, : content_len] = xb_aug
                yb_aug_pad = torch.full(
                    (1, int(yb.shape[1])),
                    -100,
                    dtype=yb_aug.dtype,
                    device=yb_aug.device,
                )
                yb_aug_pad[:, :g_out_sq] = yb_aug
                key_pad_bi = key_padding_mask[bi : bi + 1].clone()
                logits_aug = model(xb_aug_pad, key_padding_mask=key_pad_bi)
                pred_logits_aug = logits_aug[:, -(grid_tokens_batch + 1) : -1, :]
                pred_aug = torch.argmax(pred_logits_aug, dim=-1)

                src_i = xb_aug[0].detach().cpu().numpy()
                demos, test_x = _decode_prompt_src(
                    src_tokens=src_i,
                    grid_size=g_in,
                    num_demos=nd_i,
                    output_grid_size=g_out_i,
                )
                true_y = yb_aug[0].detach().cpu().numpy().reshape(g_out_i, g_out_i)
                pred_y = (
                    pred_aug[0].detach().cpu().numpy().reshape(g_batch, g_batch)[:g_out_i, :g_out_i].copy()
                )

                out_path = base_dir / f"slot{int(saved_augmented):02d}.png"
                title = (
                    f"{save_augmented_tag} latest (augmented) | s{int(dataset.skill_id)} | split={dataset.split} | "
                    f"step={step_s} | idx={ds_idx} | slot={int(saved_augmented):02d}"
                )
                save_arc_prompt_prediction_png(
                    demos=demos, test_x=test_x, pred_y=pred_y, true_y=true_y, out_path=out_path, title=title
                )
                saved_augmented += 1
                if saved_augmented >= int(save_augmented_max):
                    break

    return float(correct) / float(k)


@torch.no_grad()
def show_one_example(
    *,
    model: nn.Module,
    dataset: TensorizedDataset,
    device: torch.device,
    grid_size: Optional[int] = None,  # deprecated; use dataset grid sizes
) -> None:
    i = 0
    src_i = dataset.src_list[i].unsqueeze(0).to(device)
    g_out = int(round(dataset.tgt_list[i].shape[0] ** 0.5))
    grid_tokens = g_out * g_out
    tgt = dataset.tgt_list[i].cpu().numpy().reshape(g_out, g_out)
    key_pad = torch.zeros((1, int(src_i.shape[1])), device=device, dtype=torch.bool)

    logits = model(src_i, key_padding_mask=key_pad)
    pred_logits = logits[:, -(grid_tokens + 1) : -1, :]
    pred = torch.argmax(pred_logits, dim=-1).cpu().numpy().reshape(g_out, g_out)

    tokens = dataset.src_list[i].cpu().numpy()
    nd = int(dataset.num_demos_each[i].item())
    g_in = infer_input_grid_size(int(tokens.shape[0]), nd, g_out)
    if g_in is None:
        print(f"Cannot infer input grid size for seq_len={tokens.shape[0]} num_demos={nd} output_grid_size={g_out}")
        return
    demos, test_x = _decode_prompt_src(
        src_tokens=tokens,
        grid_size=g_in,
        num_demos=nd,
        output_grid_size=g_out,
    )

    print(f"\n=== Skill {dataset.skill_id} | split={dataset.split} ===")
    print("Demo 1:")
    x0, y0 = demos[0]
    print(render_ascii(x0))
    print(" ->")
    print(render_ascii(y0))
    print("\nTest x:")
    print(render_ascii(test_x))
    print("\nPred y:")
    print(render_ascii(pred))
    print("\nTrue y:")
    print(render_ascii(tgt))


