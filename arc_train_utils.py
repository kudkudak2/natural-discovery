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


# --- Unit test runner (used by training scripts) ---
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


def prompt_seq_len(*, grid_size: int, num_demos: int = 3) -> int:
    """
    Prompt layout (fixed):
      (x SEP y SEP) repeated `num_demos` times, then (test_x SEP)
    where x/y/test_x are grid_size*grid_size tokens.
    """
    g = int(grid_size) * int(grid_size)
    return int(num_demos) * (g + 1 + g + 1) + (g + 1)


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

        return tqdm(iterable, total=total, desc=desc)
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
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Inverse of `_flatten_prompt` for visualization/debug.
    Layout: (x SEP y SEP) repeated `num_demos` times, then (test_x SEP).
    """
    if src_tokens.ndim != 1:
        raise ValueError(f"Expected 1D src_tokens, got shape={src_tokens.shape}")
    g = int(grid_size)
    if g <= 0:
        raise ValueError(f"grid_size must be >= 1, got {g}")
    grid_tokens = g * g
    expected = prompt_seq_len(grid_size=g, num_demos=int(num_demos))
    if int(src_tokens.shape[0]) != int(expected):
        raise ValueError(f"Unexpected src length={int(src_tokens.shape[0])} (expected {expected})")

    def unflatten(block: np.ndarray) -> np.ndarray:
        return np.asarray(block, dtype=np.int64).reshape(g, g)

    demos: list[tuple[np.ndarray, np.ndarray]] = []
    off = 0
    for _ in range(int(num_demos)):
        x = unflatten(src_tokens[off : off + grid_tokens])
        off += grid_tokens
        if int(src_tokens[off]) != int(SEP_TOKEN):
            raise ValueError(f"Expected SEP after demo x at off={off}, got {int(src_tokens[off])}")
        off += 1

        y = unflatten(src_tokens[off : off + grid_tokens])
        off += grid_tokens
        if int(src_tokens[off]) != int(SEP_TOKEN):
            raise ValueError(f"Expected SEP after demo y at off={off}, got {int(src_tokens[off])}")
        off += 1
        demos.append((x, y))

    test_x = unflatten(src_tokens[off : off + grid_tokens])
    off += grid_tokens
    if int(src_tokens[off]) != int(SEP_TOKEN):
        raise ValueError(f"Expected trailing SEP after test_x at off={off}, got {int(src_tokens[off])}")
    return demos, test_x


def decode_prompt_src(
    *,
    src_tokens: np.ndarray,
    grid_size: int,
    num_demos: int = 3,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Public wrapper for `_decode_prompt_src` (used by other scripts for plotting/debug)."""
    return _decode_prompt_src(src_tokens=src_tokens, grid_size=grid_size, num_demos=num_demos)


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
    if g <= 0:
        raise ValueError(f"grid_size must be >= 1, got {g}")
    if nd <= 0:
        raise ValueError(f"num_demos must be >= 1, got {nd}")

    tasks: list[ARCTask] = []
    skipped_too_few_demos = 0
    skipped_too_large = 0
    for demo_xs, demo_ys, test_x, test_y, tid_suf in raw_tasks:
        if int(len(demo_xs)) < nd or int(len(demo_ys)) < nd:
            skipped_too_few_demos += 1
            continue
        mx = 0
        for gg in demo_xs[:nd] + demo_ys[:nd] + [test_x, test_y]:
            h, w = _grid_dims(gg)
            mx = max(int(mx), int(h), int(w))
        if int(mx) > int(g):
            skipped_too_large += 1
            continue

        demos: list[ARCExamplePair] = []
        for i in range(nd):
            demos.append(ARCExamplePair(x=_pad_to_square(demo_xs[i], size=g), y=_pad_to_square(demo_ys[i], size=g)))
        test = ARCTestCase(x=_pad_to_square(test_x, size=g), y=_pad_to_square(test_y, size=g))

        task_id = _stable_id(dataset_id, str(split), tid_suf)
        tasks.append(
            ARCTask(
                task_id=str(task_id),
                skill_id=0,
                skill_name="external_arc",
                grid_size=int(g),
                demos=demos,
                test=test,
            )
        )

    if len(tasks) == 0:
        raise ValueError(f"External dataset split {split!r} produced no usable tasks (nd={nd}).")
    return ARCDataset(
        dataset_id=str(dataset_id),
        created_at=ARCDataset.now_iso(),
        split=str(split),
        ood=False,
        skills=[0],
        grid_size=int(g),
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

    if int(num_demos) > 0:
        nd = int(num_demos)
    else:
        nd = min(int(len(dx)) for (dx, _dy, _tx, _ty, _tid) in raw_for_infer)
        nd = max(1, int(nd))

    g_infer = int(grid_size) if int(grid_size) > 0 else _infer_external_grid_size(raw_for_infer)
    if max_seq_len is not None and int(max_seq_len) > 0:
        g_cap = _max_grid_size_for_seq_len(max_seq_len=int(max_seq_len), num_demos=int(nd))
        if int(g_cap) <= 0:
            raise ValueError(
                f"max_seq_len={int(max_seq_len)} is too small for num_demos={int(nd)} "
                f"(minimum is prompt_seq_len(grid_size=1,num_demos={int(nd)})={int(prompt_seq_len(grid_size=1, num_demos=int(nd)))})."
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
            ds_train = _build_external_arcdataset(raw_tasks=raw_tr, split="train", grid_size=g, num_demos=nd, dataset_id=dataset_id)
            ds_eval = _build_external_arcdataset(
                raw_tasks=raw_ev, split="evaluation", grid_size=g, num_demos=nd, dataset_id=dataset_id
            )
            if max_seq_len is not None and int(max_seq_len) > 0:
                tr_sk = int(getattr(ds_train, "extra", {}).get("skipped_too_large", 0))
                ev_sk = int(getattr(ds_eval, "extra", {}).get("skipped_too_large", 0))
                if tr_sk > 0 or ev_sk > 0:
                    print(
                        f"[max_seq_len={int(max_seq_len)}] external_arc[{name}] skipped_too_large: "
                        f"train={tr_sk} evaluation={ev_sk} (grid_size_cap={int(g)}, num_demos={int(nd)})",
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
        ds_all = _build_external_arcdataset(raw_tasks=raw_all, split="all", grid_size=g, num_demos=nd, dataset_id=dataset_id)
        if max_seq_len is not None and int(max_seq_len) > 0:
            sk = int(getattr(ds_all, "extra", {}).get("skipped_too_large", 0))
            if sk > 0:
                print(
                    f"[max_seq_len={int(max_seq_len)}] external_arc[{name}] skipped_too_large: "
                    f"all={sk} (grid_size_cap={int(g)}, num_demos={int(nd)})",
                    flush=True,
                )
        t_all = _tensorize_dataset(ds_all, max_seq_len=max_seq_len)
        tr, ev = split_dataset(t_all, train_frac=float(train_frac_for_unsplit), rng=rng)
        tr = TensorizedDataset(
            skill_id=0,
            split="train",
            grid_size=tr.grid_size,
            num_demos=tr.num_demos,
            src=tr.src,
            tgt=tr.tgt,
            grid_size_each=tr.grid_size_each,
            num_demos_each=tr.num_demos_each,
        )
        ev = TensorizedDataset(
            skill_id=0,
            split="evaluation",
            grid_size=ev.grid_size,
            num_demos=ev.num_demos,
            src=ev.src,
            tgt=ev.tgt,
            grid_size_each=ev.grid_size_each,
            num_demos_each=ev.num_demos_each,
        )
        assert_disjoint_datasets(a=tr, b=ev, label=f"external_generic[{name}]: train vs evaluation")
        out.append((name, tr, ev))

    return out


@dataclass(frozen=True)
class TensorizedDataset:
    skill_id: int
    split: str
    grid_size: int  # max grid size in this dataset tensorization
    num_demos: int  # max num_demos in this dataset tensorization
    src: torch.Tensor  # (N, T) padded with PAD_TOKEN
    tgt: torch.Tensor  # (N, Gmax) padded with ignore_index (-100)
    grid_size_each: torch.Tensor  # (N,) long
    num_demos_each: torch.Tensor  # (N,) long

    @property
    def n(self) -> int:
        return int(self.src.shape[0])


def _subset_dataset(ds: TensorizedDataset, idx: np.ndarray, *, split_suffix: str) -> TensorizedDataset:
    return TensorizedDataset(
        skill_id=ds.skill_id,
        split=f"{ds.split}_{split_suffix}",
        grid_size=ds.grid_size,
        num_demos=ds.num_demos,
        src=ds.src[idx],
        tgt=ds.tgt[idx],
        grid_size_each=ds.grid_size_each[idx],
        num_demos_each=ds.num_demos_each[idx],
    )


def _row_digests(ds: TensorizedDataset) -> set[bytes]:
    """
    Return a set of per-example cryptographic digests for exact disjointness checks.

    Digest is computed over the concatenation of (src row, tgt row) bytes to avoid
    false matches when src is equal but tgt differs (or vice versa).
    """
    if ds.n <= 0:
        return set()
    st = torch.cat([ds.src, ds.tgt], dim=1).contiguous()
    # Always hash on CPU for determinism across devices/dtypes.
    a = st.detach().cpu().numpy()
    out: set[bytes] = set()
    for i in range(int(a.shape[0])):
        out.add(hashlib.blake2b(a[i].tobytes(), digest_size=16).digest())
    return out


def _row_digest_list(ds: TensorizedDataset) -> list[bytes]:
    """
    Per-row digests aligned with dataset row indices.

    Used for deduplication and overlap reporting. Digest definition matches `_row_digests`.
    """
    if ds.n <= 0:
        return []
    st = torch.cat([ds.src, ds.tgt], dim=1).contiguous()
    a = st.detach().cpu().numpy()
    return [hashlib.blake2b(a[i].tobytes(), digest_size=16).digest() for i in range(int(a.shape[0]))]


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
    st = torch.cat([ds.src, ds.tgt], dim=1).contiguous()
    a = st.detach().cpu().numpy()
    digests: list[bytes] = [hashlib.blake2b(a[i].tobytes(), digest_size=16).digest() for i in range(int(a.shape[0]))]

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
    nd0 = int(non_empty[0].num_demos)
    g0 = int(non_empty[0].grid_size)
    t0 = int(non_empty[0].src.shape[1])
    gmax0 = int(non_empty[0].tgt.shape[1])
    for ds in non_empty[1:]:
        if int(ds.num_demos) != int(nd0):
            raise ValueError(f"Cannot concat datasets with different num_demos: {nd0} vs {int(ds.num_demos)}")
        if int(ds.grid_size) != int(g0):
            raise ValueError(f"Cannot concat datasets with different grid_size: {g0} vs {int(ds.grid_size)}")
        if int(ds.src.shape[1]) != int(t0) or int(ds.tgt.shape[1]) != int(gmax0):
            raise ValueError("Cannot concat datasets with different padding shapes.")
    src = torch.cat([ds.src for ds in non_empty], dim=0)
    tgt = torch.cat([ds.tgt for ds in non_empty], dim=0)
    grid_each = torch.cat([ds.grid_size_each for ds in non_empty], dim=0)
    demos_each = torch.cat([ds.num_demos_each for ds in non_empty], dim=0)
    return TensorizedDataset(
        skill_id=skill_id,
        split=split,
        grid_size=int(g0),
        num_demos=int(nd0),
        src=src,
        tgt=tgt,
        grid_size_each=grid_each,
        num_demos_each=demos_each,
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
    return TensorizedDataset(
        skill_id=ds.skill_id,
        split=f"{ds.split}_cap{cap_i}",
        grid_size=ds.grid_size,
        num_demos=ds.num_demos,
        src=ds.src[idx],
        tgt=ds.tgt[idx],
        grid_size_each=ds.grid_size_each[idx],
        num_demos_each=ds.num_demos_each[idx],
    )


def _tensorize_dataset(ds: ARCDataset, *, max_seq_len: Optional[int] = None) -> TensorizedDataset:
    # Per-task variability is supported. We encode every task into a *fixed* prompt layout
    # sized by the maxima across tasks in this dataset: (max_grid_size, max_num_demos).
    tasks = list(ds.tasks)
    if len(tasks) == 0:
        raise ValueError("Dataset has no tasks.")

    parsed: list[tuple[int, int, list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]] = []
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
        if test_in.ndim != 2 or int(test_in.shape[0]) != int(test_in.shape[1]):
            raise ValueError(f"Expected square test input grid, got shape={tuple(test_in.shape)}")
        g = int(test_in.shape[0])
        if test_out.ndim != 2 or int(test_out.shape[0]) != int(test_out.shape[1]) or int(test_out.shape[0]) != int(g):
            raise ValueError(f"Expected square test output grid matching input size {g}, got shape={tuple(test_out.shape)}")

        if max_seq_len is not None and int(max_seq_len) > 0:
            tlen = int(prompt_seq_len(grid_size=int(g), num_demos=int(nd)))
            if int(tlen) > int(max_seq_len):
                dropped += 1
                continue

        parsed.append((g, nd, demos, test_in, test_out))
        max_g = max(int(max_g), int(g))
        max_nd = max(int(max_nd), int(nd))

    # Enforce that the *dataset-level* fixed token budget fits, not just each individual task.
    # Since tokenization pads all tasks to (max_g, max_nd), mixing "large-g" tasks and "large-nd"
    # tasks can exceed max_seq_len even if no single task does.
    if max_seq_len is not None and int(max_seq_len) > 0 and len(parsed) > 0:
        cap = int(max_seq_len)
        while True:
            max_g = max(int(p[0]) for p in parsed)
            max_nd = max(int(p[1]) for p in parsed)
            tlen = int(prompt_seq_len(grid_size=int(max_g), num_demos=int(max_nd)))
            if int(tlen) <= int(cap):
                break

            # Candidate: drop all tasks at the current max grid size.
            g_vals = sorted({int(p[0]) for p in parsed})
            nd_vals = sorted({int(p[1]) for p in parsed})
            next_g = int(g_vals[-2]) if len(g_vals) >= 2 else 0
            next_nd = int(nd_vals[-2]) if len(nd_vals) >= 2 else 0

            cand_g = int(prompt_seq_len(grid_size=int(next_g), num_demos=int(max_nd))) if int(next_g) > 0 else 10**18
            cand_nd = int(prompt_seq_len(grid_size=int(max_g), num_demos=int(next_nd))) if int(next_nd) > 0 else 10**18

            # If neither axis can be reduced, we cannot satisfy the constraint.
            if int(cand_g) >= 10**18 and int(cand_nd) >= 10**18:
                raise ValueError(
                    f"After applying max_seq_len={int(cap)}, dataset cannot be tensorized within the budget "
                    f"(needs seq_len={int(tlen)} for grid_size={int(max_g)}, num_demos={int(max_nd)})."
                )

            drop_by_g = int(cand_g) <= int(cand_nd)
            if int(cand_g) <= int(cap) and int(cand_nd) > int(cap):
                drop_by_g = True
            elif int(cand_nd) <= int(cap) and int(cand_g) > int(cap):
                drop_by_g = False

            if drop_by_g:
                keep = [p for p in parsed if int(p[0]) != int(max_g)]
            else:
                keep = [p for p in parsed if int(p[1]) != int(max_nd)]

            removed = int(len(parsed) - len(keep))
            dropped += int(removed)
            parsed = keep
            if len(parsed) == 0:
                raise ValueError(
                    f"After applying max_seq_len={int(cap)}, dataset became empty (dropped {int(dropped)}/{int(total)} tasks)."
                )

    if len(parsed) == 0:
        if max_seq_len is not None and int(max_seq_len) > 0:
            raise ValueError(
                f"After applying max_seq_len={int(max_seq_len)}, dataset became empty (dropped {int(dropped)}/{int(total)} tasks)."
            )
        raise ValueError("Dataset became empty after parsing tasks.")

    max_T = int(prompt_seq_len(grid_size=int(max_g), num_demos=int(max_nd)))
    max_G = int(max_g * max_g)

    src = torch.full((len(parsed), int(max_T)), int(PAD_TOKEN), dtype=torch.long)
    tgt = torch.full((len(parsed), int(max_G)), -100, dtype=torch.long)
    grid_each = torch.empty((len(parsed),), dtype=torch.long)
    demos_each = torch.empty((len(parsed),), dtype=torch.long)

    def embed_grid(grid: np.ndarray, *, g: int) -> np.ndarray:
        # IMPORTANT: within-grid padding uses background color 0 (not PAD_TOKEN).
        out = np.full((int(max_g), int(max_g)), 0, dtype=np.int64)
        out[: int(g), : int(g)] = np.asarray(grid, dtype=np.int64)
        return out

    for i, (g, nd, demos, test_in, test_out) in enumerate(parsed):
        grid_each[i] = int(g)
        demos_each[i] = int(nd)

        demos_fixed: list[tuple[np.ndarray, np.ndarray]] = []
        # Missing demos are padded with background color 0 grids (not PAD_TOKEN).
        pad_grid = np.full((int(max_g), int(max_g)), 0, dtype=np.int64)
        for di in range(int(max_nd)):
            if di < int(nd):
                x, y = demos[int(di)]
                demos_fixed.append((embed_grid(x, g=g), embed_grid(y, g=g)))
            else:
                demos_fixed.append((pad_grid, pad_grid))

        test_in_big = embed_grid(test_in, g=g)
        seq = _flatten_prompt(demos_fixed, test_in_big)
        if int(len(seq)) != int(max_T):
            raise ValueError(f"Internal error: fixed prompt len={len(seq)} != max_T={max_T}")
        src[i] = torch.tensor(seq, dtype=torch.long)

        # Fill only valid target cells (top-left g x g) into the max_g x max_g flattened grid.
        to = np.asarray(test_out, dtype=np.int64)
        for r in range(int(g)):
            for c in range(int(g)):
                tgt[i, int(r * int(max_g) + c)] = int(to[int(r), int(c)])

    if max_seq_len is not None and int(max_seq_len) > 0:
        kept = int(len(parsed))
        cap = int(max_seq_len)
        dsid = getattr(ds, "dataset_id", "?")
        split_s = getattr(ds, "split", "?")
        print(
            f"[max_seq_len={cap}] filtered tasks for dataset_id={dsid} split={split_s}: "
            f"dropped={int(dropped)}/{int(total)} kept={kept} "
            f"final(grid_size={int(max_g)}, num_demos={int(max_nd)}, seq_len={int(max_T)})",
            flush=True,
        )

    return TensorizedDataset(
        skill_id=int(ds.skills[0]) if ds.skills else -1,
        split=str(ds.split),
        grid_size=int(max_g),
        num_demos=int(max_nd),
        src=src,
        tgt=tgt,
        grid_size_each=grid_each,
        num_demos_each=demos_each,
    )


def pad_dataset_to(ds: TensorizedDataset, *, grid_size: int, num_demos: int) -> TensorizedDataset:
    """
    Retokenize a dataset to a larger (grid_size, num_demos) budget by embedding
    old grids into the top-left corner and padding missing demos with PAD.
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

    old_g = int(ds.grid_size)
    old_nd = int(ds.num_demos)
    old_grid_tokens = int(old_g * old_g)
    new_grid_tokens = int(g * g)
    new_T = int(prompt_seq_len(grid_size=int(g), num_demos=int(nd)))

    out_src = torch.full((ds.n, int(new_T)), int(PAD_TOKEN), dtype=torch.long, device=ds.src.device)
    out_tgt = torch.full((ds.n, int(new_grid_tokens)), -100, dtype=torch.long, device=ds.tgt.device)

    for i in range(int(ds.n)):
        tokens = ds.src[i]
        off = 0
        seq: list[int] = []

        for di in range(int(nd)):
            if di < int(old_nd):
                x_flat = tokens[off : off + old_grid_tokens]
                off += old_grid_tokens
                off += 1  # SEP
                y_flat = tokens[off : off + old_grid_tokens]
                off += old_grid_tokens
                off += 1  # SEP
                x_old = x_flat.reshape(old_g, old_g)
                y_old = y_flat.reshape(old_g, old_g)
            else:
                # Missing demos are padded with background color 0 (not PAD_TOKEN).
                x_old = torch.zeros((old_g, old_g), dtype=torch.long, device=tokens.device)
                y_old = torch.zeros((old_g, old_g), dtype=torch.long, device=tokens.device)

            # IMPORTANT: within-grid padding uses background color 0 (not PAD_TOKEN).
            x_big = torch.zeros((g, g), dtype=torch.long, device=tokens.device)
            y_big = torch.zeros((g, g), dtype=torch.long, device=tokens.device)
            x_big[:old_g, :old_g] = x_old
            y_big[:old_g, :old_g] = y_old

            seq += x_big.reshape(-1).tolist() + [int(SEP_TOKEN)] + y_big.reshape(-1).tolist() + [int(SEP_TOKEN)]

        # test_x from old prompt
        test_x_flat = tokens[off : off + old_grid_tokens]
        test_x_old = test_x_flat.reshape(old_g, old_g)
        test_x_big = torch.zeros((g, g), dtype=torch.long, device=tokens.device)
        test_x_big[:old_g, :old_g] = test_x_old
        seq += test_x_big.reshape(-1).tolist() + [int(SEP_TOKEN)]

        if int(len(seq)) != int(new_T):
            raise ValueError("Internal error: retokenized prompt has wrong length.")
        out_src[i] = torch.tensor(seq, dtype=torch.long, device=out_src.device)

        t_old = ds.tgt[i].reshape(old_g, old_g)
        t_big = torch.full((g, g), -100, dtype=torch.long, device=out_tgt.device)
        t_big[:old_g, :old_g] = t_old
        out_tgt[i] = t_big.reshape(-1)

    return TensorizedDataset(
        skill_id=ds.skill_id,
        split=ds.split,
        grid_size=int(g),
        num_demos=int(nd),
        src=out_src,
        tgt=out_tgt,
        grid_size_each=ds.grid_size_each.to(device=ds.src.device),
        num_demos_each=ds.num_demos_each.to(device=ds.src.device),
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
    max_nd = max(int(p.num_demos) for p in parts)
    padded = [pad_dataset_to(p, grid_size=int(max_g), num_demos=int(max_nd)) for p in parts]

    src = torch.cat([p.src for p in padded], dim=0)
    tgt = torch.cat([p.tgt for p in padded], dim=0)
    grid_each = torch.cat([p.grid_size_each for p in padded], dim=0)
    demos_each = torch.cat([p.num_demos_each for p in padded], dim=0)
    return TensorizedDataset(
        skill_id=int(skill_id),
        split=str(split),
        grid_size=int(max_g),
        num_demos=int(max_nd),
        src=src,
        tgt=tgt,
        grid_size_each=grid_each,
        num_demos_each=demos_each,
    )


def maybe_load_skill_split(
    *, data_dir: Path | list[Path], skill_id: int, split: str, max_seq_len: Optional[int] = None
) -> Optional[TensorizedDataset]:
    roots = [Path(p).expanduser().resolve() for p in _as_data_dirs(data_dir)]
    for root in roots:
        path = root / f"skill_{int(skill_id)}" / f"{split}.json"
        if path.exists():
            return load_skill_split(data_dir=data_dir, skill_id=skill_id, split=split, max_seq_len=max_seq_len)
    return None


def prepare_batch(
    *,
    batch_size: int,
    train_pool: TensorizedDataset,
    device: torch.device,
    cpu_generator: torch.Generator,
    augment: Optional[AugmentSpec] = None,
    grid_size: Optional[int] = None,
    num_demos: Optional[int] = None,
) -> Batch:
    """
    Prepare a training batch with minimal CPU overhead.

    - Sampling uses torch RNG (can run on GPU if train_pool is on GPU).
    - If train_pool tensors live on CPU and are pinned, H2D copies can be async via non_blocking=True.
    """
    bsz = int(batch_size)
    pool_device = train_pool.src.device
    # Important: torch.Generator(device=...) is not uniformly supported across torch versions.
    # To avoid generator/device mismatches, we only use an explicit generator on CPU.
    if pool_device.type == "cpu":
        idx = torch.randint(
            low=0,
            high=int(train_pool.n),
            size=(bsz,),
            device=pool_device,
            generator=cpu_generator,
            dtype=torch.long,
        )
    else:
        # Uses global RNG seeded via torch.manual_seed (covers CUDA too).
        idx = torch.randint(
            low=0,
            high=int(train_pool.n),
            size=(bsz,),
            device=pool_device,
            dtype=torch.long,
        )
    src = train_pool.src.index_select(0, idx)
    tgt = train_pool.tgt.index_select(0, idx)
    g_each = train_pool.grid_size_each.index_select(0, idx)
    nd_each = train_pool.num_demos_each.index_select(0, idx)
    if pool_device != device:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        g_each = g_each.to(device, non_blocking=True)
        nd_each = nd_each.to(device, non_blocking=True)
    if augment is not None and bool(augment.enabled):
        # Safe for variable-size grids because within-grid padding uses background color 0 and
        # augmentation preserves the target ignore mask (-100) via a transformed validity mask.
        src, tgt = augment_src_tgt_batch(
            src=src,
            tgt=tgt,
            grid_size=int(train_pool.grid_size),
            num_demos=int(train_pool.num_demos),
            generator=cpu_generator if device.type == "cpu" else None,
            spec=augment,
        )

    # Predict logits from the test_x segment (always max_g^2 positions); mask via tgt == -100.
    T = int(src.shape[1])
    Gmax = int(tgt.shape[1])
    pred_pos = torch.arange(Gmax, device=device, dtype=torch.long).unsqueeze(0).expand(int(bsz), int(Gmax))
    pred_mask = tgt != -100
    key_padding_mask = torch.zeros((int(bsz), int(T)), device=device, dtype=torch.bool)
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

    - dataset_device="gpu": move tensors onto `device` (fastest; uses more VRAM).
    - dataset_device="cpu": keep tensors on CPU but pin them when using CUDA (enables async H2D copies).
    """
    mode = str(dataset_device).lower()
    if mode not in {"cpu", "gpu"}:
        raise ValueError(f"dataset_device must be 'cpu' or 'gpu', got {dataset_device!r}")
    if mode == "gpu":
        if device.type != "cuda":
            return ds
        return TensorizedDataset(
            skill_id=ds.skill_id,
            split=ds.split,
            grid_size=ds.grid_size,
            num_demos=ds.num_demos,
            src=ds.src.to(device),
            tgt=ds.tgt.to(device),
            grid_size_each=ds.grid_size_each.to(device),
            num_demos_each=ds.num_demos_each.to(device),
        )
    # cpu mode
    return TensorizedDataset(
        skill_id=ds.skill_id,
        split=ds.split,
        grid_size=ds.grid_size,
        num_demos=ds.num_demos,
        src=_pin_if_cuda(ds.src, device=device),
        tgt=_pin_if_cuda(ds.tgt, device=device),
        grid_size_each=_pin_if_cuda(ds.grid_size_each, device=device),
        num_demos_each=_pin_if_cuda(ds.num_demos_each, device=device),
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
) -> float:
    model.eval()
    k = min(int(n_tasks), dataset.n)
    if k <= 0:
        return 0.0

    # Sample once on CPU (deterministic via numpy rng), then do batched eval on device.
    idx_np = rng.choice(dataset.n, size=k, replace=False)
    idx = torch.as_tensor(idx_np, dtype=torch.long, device=dataset.src.device)
    src = dataset.src.index_select(0, idx)
    tgt = dataset.tgt.index_select(0, idx)
    grid_each = dataset.grid_size_each.index_select(0, idx)
    if src.device != device:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

    bs = max(1, int(eval_batch_size))
    correct = 0
    saved_unsolved = 0
    saved_solved = 0
    saved_augmented = 0
    printed = 0
    for off in range(0, k, bs):
        xb = src[off : off + bs]
        yb = tgt[off : off + bs]  # (B, grid_tokens)
        valid = yb != -100

        if int(vote_augs) > 0:
            if vote_spec is None or not bool(vote_spec.enabled):
                raise ValueError("vote_augs > 0 requires vote_spec.enabled=True")
            bsz = int(xb.shape[0])
            g = int(dataset.grid_size)
            # Majority vote in the *original* space by inverting each augmentation on its prediction.
            # Vote only on valid cells by zeroing invalid positions before hashing.
            best_pred = torch.empty((bsz, int(grid_tokens)), dtype=torch.long, device="cpu")
            best_count = [0 for _ in range(bsz)]
            counts: list[dict[bytes, int]] = [dict() for _ in range(bsz)]

            for _ in range(int(vote_augs)):
                xb_aug, yb_aug, params = augment_src_tgt_batch_with_params(
                    src=xb,
                    tgt=yb,
                    grid_size=int(dataset.grid_size),
                    num_demos=int(dataset.num_demos),
                    generator=None,  # GPU-friendly; uses global RNG if on CUDA
                    spec=vote_spec,
                )
                logits = model(xb_aug)  # (B, T, V)
                pred_logits = logits[:, -(grid_tokens + 1) : -1, :]
                pred = torch.argmax(pred_logits, dim=-1).reshape(bsz, g, g).unsqueeze(1)  # (B,1,g,g)
                pred_inv = invert_grids_torch(pred, params=params).squeeze(1).reshape(bsz, int(grid_tokens))  # (B,G)
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
            logits = model(xb)  # (B, T, V)
            pred_logits = logits[:, -(grid_tokens + 1) : -1, :]  # (B, grid_tokens, V)
            pred_final = torch.argmax(pred_logits, dim=-1)  # (B, grid_tokens)

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
                for bi in bi_solved:
                    if printed >= int(print_solved_max):
                        break
                    g_i = int(g_each_b[int(bi)])
                    if g_i <= 0:
                        continue

                    # Decode prompt back into grids (max-sized), then crop to actual size.
                    _demos, test_x = _decode_prompt_src(
                        src_tokens=xb_cpu[int(bi)], grid_size=int(dataset.grid_size), num_demos=int(dataset.num_demos)
                    )
                    test_x = crop(test_x, g_i)
                    true_y = crop(yb_cpu[int(bi)].reshape(int(dataset.grid_size), int(dataset.grid_size)), g_i)
                    pred_y = crop(pred_cpu[int(bi)].reshape(int(dataset.grid_size), int(dataset.grid_size)), g_i)

                    ds_idx = int(idx_np[int(off + bi)])
                    print(
                        f"\n=== solved example | tag={print_solved_tag} | s{int(dataset.skill_id)} | split={dataset.split} | "
                        f"step={step_s} | idx={ds_idx} | grid={g_i}x{g_i} ===",
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
                    src_i = xb[bi].detach().cpu().numpy()
                    demos, test_x = _decode_prompt_src(
                        src_tokens=src_i, grid_size=int(dataset.grid_size), num_demos=int(dataset.num_demos)
                    )
                    g = int(dataset.grid_size)
                    true_y = yb[bi].detach().cpu().numpy().reshape(g, g)
                    pred_y = pred_final[bi].detach().cpu().numpy().reshape(g, g)

                    ds_idx = int(idx_np[int(off + bi)])
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
                    # Decode + save on CPU.
                    src_i = xb[bi].detach().cpu().numpy()
                    demos, test_x = _decode_prompt_src(
                        src_tokens=src_i, grid_size=int(dataset.grid_size), num_demos=int(dataset.num_demos)
                    )
                    g = int(dataset.grid_size)
                    true_y = yb[bi].detach().cpu().numpy().reshape(g, g)
                    pred_y = pred_final[bi].detach().cpu().numpy().reshape(g, g)

                    ds_idx = int(idx_np[int(off + bi)])
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
                # Apply the *train-time* augmentation distribution and run the model on the augmented prompt.
                xb1 = xb[bi : bi + 1]
                yb1 = yb[bi : bi + 1]
                xb_aug, yb_aug = augment_src_tgt_batch(
                    src=xb1,
                    tgt=yb1,
                    grid_size=int(dataset.grid_size),
                    num_demos=int(dataset.num_demos),
                    generator=None,  # GPU-friendly; uses global RNG if on CUDA
                    spec=save_augmented_spec,
                )
                logits_aug = model(xb_aug)
                pred_logits_aug = logits_aug[:, -(grid_tokens + 1) : -1, :]
                pred_aug = torch.argmax(pred_logits_aug, dim=-1)  # (1, grid_tokens)

                # Decode + save on CPU.
                src_i = xb_aug[0].detach().cpu().numpy()
                demos, test_x = _decode_prompt_src(
                    src_tokens=src_i, grid_size=int(dataset.grid_size), num_demos=int(dataset.num_demos)
                )
                g = int(dataset.grid_size)
                true_y = yb_aug[0].detach().cpu().numpy().reshape(g, g)
                pred_y = pred_aug[0].detach().cpu().numpy().reshape(g, g)
                ds_idx = int(idx_np[int(off + bi)])

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
    grid_size: int,
) -> None:
    grid_tokens = grid_size * grid_size
    i = 0
    src = dataset.src[i : i + 1].to(device)
    tgt = dataset.tgt[i].cpu().numpy().reshape(grid_size, grid_size)

    logits = model(src)
    pred_logits = logits[:, -(grid_tokens + 1) : -1, :]
    pred = torch.argmax(pred_logits, dim=-1).cpu().numpy().reshape(grid_size, grid_size)

    # Decode the first example back to grids for printing.
    # Layout: (x SEP y SEP) x3, then test_x SEP.
    tokens = dataset.src[i].cpu().numpy().tolist()
    g = grid_size * grid_size

    def unflatten(block: list[int]) -> np.ndarray:
        return np.asarray(block, dtype=np.int64).reshape(grid_size, grid_size)

    demos = []
    off = 0
    for _ in range(int(dataset.num_demos)):
        x = unflatten(tokens[off : off + g])
        off += g + 1  # +SEP
        y = unflatten(tokens[off : off + g])
        off += g + 1  # +SEP
        demos.append((x, y))
    test_x = unflatten(tokens[off : off + g])

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


