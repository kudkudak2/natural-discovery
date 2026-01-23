from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from arc_dataset_models import (
    ARCDataset,
    ARCExamplePair,
    ARCTask,
    ARCTestCase,
)
from arc_puzzles import ShrinkPerturbSpec, build_puzzle


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:24]


def _grid_to_list(grid: np.ndarray) -> list[list[int]]:
    return [[int(x) for x in row] for row in grid.tolist()]


def _task_from_seed(
    *,
    skill_id: int,
    grid_size: int,
    dataset_id: str,
    split: str,
    ood: bool,
    task_index: int,
    task_seed: int,
    shrink_perturb: Optional[ShrinkPerturbSpec],
) -> ARCTask:
    rng = np.random.default_rng(seed=int(task_seed))
    puzzle = build_puzzle(int(skill_id), size=int(grid_size), rng=rng, shrink_perturb=shrink_perturb)
    demos, (t_in, t_out), rule_color = puzzle.generate_prompt(num_demos=3, ood_test=bool(ood))

    task_id = _stable_id(dataset_id, str(int(task_index)))
    return ARCTask(
        task_id=task_id,
        skill_id=int(skill_id),
        skill_name=puzzle.name,
        puzzle_variant=puzzle.variant_id,
        puzzle_params=puzzle.variant_params(),
        grid_size=int(grid_size),
        rule_color=rule_color,
        demos=[ARCExamplePair(x=_grid_to_list(x), y=_grid_to_list(y)) for (x, y) in demos],
        test=ARCTestCase(x=_grid_to_list(t_in), y=_grid_to_list(t_out)),
    )


def _task_from_seed_packed(args: tuple[int, int, str, str, bool, int, int, Optional[ShrinkPerturbSpec]]) -> ARCTask:
    skill_id, grid_size, dataset_id, split, ood, task_index, task_seed, shrink_perturb = args
    return _task_from_seed(
        skill_id=skill_id,
        grid_size=grid_size,
        dataset_id=dataset_id,
        split=split,
        ood=ood,
        task_index=task_index,
        task_seed=task_seed,
        shrink_perturb=shrink_perturb,
    )


def _has_tqdm() -> bool:
    return importlib.util.find_spec("tqdm") is not None


def _progress(iterable, *, total: int, desc: str):
    """
    Returns an iterable that shows progress if tqdm is installed,
    otherwise returns the original iterable and prints coarse progress.
    """
    if _has_tqdm():
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc)
    return iterable


def generate_dataset(
    *,
    skill_id: int,
    n_tasks: int,
    grid_size: int,
    split: str,
    ood: bool,
    seed: int,
    shrink_perturb: Optional[ShrinkPerturbSpec] = None,
    n_jobs: int = 1,
    show_progress: bool = True,
) -> ARCDataset:
    rng = np.random.default_rng(seed=seed)

    dataset_id = _stable_id("arc_synth", f"skill={skill_id}", f"split={split}", f"ood={ood}", f"seed={seed}")
    tasks: list[ARCTask] = []

    n_jobs = int(n_jobs)
    if n_jobs <= 1:
        task_iter = range(int(n_tasks))
        if show_progress:
            task_iter = _progress(task_iter, total=int(n_tasks), desc=f"skill {skill_id} {split}: tasks")

        for i in task_iter:
            # Important: instantiate a fresh puzzle per task so any per-task latent parameters
            # (e.g., explosion kernel variants) are resampled across tasks while remaining
            # reproducible under the single top-level RNG seed.
            puzzle = build_puzzle(skill_id, size=grid_size, rng=rng, shrink_perturb=shrink_perturb)
            demos, (t_in, t_out), rule_color = puzzle.generate_prompt(num_demos=3, ood_test=ood)

            task_id = _stable_id(dataset_id, str(i))
            tasks.append(
                ARCTask(
                    task_id=task_id,
                    skill_id=int(skill_id),
                    skill_name=puzzle.name,
                    puzzle_variant=puzzle.variant_id,
                    puzzle_params=puzzle.variant_params(),
                    grid_size=int(grid_size),
                    rule_color=rule_color,
                    demos=[ARCExamplePair(x=_grid_to_list(x), y=_grid_to_list(y)) for (x, y) in demos],
                    test=ARCTestCase(x=_grid_to_list(t_in), y=_grid_to_list(t_out)),
                )
            )
    else:
        # Pre-sample per-task seeds deterministically from the base RNG.
        # Note: this makes multi-process generation deterministic, but results will differ
        # from the n_jobs=1 path because the RNG consumption pattern is different.
        task_seeds = rng.integers(0, 2**32 - 1, size=int(n_tasks), dtype=np.uint32).tolist()
        items = (
            (
                int(skill_id),
                int(grid_size),
                str(dataset_id),
                str(split),
                bool(ood),
                int(i),
                int(task_seed),
                shrink_perturb,
            )
            for i, task_seed in enumerate(task_seeds)
        )
        chunksize = max(1, int(n_tasks) // (n_jobs * 4))
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            it = ex.map(_task_from_seed_packed, items, chunksize=chunksize)
            if show_progress:
                it = _progress(it, total=int(n_tasks), desc=f"skill {skill_id} {split}: tasks")
            tasks = list(it)

    return ARCDataset(
        dataset_id=dataset_id,
        created_at=ARCDataset.now_iso(),
        split=split,
        ood=ood,
        skills=[int(skill_id)],
        grid_size=int(grid_size),
        tasks=tasks,
        extra={
            "generator": "my_research.natural_discovery.arc_dataset_generator",
            "note": "Each task contains 3 demos (x,y) and one test (x) with ground-truth y included in JSON.",
        },
    )


def _dump_json(path: Path, dataset: ARCDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(dataset, "model_dump_json"):
        path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
        return
    # Pydantic v1 fallback
    path.write_text(dataset.json(indent=2), encoding="utf-8")


def _ensure_render_deps() -> None:
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("Missing dependency: matplotlib. Install with `pip install matplotlib`.")


def save_task_png(path: Path, task: ARCTask) -> None:
    """
    Saves a single figure showing:
    - 3 demos: (x -> y)
    - 1 test: x (missing y visually)
    """
    _ensure_render_deps()
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402

    def to_np(g: list[list[int]]) -> np.ndarray:
        return np.asarray(g, dtype=np.int64)

    # Fixed palette: 0=black, 1=blue, 2=red, 3=green, 4=yellow
    cmap = ListedColormap(["#000000", "#1f77b4", "#d62728", "#2ca02c", "#ffdd57"])
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=cmap.N)

    # 7 panels -> 2x4 grid with one empty
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    ax_list = [ax for row in axes for ax in row]

    panels: list[tuple[str, np.ndarray]] = []
    for di, demo in enumerate(task.demos, start=1):
        panels.append((f"Demo {di}: x", to_np(demo.x)))
        panels.append((f"Demo {di}: y", to_np(demo.y)))
    panels.append(("Test: x", to_np(task.test.x)))

    for idx, ax in enumerate(ax_list):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        if idx < len(panels):
            title, grid = panels[idx]
            ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_title(title, fontsize=10)
        else:
            ax.axis("off")

    fig.suptitle(
        f"Skill {task.skill_id} ({task.skill_name})"
        + (f" | variant={task.puzzle_variant}" if task.puzzle_variant else "")
        + (f" | rule_color={task.rule_color}" if task.rule_color is not None else ""),
        fontsize=12,
    )
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _iter_skills(skills: Optional[Iterable[int]]) -> list[int]:
    if skills is None:
        return [1, 2, 3]
    return [int(s) for s in skills]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic ARC datasets (JSON) and sample PNGs.")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument("--n_tasks", type=int, default=400, help="Number of tasks per skill per split")
    p.add_argument("--grid_size", type=int, default=6, help="Grid size (NxN)")
    p.add_argument("--skills", type=int, nargs="*", default=range(11, 24), help="Skills to generate (e.g. 1 2 3)")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    p.add_argument("--png_per_skill", type=int, default=8, help="How many tasks to render as PNG per skill per split")
    p.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of worker processes for dataset generation. Use 1 to disable multiprocessing.",
    )
    p.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress output (tqdm if installed, otherwise periodic prints are minimal anyway).",
    )
    p.add_argument(
        "--no_zip",
        action="store_true",
        help="Disable creating a zip archive of the generated dataset folder.",
    )

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    skills = _iter_skills(args.skills)
    grid_size = int(args.grid_size)
    show_progress = not bool(args.no_progress)

    shrink_perturb: Optional[ShrinkPerturbSpec] = None

    for skill_id in skills:
        # Two splits: train-style (easy test) and ood test
        for split, ood in (("train", False), ("ood", True)):
            ds = generate_dataset(
                skill_id=skill_id,
                n_tasks=int(args.n_tasks),
                grid_size=grid_size,
                split=split,
                ood=ood,
                seed=int(args.seed) + 10_000 * skill_id + (1 if ood else 0),
                shrink_perturb=shrink_perturb,
                n_jobs=int(args.n_jobs),
                show_progress=show_progress,
            )

            json_path = out_dir / f"skill_{skill_id}" / f"{split}.json"
            _dump_json(json_path, ds)

            k = min(int(args.png_per_skill), len(ds.tasks))
            png_iter = range(k)
            if show_progress:
                png_iter = _progress(png_iter, total=k, desc=f"skill {skill_id} {split}: png")

            for i in png_iter:
                png_path = out_dir / f"skill_{skill_id}" / "png" / split / f"{ds.tasks[i].task_id}.png"
                save_task_png(png_path, ds.tasks[i])

            meta_path = out_dir / f"skill_{skill_id}" / f"{split}.meta.json"
            meta = {
                "skill_id": skill_id,
                "split": split,
                "ood": ood,
                "grid_size": grid_size,
                "dataset_json": str(json_path),
                "png_dir": str(out_dir / f"skill_{skill_id}" / "png" / split),
                "n_tasks": len(ds.tasks),
                "n_png": k,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not bool(args.no_zip):
        # Creates: <out_dir>.zip alongside the folder.
        base_name = str(out_dir)
        root_dir = str(out_dir.parent)
        base_dir = out_dir.name
        shutil.make_archive(base_name=base_name, format="zip", root_dir=root_dir, base_dir=base_dir)


if __name__ == "__main__":
    main()


