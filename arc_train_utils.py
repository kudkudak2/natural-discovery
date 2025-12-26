from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from arc_dataset_models import ARCDataset


SEP_TOKEN = 5  # colors are 0..4; SEP=5
VOCAB_SIZE = 6


def prompt_seq_len(*, grid_size: int, num_demos: int = 3) -> int:
    """
    Prompt layout (fixed):
      (x SEP y SEP) repeated `num_demos` times, then (test_x SEP)
    where x/y/test_x are grid_size*grid_size tokens.
    """
    g = int(grid_size) * int(grid_size)
    return int(num_demos) * (g + 1 + g + 1) + (g + 1)


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
    chars = {0: ".", 1: "B", 2: "R", 3: "G", 4: "Y"}
    return "\n".join(" ".join(chars.get(int(c), "?") for c in row) for row in grid)


@dataclass(frozen=True)
class Batch:
    src: torch.Tensor  # (B, seq_len)
    tgt: torch.Tensor  # (B, grid_tokens)


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


@dataclass(frozen=True)
class TensorizedDataset:
    skill_id: int
    split: str
    grid_size: int
    src: torch.Tensor  # (N, T)
    tgt: torch.Tensor  # (N, grid_tokens)

    @property
    def n(self) -> int:
        return int(self.src.shape[0])


def _subset_dataset(ds: TensorizedDataset, idx: np.ndarray, *, split_suffix: str) -> TensorizedDataset:
    return TensorizedDataset(
        skill_id=ds.skill_id,
        split=f"{ds.split}_{split_suffix}",
        grid_size=ds.grid_size,
        src=ds.src[idx],
        tgt=ds.tgt[idx],
    )


def split_dataset(
    ds: TensorizedDataset, *, train_frac: float, rng: np.random.Generator
) -> tuple[TensorizedDataset, TensorizedDataset]:
    """
    Deterministically split a dataset into (train_part, eval_part).
    Ensures both are non-empty when ds.n >= 2.
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
    perm = rng.permutation(ds.n)
    train_idx = perm[:n_train]
    eval_idx = perm[n_train:]
    return _subset_dataset(ds, train_idx, split_suffix=f"train{n_train}"), _subset_dataset(ds, eval_idx, split_suffix="heldout")


def concat_datasets(datasets: list[TensorizedDataset], *, skill_id: int, split: str, grid_size: int) -> TensorizedDataset:
    non_empty = [ds for ds in datasets if ds.n > 0]
    if len(non_empty) == 0:
        raise ValueError("No datasets to concatenate (all empty).")
    src = torch.cat([ds.src for ds in non_empty], dim=0)
    tgt = torch.cat([ds.tgt for ds in non_empty], dim=0)
    return TensorizedDataset(skill_id=skill_id, split=split, grid_size=grid_size, src=src, tgt=tgt)


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
        src=ds.src[idx],
        tgt=ds.tgt[idx],
    )


def _tensorize_dataset(ds: ARCDataset) -> TensorizedDataset:
    grid_size = int(ds.grid_size)
    grid_tokens = grid_size * grid_size

    src_rows: list[torch.Tensor] = []
    tgt_rows: list[torch.Tensor] = []

    for task in ds.tasks:
        demos = []
        for demo in task.demos:
            x = np.asarray(demo.x, dtype=np.int64)
            y = np.asarray(demo.y, dtype=np.int64)
            demos.append((x, y))
        test_in = np.asarray(task.test.x, dtype=np.int64)
        test_out = np.asarray(task.test.y, dtype=np.int64).reshape(-1)

        seq = _flatten_prompt(demos, test_in)
        if len(seq) != (3 * (grid_tokens + 1 + grid_tokens + 1) + (grid_tokens + 1)):
            raise ValueError(f"Unexpected seq_len={len(seq)} for grid_size={grid_size}")

        src_rows.append(torch.tensor(seq, dtype=torch.long))
        tgt_rows.append(torch.tensor(test_out.tolist(), dtype=torch.long))

    if len(src_rows) == 0:
        raise ValueError("Dataset has no tasks.")

    return TensorizedDataset(
        skill_id=int(ds.skills[0]) if ds.skills else -1,
        split=str(ds.split),
        grid_size=grid_size,
        src=torch.stack(src_rows, dim=0),
        tgt=torch.stack(tgt_rows, dim=0),
    )


def load_skill_split(*, data_dir: Path, skill_id: int, split: str) -> TensorizedDataset:
    path = data_dir / f"skill_{int(skill_id)}" / f"{split}.json"
    ds = _parse_dataset_json(path)
    tds = _tensorize_dataset(ds)
    if int(tds.grid_size) <= 0:
        raise ValueError("Invalid grid_size in dataset.")
    return tds


def maybe_load_skill_split(*, data_dir: Path, skill_id: int, split: str) -> Optional[TensorizedDataset]:
    path = data_dir / f"skill_{int(skill_id)}" / f"{split}.json"
    if not path.exists():
        return None
    return load_skill_split(data_dir=data_dir, skill_id=skill_id, split=split)


def prepare_batch(
    *,
    batch_size: int,
    train_pool: TensorizedDataset,
    device: torch.device,
    cpu_generator: torch.Generator,
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
    if pool_device != device:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
    return Batch(src=src, tgt=tgt)


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
            src=ds.src.to(device),
            tgt=ds.tgt.to(device),
        )
    # cpu mode
    return TensorizedDataset(
        skill_id=ds.skill_id,
        split=ds.split,
        grid_size=ds.grid_size,
        src=_pin_if_cuda(ds.src, device=device),
        tgt=_pin_if_cuda(ds.tgt, device=device),
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
    if not _has_matplotlib():
        raise RuntimeError("Missing dependency: matplotlib. Install with `pip install matplotlib` or pass --no_plots.")

    import matplotlib.pyplot as plt  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
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
    if src.device != device:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

    bs = max(1, int(eval_batch_size))
    correct = 0
    for off in range(0, k, bs):
        xb = src[off : off + bs]
        yb = tgt[off : off + bs]  # (B, grid_tokens)
        logits = model(xb)  # (B, T, V)
        pred_logits = logits[:, -(grid_tokens + 1) : -1, :]  # (B, grid_tokens, V)
        pred = torch.argmax(pred_logits, dim=-1)  # (B, grid_tokens)
        correct += int((pred == yb).all(dim=1).sum().item())

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
    for _ in range(3):
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


