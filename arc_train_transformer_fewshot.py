from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from arc_dataset_models import ARCDataset
from arc_train_utils import (
    LearningCurves,
    SEP_TOKEN,
    VOCAB_SIZE,
    count_params,
    plot_learning_curves,
    progress,
    prompt_seq_len,
    write_learning_curves_csv,
)


def _parse_dataset_json(path: Path) -> ARCDataset:
    raw = path.read_text(encoding="utf-8")
    if hasattr(ARCDataset, "model_validate_json"):
        return ARCDataset.model_validate_json(raw)  # pydantic v2
    return ARCDataset.parse_raw(raw)  # pydantic v1


def _grid_to_np(g: list[list[int]]) -> np.ndarray:
    return np.asarray(g, dtype=np.int64)


def _flatten_prompt(
    *,
    demos: list[tuple[np.ndarray, np.ndarray]],
    test_in: np.ndarray,
) -> list[int]:
    seq: list[int] = []
    for x, y in demos:
        seq += x.flatten().tolist() + [int(SEP_TOKEN)] + y.flatten().tolist() + [int(SEP_TOKEN)]
    seq += test_in.flatten().tolist() + [int(SEP_TOKEN)]
    return [int(t) for t in seq]


def _sanitize_key(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


@dataclass(frozen=True)
class VariantKey:
    skill_id: int
    variant: str

    def to_str(self) -> str:
        return f"s{int(self.skill_id)}__v{_sanitize_key(self.variant)}"


@dataclass(frozen=True)
class VariantPool:
    key: VariantKey
    grid_size: int
    split: str
    src: torch.Tensor  # (N,T)
    tgt: torch.Tensor  # (N,G)

    @property
    def n(self) -> int:
        return int(self.src.shape[0])


def _split_indices(*, n: int, val_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    if n == 1:
        # Avoid leaking the only sample: keep it for val.
        return np.asarray([], dtype=np.int64), np.asarray([0], dtype=np.int64)
    frac = float(val_frac)
    if not (0.0 < frac < 1.0):
        raise ValueError(f"val_frac must be in (0,1), got {frac}")
    n_val = int(round(frac * n))
    n_val = max(1, n_val)
    n_val = min(n - 1, n_val)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _subset_pool(pool: VariantPool, idx: np.ndarray, *, split_suffix: str) -> VariantPool:
    return VariantPool(
        key=pool.key,
        grid_size=pool.grid_size,
        split=f"{pool.split}_{split_suffix}",
        src=pool.src[idx],
        tgt=pool.tgt[idx],
    )


def load_variant_pools(
    *,
    data_dir: Path,
    skill_ids: list[int],
    split: str,
    rng: np.random.Generator,
    val_frac: float,
) -> tuple[dict[VariantKey, VariantPool], dict[VariantKey, VariantPool]]:
    """
    Load <data_dir>/skill_<id>/<split>.json for each skill and group tasks by (skill_id, puzzle_variant).
    Returns (train_pools_by_key, val_pools_by_key) using a per-key deterministic split.
    """
    train_pools: dict[VariantKey, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    val_pools: dict[VariantKey, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    for sid in skill_ids:
        path = data_dir / f"skill_{int(sid)}" / f"{split}.json"
        ds = _parse_dataset_json(path)
        grid_size = int(ds.grid_size)
        grid_tokens = grid_size * grid_size
        seq_len = prompt_seq_len(grid_size=grid_size, num_demos=3)

        # First, collect rows per key in numpy/torch.
        per_key_rows: dict[VariantKey, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for task in ds.tasks:
            v = task.puzzle_variant if task.puzzle_variant is not None else "none"
            key = VariantKey(skill_id=int(sid), variant=str(v))

            demos: list[tuple[np.ndarray, np.ndarray]] = []
            for demo in task.demos:
                demos.append((_grid_to_np(demo.x), _grid_to_np(demo.y)))
            test_in = _grid_to_np(task.test.x)
            test_out = _grid_to_np(task.test.y).reshape(-1)

            seq = _flatten_prompt(demos=demos, test_in=test_in)
            if len(seq) != int(seq_len):
                raise ValueError(f"Unexpected seq_len={len(seq)} (expected {seq_len}) for skill={sid} split={split}")
            if int(test_out.shape[0]) != int(grid_tokens):
                raise ValueError("Unexpected test_out shape.")

            src = torch.tensor(seq, dtype=torch.long)
            tgt = torch.tensor(test_out.tolist(), dtype=torch.long)
            per_key_rows.setdefault(key, []).append((src, tgt))

        # Now split each key into train/val.
        for key, rows in per_key_rows.items():
            n = len(rows)
            train_idx, val_idx = _split_indices(n=n, val_frac=float(val_frac), rng=rng)
            if train_idx.size > 0:
                for i in train_idx.tolist():
                    train_pools.setdefault(key, []).append(rows[int(i)])
            if val_idx.size > 0:
                for i in val_idx.tolist():
                    val_pools.setdefault(key, []).append(rows[int(i)])

    def stack_pool(rows: list[tuple[torch.Tensor, torch.Tensor]], *, key: VariantKey, grid_size: int, split: str) -> VariantPool:
        if len(rows) == 0:
            raise ValueError("Empty pool.")
        src = torch.stack([r[0] for r in rows], dim=0)
        tgt = torch.stack([r[1] for r in rows], dim=0)
        return VariantPool(key=key, grid_size=int(grid_size), split=split, src=src, tgt=tgt)

    # Convert list rows into VariantPool tensors (grid_size is consistent within a skill; validate).
    train_out: dict[VariantKey, VariantPool] = {}
    val_out: dict[VariantKey, VariantPool] = {}
    for key, rows in train_pools.items():
        # Find grid_size from any row by looking at tgt length => g^2 (we assume square).
        g2 = int(rows[0][1].shape[0])
        g = int(round(g2 ** 0.5))
        train_out[key] = stack_pool(rows, key=key, grid_size=g, split=f"{split}_train")
    for key, rows in val_pools.items():
        g2 = int(rows[0][1].shape[0])
        g = int(round(g2 ** 0.5))
        val_out[key] = stack_pool(rows, key=key, grid_size=g, split=f"{split}_val")

    # Ensure every key has a validation pool (needed for admission).
    missing = [k for k in train_out.keys() if k not in val_out]
    if missing:
        miss_s = ", ".join(k.to_str() for k in missing[:8])
        raise ValueError(f"Some variant keys have no val pool (need >=1 example each). Examples: {miss_s}")
    return train_out, val_out


def _sample_batch(
    *,
    pool: VariantPool,
    batch_size: int,
    device: torch.device,
    cpu_generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = int(batch_size)
    if bsz <= 0:
        raise ValueError(f"batch_size must be >= 1, got {bsz}")
    if pool.n <= 0:
        raise ValueError("Cannot sample from empty pool.")
    pool_device = pool.src.device
    n = int(pool.n)
    # Prefer sampling *without replacement* when possible to maximize unique examples per update.
    # Falls back to sampling with replacement if bsz > n.
    if bsz <= n:
        if pool_device.type == "cpu":
            idx = torch.randperm(n, device=pool_device, generator=cpu_generator, dtype=torch.long)[:bsz]
        else:
            # torch.Generator(device=...) is not uniformly supported, rely on global RNG (seeded via torch.manual_seed).
            idx = torch.randperm(n, device=pool_device, dtype=torch.long)[:bsz]
    else:
        if pool_device.type == "cpu":
            idx = torch.randint(
                low=0,
                high=n,
                size=(bsz,),
                device=pool_device,
                generator=cpu_generator,
                dtype=torch.long,
            )
        else:
            idx = torch.randint(low=0, high=n, size=(bsz,), device=pool_device, dtype=torch.long)

    src = pool.src.index_select(0, idx)
    tgt = pool.tgt.index_select(0, idx)
    if src.device != device:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
    return src, tgt


class VariantExpert(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        num_layers: int = 2,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            batch_first=True,
            dropout=float(dropout),
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.fc_out = nn.Linear(int(embed_dim), int(vocab_size))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B,T,D) -> (B,T,V)
        hh = self.encoder(h)
        return self.fc_out(hh)


class WarmupTrunk(nn.Module):
    """
    Generic trunk-only model for warmup pretraining on the mixed pool (ignores variants).
    This model is thrown away after warmup; we only copy its trunk weights.
    """

    def __init__(
        self,
        *,
        grid_size: int,
        max_len: int,
        pos_encoding: str,
        embed_dim: int,
        num_heads: int,
        trunk_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.0,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.grid_tokens = int(grid_size) * int(grid_size)
        self.pos_encoding = str(pos_encoding).lower()
        if self.pos_encoding not in {"2d", "1d"}:
            raise ValueError(f"pos_encoding must be one of {{'2d','1d'}}, got {pos_encoding!r}")

        self.embed = nn.Embedding(int(vocab_size), int(embed_dim))
        self.global_pos_enc = nn.Parameter(torch.randn(1, int(max_len), int(embed_dim)) * 0.02)
        if self.pos_encoding == "2d":
            self.row_embed = nn.Embedding(int(self.grid_size), int(embed_dim))
            self.col_embed = nn.Embedding(int(self.grid_size), int(embed_dim))

        trunk_layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            batch_first=True,
            dropout=float(dropout),
        )
        self.trunk = nn.TransformerEncoder(trunk_layer, num_layers=int(trunk_layers))
        self.fc_out = nn.Linear(int(embed_dim), int(vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, t = x.shape
        if t > int(self.global_pos_enc.shape[1]):
            raise ValueError(
                f"Sequence too long: t={t} > max_len={int(self.global_pos_enc.shape[1])}. Increase max_len."
            )
        emb = self.embed(x) + self.global_pos_enc[:, :t, :]
        if self.pos_encoding == "2d":
            block = self.grid_tokens + 1
            pos = torch.arange(t, device=x.device)
            within = pos % block
            is_sep = within == self.grid_tokens
            cell = torch.clamp(within, max=self.grid_tokens - 1)
            row = (cell // self.grid_size).to(torch.long)
            col = (cell % self.grid_size).to(torch.long)
            pos_emb_2d = self.row_embed(row) + self.col_embed(col)
            pos_emb_2d = pos_emb_2d.masked_fill(is_sep.unsqueeze(-1), 0.0)
            emb = emb + pos_emb_2d.unsqueeze(0)
        h = self.trunk(emb)
        return self.fc_out(h)


def _make_mixed_pool(*, pools: dict[VariantKey, VariantPool], split: str) -> VariantPool:
    """Concatenate all examples across variant pools into a single pool for warmup."""
    non_empty = [p for p in pools.values() if int(p.n) > 0]
    if not non_empty:
        raise ValueError("No non-empty pools to mix.")
    grid_size = int(non_empty[0].grid_size)
    src = torch.cat([p.src for p in non_empty], dim=0)
    tgt = torch.cat([p.tgt for p in non_empty], dim=0)
    return VariantPool(
        key=VariantKey(skill_id=-1, variant="mixed"),
        grid_size=grid_size,
        split=str(split),
        src=src,
        tgt=tgt,
    )


class TrunkPlusExperts(nn.Module):
    """
    Shared token embedding + (optional) 2D positional encoding + 4-layer trunk Transformer,
    followed by a per-(skill,variant) 2-layer Transformer expert with its own output head.
    """

    def __init__(
        self,
        *,
        grid_size: int,
        max_len: int,
        pos_encoding: str,
        embed_dim: int,
        num_heads: int,
        trunk_layers: int = 4,
        expert_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.0,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.grid_tokens = int(grid_size) * int(grid_size)
        self.pos_encoding = str(pos_encoding).lower()
        if self.pos_encoding not in {"2d", "1d"}:
            raise ValueError(f"pos_encoding must be one of {{'2d','1d'}}, got {pos_encoding!r}")

        self.embed = nn.Embedding(int(vocab_size), int(embed_dim))
        self.global_pos_enc = nn.Parameter(torch.randn(1, int(max_len), int(embed_dim)) * 0.02)

        if self.pos_encoding == "2d":
            self.row_embed = nn.Embedding(int(self.grid_size), int(embed_dim))
            self.col_embed = nn.Embedding(int(self.grid_size), int(embed_dim))

        trunk_layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            batch_first=True,
            dropout=float(dropout),
        )
        self.trunk = nn.TransformerEncoder(trunk_layer, num_layers=int(trunk_layers))

        self._expert_layers = int(expert_layers)
        self._expert_ff_dim = int(ff_dim)
        self._expert_heads = int(num_heads)
        self._dropout = float(dropout)
        self._embed_dim = int(embed_dim)
        self._vocab_size = int(vocab_size)

        self.experts = nn.ModuleDict()

    def ensure_expert(self, key: VariantKey) -> str:
        k = key.to_str()
        if k not in self.experts:
            # IMPORTANT: experts are created dynamically; if the parent model was already moved
            # to GPU, newly-added submodules will otherwise stay on CPU and cause device mismatch.
            device = self.embed.weight.device
            self.experts[k] = VariantExpert(
                embed_dim=int(self._embed_dim),
                num_heads=int(self._expert_heads),
                ff_dim=int(self._expert_ff_dim),
                dropout=float(self._dropout),
                num_layers=int(self._expert_layers),
                vocab_size=int(self._vocab_size),
            ).to(device)
        return k

    def forward(self, x: torch.Tensor, *, expert_key: str) -> torch.Tensor:
        # x: (B,T)
        _b, t = x.shape
        if t > int(self.global_pos_enc.shape[1]):
            raise ValueError(
                f"Sequence too long: t={t} > max_len={int(self.global_pos_enc.shape[1])}. Increase max_len."
            )

        emb = self.embed(x) + self.global_pos_enc[:, :t, :]

        if self.pos_encoding == "2d":
            block = self.grid_tokens + 1
            pos = torch.arange(t, device=x.device)
            within = pos % block
            is_sep = within == self.grid_tokens
            cell = torch.clamp(within, max=self.grid_tokens - 1)
            row = (cell // self.grid_size).to(torch.long)
            col = (cell % self.grid_size).to(torch.long)
            pos_emb_2d = self.row_embed(row) + self.col_embed(col)
            pos_emb_2d = pos_emb_2d.masked_fill(is_sep.unsqueeze(-1), 0.0)
            emb = emb + pos_emb_2d.unsqueeze(0)

        h = self.trunk(emb)
        expert = self.experts[expert_key]
        if next(expert.parameters()).device != h.device:
            # Defensive: ensures correct device even if an expert was created before .to(device).
            expert = expert.to(h.device)
            self.experts[expert_key] = expert
        return expert(h)


@torch.no_grad()
def _exact_match_acc(
    *,
    model: TrunkPlusExperts,
    pool: VariantPool,
    expert_key: str,
    device: torch.device,
    eval_batch_size: int,
) -> float:
    model.eval()
    grid_tokens = int(pool.grid_size) * int(pool.grid_size)
    n = int(pool.n)
    if n <= 0:
        return 0.0
    bs = max(1, int(eval_batch_size))
    correct = 0
    for off in range(0, n, bs):
        xb = pool.src[off : off + bs].to(device, non_blocking=True)
        yb = pool.tgt[off : off + bs].to(device, non_blocking=True)
        logits = model(xb, expert_key=expert_key)  # (B,T,V)
        pred_logits = logits[:, -(grid_tokens + 1) : -1, :]
        pred = torch.argmax(pred_logits, dim=-1)
        correct += int((pred == yb).all(dim=1).sum().item())
    return float(correct) / float(n)


def main(
    *,
    data_dir: Path,
    grid_size: int,
    pos_encoding: str,
    train_skills: Optional[list[int]],
    n_warmup: int = 2000,
    n_outer: int = 2000,
    n_inner_steps: int = 100,
    inner_batch_size: int = 256,
    warmup_batch_size: int = 64,
    val_frac: float = 0.2,
    admit_threshold: float = 0.50,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    dropout: float = 0.0,
    lr_warmup: float = 5e-4,
    lr_inner: float = 5e-4,
    lr_trunk_inner: float = 5e-6,
    eval_batch_size: int = 256,
    eval_every: int = 500,
    eval_keys_per_skill: int = 8,
    outer_log_every: int = 5,
    seed: int = 0,
    device: str = "cuda",
    dataset_device: str = "gpu",
    progress_bar: bool = False,
    out_dir: Path = Path("arc_train_runs_fewshot"),
    no_plots: bool = False,
) -> None:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    device_t = torch.device(str(device))

    skills = [14, 15, 16] if train_skills is None else [int(s) for s in train_skills]
    for sid in skills:
        if sid < 1:
            raise ValueError(f"Invalid skill id: {sid}")

    # Load and group by (skill,variant); use ID split only for this fewshot admission training.
    train_pools, val_pools = load_variant_pools(
        data_dir=Path(data_dir),
        skill_ids=skills,
        split="train",
        rng=rng,
        val_frac=float(val_frac),
    )

    # Sanity: grid sizes must match the configured grid_size for this run.
    for k, p in list(train_pools.items()) + list(val_pools.items()):
        if int(p.grid_size) != int(grid_size):
            raise ValueError(f"Pool {k.to_str()} grid_size={p.grid_size} != --grid_size={grid_size}")

    seq_len = prompt_seq_len(grid_size=int(grid_size), num_demos=3)
    model = TrunkPlusExperts(
        grid_size=int(grid_size),
        max_len=int(seq_len),
        pos_encoding=str(pos_encoding),
        embed_dim=int(embed_dim),
        num_heads=int(num_heads),
        trunk_layers=4,
        expert_layers=2,
        ff_dim=int(ff_dim),
        dropout=float(dropout),
        vocab_size=int(VOCAB_SIZE),
    ).to(device_t)

    # Pre-create experts for all variant keys for deterministic parameter counts.
    keys = sorted(train_pools.keys(), key=lambda kk: kk.to_str())
    for k in keys:
        model.ensure_expert(k)

    total_params, trainable_params = count_params(model)
    print(f"Model params: total={total_params:,} trainable={trainable_params:,} | experts={len(model.experts)}")

    # Optionally move datasets to GPU for speed.
    def maybe_move_pool(pool: VariantPool) -> VariantPool:
        mode = str(dataset_device).lower()
        if mode not in {"cpu", "gpu"}:
            raise ValueError(f"dataset_device must be 'cpu' or 'gpu', got {dataset_device!r}")
        if mode == "gpu" and device_t.type == "cuda":
            return VariantPool(
                key=pool.key,
                grid_size=pool.grid_size,
                split=pool.split,
                src=pool.src.to(device_t),
                tgt=pool.tgt.to(device_t),
            )
        # cpu mode: leave on cpu (training batches will .to(device))
        return pool

    train_pools = {k: maybe_move_pool(p) for k, p in train_pools.items()}
    val_pools = {k: maybe_move_pool(p) for k, p in val_pools.items()}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_enabled = not bool(no_plots)

    # Warmup: mix *all examples* together and train a generic trunk-only model.
    # This warmup model is thrown away after warmup; we only copy its trunk weights.
    mixed_train = maybe_move_pool(_make_mixed_pool(pools=train_pools, split="train_mixed"))
    mixed_val = maybe_move_pool(_make_mixed_pool(pools=val_pools, split="val_mixed"))

    warm_model = WarmupTrunk(
        grid_size=int(grid_size),
        max_len=int(seq_len),
        pos_encoding=str(pos_encoding),
        embed_dim=int(embed_dim),
        num_heads=int(num_heads),
        trunk_layers=4,
        ff_dim=int(ff_dim),
        dropout=float(dropout),
        vocab_size=int(VOCAB_SIZE),
    ).to(device_t)

    warmup_opt = optim.AdamW(warm_model.parameters(), lr=float(lr_warmup), weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    gen_cpu = torch.Generator().manual_seed(int(seed))

    admitted: set[str] = set()
    fail_count: dict[str, int] = {}

    # Learning curves (same artifact names/format as arc_train_transformer.py).
    curves = LearningCurves(
        steps=[],
        loss=[],
        acc_train={},
        acc_id={},
        acc_ood={},
        probe_train_ood=[],
        probe_fully_heldout_ood=[],
    )

    keys_by_skill: dict[int, list[VariantKey]] = {}
    for kk in keys:
        keys_by_skill.setdefault(int(kk.skill_id), []).append(kk)
    for sid in sorted(keys_by_skill.keys()):
        curves.ensure_skill(int(sid))

    def _eval_skill_acc(*, which: str) -> dict[int, float]:
        if which not in {"train", "val"}:
            raise ValueError(f"which must be 'train' or 'val', got {which!r}")
        pools = train_pools if which == "train" else val_pools
        out: dict[int, float] = {}
        for sid, kk_list in sorted(keys_by_skill.items()):
            if len(kk_list) == 0:
                out[int(sid)] = 0.0
                continue
            m = min(int(eval_keys_per_skill), len(kk_list))
            pick = rng.choice(len(kk_list), size=m, replace=False).tolist()
            accs: list[float] = []
            for j in pick:
                kk2 = kk_list[int(j)]
                ek2 = model.ensure_expert(kk2)
                accs.append(
                    _exact_match_acc(
                        model=model,
                        pool=pools[kk2],
                        expert_key=ek2,
                        device=device_t,
                        eval_batch_size=int(eval_batch_size),
                    )
                )
            out[int(sid)] = float(np.mean(accs)) if len(accs) > 0 else 0.0
        return out

    def _maybe_log_eval(*, step: int, last_loss: float) -> None:
        if int(eval_every) <= 0:
            return
        if int(step) % int(eval_every) != 0:
            return

        model.eval()
        acc_train = _eval_skill_acc(which="train")
        acc_val = _eval_skill_acc(which="val")

        curves.steps.append(int(step))
        curves.loss.append(float(last_loss))
        curves.probe_train_ood.append(0.0)
        curves.probe_fully_heldout_ood.append(0.0)
        for sid in sorted(keys_by_skill.keys()):
            curves.acc_train[int(sid)].append(float(acc_train.get(int(sid), 0.0)))
            curves.acc_id[int(sid)].append(float(acc_val.get(int(sid), 0.0)))
            # No separate OOD split in this fewshot script; mirror val for plotting compatibility.
            curves.acc_ood[int(sid)].append(float(acc_val.get(int(sid), 0.0)))

        metrics_csv = out_dir / "plots" / "learning_curves_latest.csv"
        write_learning_curves_csv(curves=curves, skills=sorted(keys_by_skill.keys()), out_path=metrics_csv)

        if plots_enabled:
            title = (
                "ARC fewshot (trunk+experts) learning curves (exact-match acc)\n"
                f"skills={sorted(keys_by_skill.keys())} | eval_keys_per_skill={int(eval_keys_per_skill)} | "
                f"admit_threshold={float(admit_threshold):.2f} | admitted={len(admitted)} | "
                f"resets={sum(int(v) for v in fail_count.values())} | "
                f"lr_warmup={float(lr_warmup):.2e} lr_inner={float(lr_inner):.2e} lr_trunk_inner={float(lr_trunk_inner):.2e}"
            )
            latest = out_dir / "plots" / "learning_curves_latest.png"
            plot_learning_curves(curves=curves, skills=sorted(keys_by_skill.keys()), out_path=latest, title=title)

        def fmt(d: dict[int, float]) -> str:
            return " ".join(f"s{sid}={d[int(sid)]:.3f}" for sid in sorted(keys_by_skill.keys()))

        print(
            f"eval step={int(step):6d} loss={float(last_loss):.4f} | "
            f"train: {fmt(acc_train)} | val: {fmt(acc_val)}"
        )
        model.train()

    warm_model.train()
    warm_iter = progress(range(int(n_warmup)), total=int(n_warmup), desc="warmup", enabled=bool(progress_bar))
    for step in warm_iter:
        xb, yb = _sample_batch(
            pool=mixed_train,
            batch_size=int(warmup_batch_size),
            device=device_t,
            cpu_generator=gen_cpu,
        )
        warmup_opt.zero_grad(set_to_none=True)
        logits = warm_model(xb)  # (B,T,V)
        grid_tokens = int(grid_size) * int(grid_size)
        pred_logits = logits[:, -(grid_tokens + 1) : -1, :]
        loss = loss_fn(pred_logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
        loss.backward()
        warmup_opt.step()

        if (int(step) % 250 == 0) or (int(step) == int(n_warmup) - 1):
            warm_model.eval()
            # Exact-match on the mixed val pool.
            n = int(mixed_val.n)
            bs = max(1, int(eval_batch_size))
            correct = 0
            for off in range(0, n, bs):
                xb2 = mixed_val.src[off : off + bs].to(device_t, non_blocking=True)
                yb2 = mixed_val.tgt[off : off + bs].to(device_t, non_blocking=True)
                logits2 = warm_model(xb2)
                pred_logits2 = logits2[:, -(grid_tokens + 1) : -1, :]
                pred2 = torch.argmax(pred_logits2, dim=-1)
                correct += int((pred2 == yb2).all(dim=1).sum().item())
            acc = float(correct) / float(max(1, n))
            print(f"warmup step={int(step):5d} | mixed_val_acc={acc:.3f}")
            warm_model.train()

    # Re-initialize the fewshot model and copy in the warmup trunk weights.
    model = TrunkPlusExperts(
        grid_size=int(grid_size),
        max_len=int(seq_len),
        pos_encoding=str(pos_encoding),
        embed_dim=int(embed_dim),
        num_heads=int(num_heads),
        trunk_layers=4,
        expert_layers=2,
        ff_dim=int(ff_dim),
        dropout=float(dropout),
        vocab_size=int(VOCAB_SIZE),
    ).to(device_t)
    with torch.no_grad():
        model.embed.weight.copy_(warm_model.embed.weight)
        model.global_pos_enc.copy_(warm_model.global_pos_enc)
        if model.pos_encoding == "2d":
            model.row_embed.weight.copy_(warm_model.row_embed.weight)  # type: ignore[attr-defined]
            model.col_embed.weight.copy_(warm_model.col_embed.weight)  # type: ignore[attr-defined]
        model.trunk.load_state_dict(warm_model.trunk.state_dict())

    for k in keys:
        model.ensure_expert(k)
    total_params, trainable_params = count_params(model)
    print(f"Model params: total={total_params:,} trainable={trainable_params:,} | experts={len(model.experts)}")

    # Reset admission bookkeeping and training step counter for the fewshot phase.
    admitted.clear()
    fail_count.clear()
    global_step = 0
    last_loss = 0.0

    # After warmup: keep the trunk learning slowly while experts learn fast (fast/slow weights).
    # This avoids the "bag of heuristics" failure mode where the trunk never gets a chance to
    # acquire new concepts that appear only after warmup (e.g., diagonal symmetry).

    outer_iter = progress(range(int(n_outer)), total=int(n_outer), desc="outer", enabled=bool(progress_bar))
    for outer_step in outer_iter:
        # Sample a candidate (skill,variant) pair.
        # Prefer unadmitted keys while there are any; otherwise sample from all keys.
        unadmitted = [k for k in keys if model.ensure_expert(k) not in admitted]
        cand = unadmitted if len(unadmitted) > 0 else keys
        k = cand[int(rng.integers(0, len(cand)))]
        ek = model.ensure_expert(k)

        # Snapshot expert params so we can revert if not admitted.
        expert: VariantExpert = model.experts[ek]  # type: ignore[assignment]
        before = {name: p.detach().clone() for name, p in expert.named_parameters()}

        # Inner-loop optimizer: fast expert + slow trunk (and embeddings/pos).
        trunk_params: list[torch.nn.Parameter] = []
        trunk_params += list(model.embed.parameters())
        trunk_params += list(model.trunk.parameters())
        if model.pos_encoding == "2d":
            trunk_params += list(model.row_embed.parameters())  # type: ignore[attr-defined]
            trunk_params += list(model.col_embed.parameters())  # type: ignore[attr-defined]
        # global_pos_enc is a Parameter, not an nn.Module.
        trunk_params += [model.global_pos_enc]

        inner_opt = optim.AdamW(
            [
                {"params": list(expert.parameters()), "lr": float(lr_inner), "weight_decay": 0.0},
                {"params": trunk_params, "lr": float(lr_trunk_inner), "weight_decay": 0.01},
            ]
        )

        model.train()
        for _ in range(int(n_inner_steps)):
            xb, yb = _sample_batch(
                pool=train_pools[k],
                batch_size=int(inner_batch_size),
                device=device_t,
                cpu_generator=gen_cpu,
            )
            inner_opt.zero_grad(set_to_none=True)
            logits = model(xb, expert_key=ek)
            grid_tokens = int(grid_size) * int(grid_size)
            pred_logits = logits[:, -(grid_tokens + 1) : -1, :]
            loss = loss_fn(pred_logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1))
            last_loss = float(loss.item())
            loss.backward()
            inner_opt.step()
            _maybe_log_eval(step=int(global_step), last_loss=float(last_loss))
            global_step += 1

        # Admission check on validation.
        model.eval()
        acc = _exact_match_acc(
            model=model,
            pool=val_pools[k],
            expert_key=ek,
            device=device_t,
            eval_batch_size=int(eval_batch_size),
        )
        ok = float(acc) >= float(admit_threshold)
        if ok:
            admitted.add(ek)
        else:
            # Reset expert weights (do not admit yet). Importantly, we do NOT revert the trunk:
            # even "failed" variants can still provide useful trunk learning signal.
            with torch.no_grad():
                for name, p in expert.named_parameters():
                    p.copy_(before[name])
            fail_count[k.to_str()] = int(fail_count.get(k.to_str(), 0)) + 1

        log_every = int(outer_log_every)
        if log_every <= 0:
            raise ValueError(f"outer_log_every must be >= 1, got {log_every}")
        if (int(outer_step) % log_every == 0) or (int(outer_step) == int(n_outer) - 1):
            print(
                f"outer step={int(outer_step):5d} | key={k.to_str():>20s} | val_acc={acc:.3f} | "
                f"{'ADMIT' if ok else 'RESET'} | admitted={len(admitted)} resets={sum(int(v) for v in fail_count.values())}"
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fewshot ARC training with per-(skill,variant) experts + admission.")
    p.add_argument("--data_dir", type=Path, default=Path("tmp"))
    p.add_argument("--grid_size", type=int, default=6)
    p.add_argument("--pos_encoding", type=str, default="2d", choices=["2d", "1d"])
    p.add_argument("--train_skills", type=int, nargs="*", default=None)

    p.add_argument("--n_warmup", type=int, default=2000)
    p.add_argument("--n_outer", type=int, default=2000)
    p.add_argument("--n_inner_steps", type=int, default=100)
    p.add_argument("--inner_batch_size", type=int, default=256)
    p.add_argument("--warmup_batch_size", type=int, default=64)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--admit_threshold", type=float, default=0.5)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr_warmup", type=float, default=5e-4)
    p.add_argument("--lr_inner", type=float, default=5e-4)
    p.add_argument(
        "--lr_trunk_inner",
        type=float,
        default=5e-6,
        help="Slow learning rate for trunk/embeddings during inner-loop adaptation (fast/slow weights).",
    )
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=500, help="Global-step interval for saving plots/CSV + printing.")
    p.add_argument(
        "--outer_log_every",
        type=int,
        default=5,
        help="How often to print outer-loop admission/reset status (in outer steps). Use 1 for every step.",
    )
    p.add_argument(
        "--eval_keys_per_skill",
        type=int,
        default=8,
        help="How many (skill,variant) keys to sample per skill during evaluation (bounds cost).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset_device", type=str, default="gpu", choices=["cpu", "gpu"])
    p.add_argument("--progress", action="store_true")
    p.add_argument("--out_dir", type=Path, default=Path("arc_train_runs_fewshot"), help="Where to write plots/metrics")
    p.add_argument("--no_plots", action="store_true", help="Disable saving learning-curve PNGs (CSV still written)")
    return p


def cli_main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    main(
        data_dir=Path(args.data_dir),
        grid_size=int(args.grid_size),
        pos_encoding=str(args.pos_encoding),
        train_skills=[int(s) for s in args.train_skills] if args.train_skills is not None else None,
        n_warmup=int(args.n_warmup),
        n_outer=int(args.n_outer),
        n_inner_steps=int(args.n_inner_steps),
        inner_batch_size=int(args.inner_batch_size),
        warmup_batch_size=int(args.warmup_batch_size),
        val_frac=float(args.val_frac),
        admit_threshold=float(args.admit_threshold),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        lr_warmup=float(args.lr_warmup),
        lr_inner=float(args.lr_inner),
        lr_trunk_inner=float(args.lr_trunk_inner),
        eval_batch_size=int(args.eval_batch_size),
        eval_every=int(args.eval_every),
        eval_keys_per_skill=int(args.eval_keys_per_skill),
        outer_log_every=int(args.outer_log_every),
        seed=int(args.seed),
        device=str(args.device),
        dataset_device=str(args.dataset_device),
        progress_bar=bool(args.progress),
        out_dir=Path(args.out_dir),
        no_plots=bool(args.no_plots),
    )


if __name__ == "__main__":
    cli_main()


