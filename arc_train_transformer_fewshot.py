from __future__ import annotations

import argparse
import json
import sys
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
    decode_prompt_src,
    plot_learning_curves,
    progress,
    prompt_seq_len,
    save_arc_prompt_prediction_png,
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


class Adapter(nn.Module):
    """Per-layer residual adapter (bottleneck MLP), typically trained per-variant."""

    def __init__(self, *, embed_dim: int, adapter_dim: int, scale: float) -> None:
        super().__init__()
        d = int(embed_dim)
        r = int(adapter_dim)
        if r <= 0:
            raise ValueError(f"adapter_dim must be >= 1, got {r}")
        self.down = nn.Linear(d, r, bias=False)
        self.up = nn.Linear(r, d, bias=False)
        self.scale = float(scale)

        # Small init so adapters start near-no-op.
        with torch.no_grad():
            self.down.weight.mul_(0.02)
            self.up.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + float(self.scale) * self.up(torch.nn.functional.gelu(self.down(x)))


class SkillExpert(nn.Module):
    """
    Shared per-skill expert stack. Optionally applies per-layer adapters (e.g., per variant).
    """

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
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=int(embed_dim),
                    nhead=int(num_heads),
                    dim_feedforward=int(ff_dim),
                    batch_first=True,
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.fc_out = nn.Linear(int(embed_dim), int(vocab_size))

    def forward(self, h: torch.Tensor, *, adapters: Optional[nn.ModuleList] = None) -> torch.Tensor:
        x = h
        if adapters is not None and len(adapters) != len(self.layers):
            raise ValueError(f"Adapter count {len(adapters)} != num_layers {len(self.layers)}")
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if adapters is not None:
                x = adapters[int(i)](x)
        return self.fc_out(x)


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
    followed by either:
      - per-(skill,variant) Transformer expert (legacy), OR
      - per-skill Transformer expert + per-(skill,variant) per-layer adapters (default).
    """

    def __init__(
        self,
        *,
        grid_size: int,
        max_len: int,
        pos_encoding: str,
        embed_dim: int,
        num_heads_trunk: int,
        num_heads_expert: int,
        trunk_layers: int = 4,
        expert_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.0,
        vocab_size: int = VOCAB_SIZE,
        expert_mode: str = "skill_adapter",
        adapter_dim: int = 32,
        adapter_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.grid_tokens = int(grid_size) * int(grid_size)
        self.pos_encoding = str(pos_encoding).lower()
        if self.pos_encoding not in {"2d", "1d"}:
            raise ValueError(f"pos_encoding must be one of {{'2d','1d'}}, got {pos_encoding!r}")

        trunk_heads = int(num_heads_trunk)
        expert_heads = int(num_heads_expert)
        if trunk_heads < 1:
            raise ValueError(f"num_heads_trunk must be >= 1, got {trunk_heads}")
        if expert_heads < 1:
            raise ValueError(f"num_heads_expert must be >= 1, got {expert_heads}")
        if int(embed_dim) % trunk_heads != 0:
            raise ValueError(f"embed_dim ({int(embed_dim)}) must be divisible by num_heads_trunk ({trunk_heads})")
        if int(embed_dim) % expert_heads != 0:
            raise ValueError(f"embed_dim ({int(embed_dim)}) must be divisible by num_heads_expert ({expert_heads})")

        self.embed = nn.Embedding(int(vocab_size), int(embed_dim))
        self.global_pos_enc = nn.Parameter(torch.randn(1, int(max_len), int(embed_dim)) * 0.02)

        if self.pos_encoding == "2d":
            self.row_embed = nn.Embedding(int(self.grid_size), int(embed_dim))
            self.col_embed = nn.Embedding(int(self.grid_size), int(embed_dim))

        trunk_layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(trunk_heads),
            dim_feedforward=int(ff_dim),
            batch_first=True,
            dropout=float(dropout),
        )
        self.trunk = nn.TransformerEncoder(trunk_layer, num_layers=int(trunk_layers))

        self._expert_layers = int(expert_layers)
        self._expert_ff_dim = int(ff_dim)
        self._expert_heads = int(expert_heads)
        self._dropout = float(dropout)
        self._embed_dim = int(embed_dim)
        self._vocab_size = int(vocab_size)

        self.expert_mode = str(expert_mode).lower()
        if self.expert_mode not in {"variant", "skill_adapter"}:
            raise ValueError(f"expert_mode must be one of {{'variant','skill_adapter'}}, got {expert_mode!r}")
        self._adapter_dim = int(adapter_dim)
        self._adapter_scale = float(adapter_scale)

        # Legacy: per-(skill,variant) experts.
        self.experts = nn.ModuleDict()
        # New default: shared per-skill experts + per-(skill,variant) adapters.
        self.skill_experts = nn.ModuleDict()
        self.adapters = nn.ModuleDict()

    @staticmethod
    def _skill_key_from_expert_key(expert_key: str) -> str:
        # expert_key comes from VariantKey.to_str(): "s{skill_id}__v{variant}"
        s = str(expert_key)
        if not s.startswith("s"):
            raise ValueError(f"Bad expert_key (expected 's<id>__v...'): {expert_key!r}")
        cut = s.split("__v", 1)[0]
        sid = int(cut[1:])
        return f"s{int(sid)}"

    def ensure_expert(self, key: VariantKey) -> str:
        k = key.to_str()
        # IMPORTANT: submodules are created dynamically; if the parent model was already moved
        # to GPU, newly-added submodules will otherwise stay on CPU and cause device mismatch.
        device = self.embed.weight.device
        if self.expert_mode == "variant":
            if k not in self.experts:
                self.experts[k] = VariantExpert(
                    embed_dim=int(self._embed_dim),
                    num_heads=int(self._expert_heads),
                    ff_dim=int(self._expert_ff_dim),
                    dropout=float(self._dropout),
                    num_layers=int(self._expert_layers),
                    vocab_size=int(self._vocab_size),
                ).to(device)
        else:
            sk = f"s{int(key.skill_id)}"
            if sk not in self.skill_experts:
                self.skill_experts[sk] = SkillExpert(
                    embed_dim=int(self._embed_dim),
                    num_heads=int(self._expert_heads),
                    ff_dim=int(self._expert_ff_dim),
                    dropout=float(self._dropout),
                    num_layers=int(self._expert_layers),
                    vocab_size=int(self._vocab_size),
                ).to(device)
            if k not in self.adapters:
                self.adapters[k] = nn.ModuleList(
                    [
                        Adapter(
                            embed_dim=int(self._embed_dim),
                            adapter_dim=int(self._adapter_dim),
                            scale=float(self._adapter_scale),
                        )
                        for _ in range(int(self._expert_layers))
                    ]
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
        if self.expert_mode == "variant":
            expert = self.experts[expert_key]
            if next(expert.parameters()).device != h.device:
                # Defensive: ensures correct device even if an expert was created before .to(device).
                expert = expert.to(h.device)
                self.experts[expert_key] = expert
            return expert(h)

        sk = self._skill_key_from_expert_key(expert_key)
        expert2 = self.skill_experts[sk]
        adapters = self.adapters[expert_key]
        if next(expert2.parameters()).device != h.device:
            expert2 = expert2.to(h.device)
            self.skill_experts[sk] = expert2
        if len(list(adapters.parameters())) > 0 and next(adapters.parameters()).device != h.device:
            adapters = adapters.to(h.device)
            self.adapters[expert_key] = adapters
        return expert2(h, adapters=adapters)


@torch.no_grad()
def _exact_match_acc(
    *,
    model: TrunkPlusExperts,
    pool: VariantPool,
    expert_key: str,
    device: torch.device,
    eval_batch_size: int,
    save_unsolved_dir: Optional[Path] = None,
    save_unsolved_max: int = 0,
    save_unsolved_step: Optional[int] = None,
    save_unsolved_tag: str = "test",
    save_unsolved_class: Optional[str] = None,
) -> float:
    model.eval()
    grid_tokens = int(pool.grid_size) * int(pool.grid_size)
    n = int(pool.n)
    if n <= 0:
        return 0.0
    bs = max(1, int(eval_batch_size))
    correct = 0
    saved = 0
    for off in range(0, n, bs):
        xb = pool.src[off : off + bs].to(device, non_blocking=True)
        yb = pool.tgt[off : off + bs].to(device, non_blocking=True)
        logits = model(xb, expert_key=expert_key)  # (B,T,V)
        pred_logits = logits[:, -(grid_tokens + 1) : -1, :]
        pred = torch.argmax(pred_logits, dim=-1)
        eq = (pred == yb).all(dim=1)
        correct += int(eq.sum().item())

        # Save fixed "latest" slots so evals overwrite in-place (no per-step file accumulation).
        if save_unsolved_dir is not None and int(save_unsolved_max) > 0 and saved < int(save_unsolved_max):
            step_s = "na" if save_unsolved_step is None else f"{int(save_unsolved_step):07d}"
            cls = str(save_unsolved_class) if save_unsolved_class is not None else "unknown_class"
            base_dir = Path(save_unsolved_dir) / str(save_unsolved_tag) / cls
            bad = (~eq).nonzero(as_tuple=False).reshape(-1).tolist()
            good = (eq).nonzero(as_tuple=False).reshape(-1).tolist()
            pick = bad + good
            if len(pick) == 0:
                continue
            while saved < int(save_unsolved_max):
                bi = int(pick[saved % len(pick)])
                g = int(pool.grid_size)
                demos, test_x = decode_prompt_src(
                    src_tokens=xb[bi].detach().cpu().numpy(),
                    grid_size=g,
                    num_demos=3,
                )
                pred_y = pred[bi].detach().cpu().numpy().reshape(g, g)
                true_y = yb[bi].detach().cpu().numpy().reshape(g, g)

                row = int(off) + int(bi)
                out_path = base_dir / f"slot{int(saved):02d}.png"
                title = f"{save_unsolved_tag} latest | {cls} | step={step_s} | row={row} | slot={int(saved):02d}"
                save_arc_prompt_prediction_png(
                    demos=demos,
                    test_x=test_x,
                    pred_y=pred_y,
                    true_y=true_y,
                    out_path=out_path,
                    title=title,
                )
                saved += 1
                if saved >= int(save_unsolved_max):
                    break
    return float(correct) / float(n)


def main(
    *,
    data_dir: Path,
    grid_size: int,
    pos_encoding: str,
    train_skills: Optional[list[int]],
    skill_allowed_times: Optional[list[int]] = None,
    curriculum_unadmitted_prob: float = 0.0,
    balanced_skill_sampling: bool = True,
    skill_freeze_admit_ratio_threshold: float = 0.05,
    skill_freeze_min_new_tries: int = 10,
    skill_freeze_outer_steps: int = 20,
    expert_mode: str = "skill_adapter",
    adapter_dim: int = 32,
    adapter_scale: float = 1.0,
    n_warmup: int = 2000,
    n_outer: int = 2000,
    n_inner_steps: int = 100,
    inner_batch_size: int = 256,
    warmup_batch_size: int = 64,
    val_frac: float = 0.2,
    admit_threshold: float = 0.50,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_heads_trunk: Optional[int] = None,
    num_heads_expert: Optional[int] = None,
    trunk_layers: int = 4,
    expert_layers: int = 2,
    ff_dim: int = 256,
    dropout: float = 0.0,
    lr_warmup: float = 5e-4,
    lr_inner: float = 5e-4,
    lr_trunk_inner: float = 5e-6,
    eval_batch_size: int = 256,
    eval_every: int = 500,
    eval_keys_per_skill: int = 8,
    plot_unsolved_n: int = 3,
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

    # Load and group by (skill,variant).
    # - We train/admit on the ID "train" split (with an internal heldout partition controlled by val_frac).
    # - We also load the OOD split for evaluation (also internally split by val_frac so each key has eval data).
    train_pools, val_pools = load_variant_pools(
        data_dir=Path(data_dir),
        skill_ids=skills,
        split="train",
        rng=rng,
        val_frac=float(val_frac),
    )
    ood_train_pools, ood_val_pools = load_variant_pools(
        data_dir=Path(data_dir),
        skill_ids=skills,
        split="ood",
        rng=rng,
        val_frac=float(val_frac),
    )

    # Ensure OOD pools cover all keys seen in ID training (needed for consistent learning curves).
    missing_ood = [k for k in train_pools.keys() if k not in ood_val_pools]
    if missing_ood:
        miss_s = ", ".join(k.to_str() for k in missing_ood[:8])
        raise ValueError(
            "Some (skill,variant) keys exist in train split but are missing from OOD split. "
            f"Examples: {miss_s}"
        )

    # Sanity: grid sizes must match the configured grid_size for this run.
    for k, p in list(train_pools.items()) + list(val_pools.items()):
        if int(p.grid_size) != int(grid_size):
            raise ValueError(f"Pool {k.to_str()} grid_size={p.grid_size} != --grid_size={grid_size}")

    seq_len = prompt_seq_len(grid_size=int(grid_size), num_demos=3)
    trunk_heads = int(num_heads) if num_heads_trunk is None else int(num_heads_trunk)
    expert_heads = int(num_heads) if num_heads_expert is None else int(num_heads_expert)
    trunk_layers_i = int(trunk_layers)
    expert_layers_i = int(expert_layers)
    if trunk_layers_i < 1:
        raise ValueError(f"trunk_layers must be >= 1, got {trunk_layers_i}")
    if expert_layers_i < 1:
        raise ValueError(f"expert_layers must be >= 1, got {expert_layers_i}")
    model = TrunkPlusExperts(
        grid_size=int(grid_size),
        max_len=int(seq_len),
        pos_encoding=str(pos_encoding),
        embed_dim=int(embed_dim),
        num_heads_trunk=int(trunk_heads),
        num_heads_expert=int(expert_heads),
        trunk_layers=int(trunk_layers_i),
        expert_layers=int(expert_layers_i),
        ff_dim=int(ff_dim),
        dropout=float(dropout),
        vocab_size=int(VOCAB_SIZE),
        expert_mode=str(expert_mode),
        adapter_dim=int(adapter_dim),
        adapter_scale=float(adapter_scale),
    ).to(device_t)

    # Pre-create experts for all variant keys for deterministic parameter counts.
    keys = sorted(train_pools.keys(), key=lambda kk: kk.to_str())
    for k in keys:
        model.ensure_expert(k)

    total_params, trainable_params = count_params(model)
    if model.expert_mode == "variant":
        print(f"Model params: total={total_params:,} trainable={trainable_params:,} | experts={len(model.experts)}")
    else:
        print(
            f"Model params: total={total_params:,} trainable={trainable_params:,} | "
            f"skill_experts={len(model.skill_experts)} adapters={len(model.adapters)}"
        )

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
    ood_train_pools = {k: maybe_move_pool(p) for k, p in ood_train_pools.items()}
    ood_val_pools = {k: maybe_move_pool(p) for k, p in ood_val_pools.items()}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_enabled = not bool(no_plots)
    plot_unsolved_n_i = int(plot_unsolved_n)
    if plot_unsolved_n_i < 0:
        raise ValueError(f"plot_unsolved_n must be >= 0, got {plot_unsolved_n_i}")

    # Save run config (CLI args) alongside plots/CSV/weights so downstream scripts can reconstruct the model.
    run_config = {
        "data_dir": str(Path(data_dir)),
        "out_dir": str(out_dir),
        "grid_size": int(grid_size),
        "pos_encoding": str(pos_encoding),
        "train_skills": [int(s) for s in skills],
        "val_frac": float(val_frac),
        "seed": int(seed),
        "device": str(device),
        "dataset_device": str(dataset_device),
        # Curriculum / sampling
        "skill_allowed_times": [int(x) for x in skill_allowed_times] if skill_allowed_times is not None else None,
        "curriculum_unadmitted_prob": float(curriculum_unadmitted_prob),
        "balanced_skill_sampling": bool(balanced_skill_sampling),
        # Additional gating heuristic
        "skill_freeze_admit_ratio_threshold": float(skill_freeze_admit_ratio_threshold),
        "skill_freeze_min_new_tries": int(skill_freeze_min_new_tries),
        "skill_freeze_outer_steps": int(skill_freeze_outer_steps),
        # Model hyperparams (must match checkpoint load)
        "embed_dim": int(embed_dim),
        # Keep legacy "num_heads" for backwards compatibility with older analysis scripts/config readers.
        # When trunk/expert head counts differ, "num_heads" is set to the trunk value.
        "num_heads": int(trunk_heads),
        "num_heads_trunk": int(trunk_heads),
        "num_heads_expert": int(expert_heads),
        "ff_dim": int(ff_dim),
        "dropout": float(dropout),
        "trunk_layers": int(trunk_layers_i),
        "expert_layers": int(expert_layers_i),
        "expert_mode": str(expert_mode),
        "adapter_dim": int(adapter_dim),
        "adapter_scale": float(adapter_scale),
        "vocab_size": int(VOCAB_SIZE),
        "num_demos": 3,
        # Training schedule (useful metadata)
        "n_warmup": int(n_warmup),
        "n_outer": int(n_outer),
        "n_inner_steps": int(n_inner_steps),
        "inner_batch_size": int(inner_batch_size),
        "warmup_batch_size": int(warmup_batch_size),
        "admit_threshold": float(admit_threshold),
        "lr_warmup": float(lr_warmup),
        "lr_inner": float(lr_inner),
        "lr_trunk_inner": float(lr_trunk_inner),
        "eval_batch_size": int(eval_batch_size),
        "eval_every": int(eval_every),
        "eval_keys_per_skill": int(eval_keys_per_skill),
        "outer_log_every": int(outer_log_every),
        "no_plots": bool(no_plots),
    }
    (plots_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Warmup: mix *all examples* together and train a generic trunk-only model.
    # This warmup model is thrown away after warmup; we only copy its trunk weights.
    mixed_train = maybe_move_pool(_make_mixed_pool(pools=train_pools, split="train_mixed"))
    mixed_val = maybe_move_pool(_make_mixed_pool(pools=val_pools, split="val_mixed"))

    warm_model = WarmupTrunk(
        grid_size=int(grid_size),
        max_len=int(seq_len),
        pos_encoding=str(pos_encoding),
        embed_dim=int(embed_dim),
        num_heads=int(trunk_heads),
        trunk_layers=int(trunk_layers_i),
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

    # Curriculum over skills: restrict which (skill,variant) keys can be sampled at each outer step.
    # skill_allowed_times is a list [n_0, n_1, ...] aligned to the --train_skills order (NOT sorted).
    # This makes curricula easier to reason about when you want a specific progression.
    skill_ids_ordered = [int(sid) for sid in skills]
    missing_in_data = [int(sid) for sid in skill_ids_ordered if int(sid) not in keys_by_skill]
    if missing_in_data:
        raise ValueError(
            "Some skills from --train_skills have no (skill,variant) keys in the loaded dataset. "
            f"Missing skills: {missing_in_data}"
        )
    allowed_time_by_skill: dict[int, int] = {int(sid): 0 for sid in skill_ids_ordered}
    if skill_allowed_times is not None:
        times = [int(x) for x in skill_allowed_times]
        if len(times) != len(skill_ids_ordered):
            raise ValueError(
                "--skill_allowed_times must have exactly one entry per --train_skills entry (aligned to that order). "
                f"Got len(times)={len(times)} for train_skills={skill_ids_ordered}"
            )
        if any(int(t) < 0 for t in times):
            raise ValueError(f"--skill_allowed_times entries must be >= 0, got {times}")
        if times and int(times[0]) != 0:
            raise ValueError(
                "The first skill must be allowed at outer step 0 (n_0 == 0), otherwise no keys are available to train."
            )
        for i in range(1, len(times)):
            if int(times[i]) < int(times[i - 1]):
                raise ValueError(
                    "--skill_allowed_times must be non-decreasing to form a curriculum. "
                    f"Got times={times} for train_skills={skill_ids_ordered}"
                )
        allowed_time_by_skill = {int(skill_ids_ordered[i]): int(times[i]) for i in range(len(skill_ids_ordered))}

    def _is_skill_allowed(*, skill_id: int, outer_step: int) -> bool:
        return int(outer_step) >= int(allowed_time_by_skill.get(int(skill_id), 0))

    # Additional gating heuristic:
    # If a skill's recent admitted ratio is very low, temporarily freeze it to avoid repeatedly spending
    # compute on variants that almost always RESET. This uses gating-only counters that reset on each freeze.
    freeze_until_by_skill: dict[int, int] = {int(sid): 0 for sid in skill_ids_ordered}
    gate_tries_by_skill: dict[int, int] = {int(sid): 0 for sid in skill_ids_ordered}
    gate_admits_by_skill: dict[int, int] = {int(sid): 0 for sid in skill_ids_ordered}

    ratio_thr = float(skill_freeze_admit_ratio_threshold)
    if not (0.0 <= ratio_thr <= 1.0):
        raise ValueError(f"skill_freeze_admit_ratio_threshold must be in [0,1], got {ratio_thr}")
    min_new_tries = int(skill_freeze_min_new_tries)
    if min_new_tries < 1:
        raise ValueError(f"skill_freeze_min_new_tries must be >= 1, got {min_new_tries}")
    freeze_steps = int(skill_freeze_outer_steps)
    if freeze_steps < 0:
        raise ValueError(f"skill_freeze_outer_steps must be >= 0, got {freeze_steps}")

    def _is_skill_frozen(*, skill_id: int, outer_step: int) -> bool:
        return int(outer_step) < int(freeze_until_by_skill.get(int(skill_id), 0))

    def _fmt_skill_gate_stats(*, outer_step: int) -> str:
        parts: list[str] = []
        for sid in skill_ids_ordered:
            sid_i = int(sid)
            tries = int(gate_tries_by_skill.get(sid_i, 0))
            admits = int(gate_admits_by_skill.get(sid_i, 0))
            ratio = float(admits) / float(max(1, tries))
            rem = max(0, int(freeze_until_by_skill.get(sid_i, 0)) - int(outer_step))
            if rem > 0:
                parts.append(f"s{sid_i} {admits}/{tries}={ratio:.1%} frozen={rem}")
            else:
                parts.append(f"s{sid_i} {admits}/{tries}={ratio:.1%}")
        return " | ".join(parts)

    # Sampling policy:
    # - Historically we only sampled from unadmitted keys until they were admitted.
    # - With a curriculum, that can starve earlier skills once a new skill unlocks.
    # So: when curriculum is enabled, sample from unadmitted keys with probability p, else from all allowed keys.
    p_unadmitted = float(curriculum_unadmitted_prob)
    if not (0.0 <= p_unadmitted <= 1.0):
        raise ValueError(f"curriculum_unadmitted_prob must be in [0,1], got {p_unadmitted}")

    def _eval_skill_acc(*, which: str, unsolved_dir: Optional[Path], unsolved_step: int) -> dict[int, float]:
        if which not in {"train", "test", "ood"}:
            raise ValueError(f"which must be 'train', 'test', or 'ood', got {which!r}")
        if which == "train":
            pools = train_pools
        elif which == "test":
            pools = val_pools
        else:
            pools = ood_val_pools
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
                        save_unsolved_dir=unsolved_dir if which in {"test", "ood"} else None,
                        save_unsolved_max=int(plot_unsolved_n_i) if which in {"test", "ood"} else 0,
                        save_unsolved_step=int(unsolved_step),
                        save_unsolved_tag=str(which),
                        save_unsolved_class=str(kk2.to_str()),
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
        unsolved_dir = plots_dir / "unsolved_examples"
        acc_train = _eval_skill_acc(which="train", unsolved_dir=None, unsolved_step=int(step))
        acc_test = _eval_skill_acc(which="test", unsolved_dir=unsolved_dir, unsolved_step=int(step))
        acc_ood = _eval_skill_acc(which="ood", unsolved_dir=unsolved_dir, unsolved_step=int(step))

        curves.steps.append(int(step))
        curves.loss.append(float(last_loss))
        curves.probe_train_ood.append(0.0)
        curves.probe_fully_heldout_ood.append(0.0)
        for sid in sorted(keys_by_skill.keys()):
            curves.acc_train[int(sid)].append(float(acc_train.get(int(sid), 0.0)))
            curves.acc_id[int(sid)].append(float(acc_test.get(int(sid), 0.0)))
            curves.acc_ood[int(sid)].append(float(acc_ood.get(int(sid), 0.0)))

        metrics_csv = plots_dir / "learning_curves_latest.csv"
        write_learning_curves_csv(curves=curves, skills=sorted(keys_by_skill.keys()), out_path=metrics_csv)

        # Save weights alongside CSV/PNG.
        # - weights_latest.pt is overwritten each eval
        ckpt_latest = plots_dir / "weights_latest.pt"
        payload = {
            "step": int(step),
            "model_state_dict": model.state_dict(),
        }
        torch.save(payload, ckpt_latest)

        if plots_enabled:
            title = (
                "ARC fewshot (trunk+experts) learning curves (exact-match acc)\n"
                f"skills={sorted(keys_by_skill.keys())} | eval_keys_per_skill={int(eval_keys_per_skill)} | "
                f"admit_threshold={float(admit_threshold):.2f} | admitted={len(admitted)} | "
                f"resets={sum(int(v) for v in fail_count.values())} | "
                f"lr_warmup={float(lr_warmup):.2e} lr_inner={float(lr_inner):.2e} lr_trunk_inner={float(lr_trunk_inner):.2e}"
            )
            latest = plots_dir / "learning_curves_latest.png"
            plot_learning_curves(curves=curves, skills=sorted(keys_by_skill.keys()), out_path=latest, title=title)

        def fmt(d: dict[int, float]) -> str:
            return " ".join(f"s{sid}={d[int(sid)]:.3f}" for sid in sorted(keys_by_skill.keys()))

        print(
            f"eval step={int(step):6d} loss={float(last_loss):.4f} | "
            f"train: {fmt(acc_train)} | test: {fmt(acc_test)} | ood: {fmt(acc_ood)}"
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

    # Copy warmup trunk weights into the existing fewshot model (do NOT re-initialize the model).
    # This preserves expert initialization and avoids a second model construction after warmup.
    with torch.no_grad():
        model.embed.weight.copy_(warm_model.embed.weight)
        model.global_pos_enc.copy_(warm_model.global_pos_enc)
        if model.pos_encoding == "2d":
            model.row_embed.weight.copy_(warm_model.row_embed.weight)  # type: ignore[attr-defined]
            model.col_embed.weight.copy_(warm_model.col_embed.weight)  # type: ignore[attr-defined]
        model.trunk.load_state_dict(warm_model.trunk.state_dict())
    # Free warmup model memory (useful on GPU).
    del warm_model

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
        # Optionally balance across skills by sampling skill uniformly, then variant within skill.
        allowed_skill_ids = [
            int(sid)
            for sid in skill_ids_ordered
            if _is_skill_allowed(skill_id=int(sid), outer_step=int(outer_step))
        ]
        if len(allowed_skill_ids) == 0:
            # Should be prevented by validation (n_0 == 0), but keep this explicit and readable.
            raise ValueError(
                f"No skills are allowed at outer_step={int(outer_step)}. "
                f"--skill_allowed_times={skill_allowed_times} train_skills={skill_ids_ordered}"
            )

        eligible_skill_ids = [
            int(sid) for sid in allowed_skill_ids if not _is_skill_frozen(skill_id=int(sid), outer_step=int(outer_step))
        ]
        # If everything is frozen, fall back to allowed skills so training can still proceed.
        if len(eligible_skill_ids) == 0:
            eligible_skill_ids = allowed_skill_ids

        prefer_unadmitted = float(rng.random()) < float(p_unadmitted)

        if bool(balanced_skill_sampling):
            # Balanced sampling means: pick a skill uniformly, then pick a variant uniformly within that skill.
            # We still "prefer unadmitted" *within the chosen skill*, but we do NOT drop the skill entirely if it
            # has no unadmitted variants (otherwise skills with many variants dominate after early admissions).
            cand_by_skill: dict[int, list[VariantKey]] = {}
            for sid in eligible_skill_ids:
                all_kk = keys_by_skill.get(int(sid), [])
                if len(all_kk) == 0:
                    continue
                if prefer_unadmitted:
                    unadm = [kk for kk in all_kk if model.ensure_expert(kk) not in admitted]
                    kk_list = unadm if len(unadm) > 0 else all_kk
                else:
                    kk_list = all_kk
                cand_by_skill[int(sid)] = kk_list
            if len(cand_by_skill) == 0:
                raise ValueError(f"No variant keys available for eligible skills={eligible_skill_ids}")

            pick_skills = list(cand_by_skill.keys())
            sid = int(pick_skills[int(rng.integers(0, len(pick_skills)))])
            kk_list2 = cand_by_skill[int(sid)]
            k = kk_list2[int(rng.integers(0, len(kk_list2)))]
        else:
            allowed_keys = [k for sid in eligible_skill_ids for k in keys_by_skill.get(int(sid), [])]
            if len(allowed_keys) == 0:
                raise ValueError(f"No variant keys available for eligible skills={eligible_skill_ids}")
            if prefer_unadmitted:
                unadmitted = [kk for kk in allowed_keys if model.ensure_expert(kk) not in admitted]
                cand = unadmitted if len(unadmitted) > 0 else allowed_keys
            else:
                cand = allowed_keys
            k = cand[int(rng.integers(0, len(cand)))]

        ek = model.ensure_expert(k)

        # Snapshot expert params so we can revert if not admitted.
        if model.expert_mode == "variant":
            fast_module: nn.Module = model.experts[ek]
            reset_module: nn.Module = fast_module
            fast_params = list(fast_module.parameters())
        else:
            # Shared skill expert + per-variant adapters. We only RESET the adapters on failure.
            sk = f"s{int(k.skill_id)}"
            skill_module = model.skill_experts[sk]
            adapter_module = model.adapters[ek]
            reset_module = adapter_module
            fast_params = list(skill_module.parameters()) + list(adapter_module.parameters())
        before = {name: p.detach().clone() for name, p in reset_module.named_parameters()}

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
                {"params": fast_params, "lr": float(lr_inner), "weight_decay": 0.0},
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
                for name, p in reset_module.named_parameters():
                    p.copy_(before[name])
            fail_count[k.to_str()] = int(fail_count.get(k.to_str(), 0)) + 1

        # Update gating-only stats for this skill and possibly freeze.
        sid_int = int(k.skill_id)
        gate_tries_by_skill[sid_int] = int(gate_tries_by_skill.get(sid_int, 0)) + 1
        if ok:
            gate_admits_by_skill[sid_int] = int(gate_admits_by_skill.get(sid_int, 0)) + 1
        tries = int(gate_tries_by_skill.get(sid_int, 0))
        admits = int(gate_admits_by_skill.get(sid_int, 0))
        admit_ratio = float(admits) / float(max(1, tries))
        if freeze_steps > 0 and tries >= int(min_new_tries) and float(admit_ratio) < float(ratio_thr):
            freeze_until_by_skill[sid_int] = int(outer_step) + int(freeze_steps)
            resets = int(tries) - int(admits)
            print(
                f"FREEZE | skill=s{sid_int} | admit_ratio={float(admit_ratio):.3%} "
                f"(admits={admits} resets={resets} tries={tries}) | frozen_for={int(freeze_steps)} outer steps"
            )
            # Reset gating-only counters for this skill (does NOT touch global admission/fail_count).
            gate_tries_by_skill[sid_int] = 0
            gate_admits_by_skill[sid_int] = 0

        log_every = int(outer_log_every)
        if log_every <= 0:
            raise ValueError(f"outer_log_every must be >= 1, got {log_every}")
        if (int(outer_step) % log_every == 0) or (int(outer_step) == int(n_outer) - 1):
            print(
                f"outer step={int(outer_step):5d} | key={k.to_str():>20s} | val_acc={acc:.3f} | "
                f"{'ADMIT' if ok else 'RESET'} | admitted={len(admitted)} resets={sum(int(v) for v in fail_count.values())}"
            )
            print(f"  per-skill gate: {_fmt_skill_gate_stats(outer_step=int(outer_step))}")

    # Final weights snapshot.
    torch.save({"step": int(global_step), "model_state_dict": model.state_dict()}, plots_dir / "weights_final.pt")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fewshot ARC training with per-(skill,variant) experts + admission.")
    p.add_argument("--data_dir", type=Path, default=Path("tmp"))
    p.add_argument("--grid_size", type=int, default=6)
    p.add_argument("--pos_encoding", type=str, default="2d", choices=["2d", "1d"])
    p.add_argument("--train_skills", type=int, nargs="*", default=None)
    p.add_argument(
        "--skill_allowed_times",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Curriculum over skills: list of outer-step thresholds [n_0, n_1, ...] aligned to the --train_skills order. "
            "Skill i (in that order) is only allowed when outer_step >= n_i. Must be non-decreasing and start with 0."
        ),
    )
    p.add_argument(
        "--curriculum_unadmitted_prob",
        type=float,
        default=0.0,
        help=(
            "Sample from unadmitted keys with probability p (otherwise sample from any allowed key). "
            "Default 0.0 means no bias toward unadmitted (more replay, less forgetting)."
        ),
    )
    p.add_argument(
        "--balanced_skill_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set (default), sample skill uniformly then variant uniformly within skill (prevents skills with many variants from dominating).",
    )
    p.add_argument(
        "--expert_mode",
        type=str,
        default="skill_adapter",
        choices=["skill_adapter", "variant"],
        help="Expert parameterization. Default uses shared per-skill experts plus per-variant per-layer adapters.",
    )
    p.add_argument("--adapter_dim", type=int, default=32, help="Adapter bottleneck dim (for --expert_mode=skill_adapter).")
    p.add_argument("--adapter_scale", type=float, default=1.0, help="Adapter residual scale (for --expert_mode=skill_adapter).")

    p.add_argument("--n_warmup", type=int, default=2000)
    p.add_argument("--n_outer", type=int, default=2000)
    p.add_argument("--n_inner_steps", type=int, default=100)
    p.add_argument("--inner_batch_size", type=int, default=256)
    p.add_argument("--warmup_batch_size", type=int, default=64)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--admit_threshold", type=float, default=0.5)
    p.add_argument(
        "--skill_freeze_admit_ratio_threshold",
        type=float,
        default=0.05,
        help="If a skill's recent admitted ratio (admits/tries) is below this threshold and it has enough new tries, freeze it temporarily.",
    )
    p.add_argument(
        "--skill_freeze_min_new_tries",
        type=int,
        default=10,
        help="Minimum number of new tries (since last gating reset) before considering freezing a skill.",
    )
    p.add_argument(
        "--skill_freeze_outer_steps",
        type=int,
        default=20,
        help="How many outer steps to freeze a skill for when the freeze heuristic triggers. Set to 0 to disable.",
    )

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument(
        "--num_heads",
        type=int,
        default=10,
        help="Legacy alias for both trunk and expert heads. Prefer --num_heads_trunk / --num_heads_expert.",
    )
    p.add_argument(
        "--num_heads_trunk",
        type=int,
        default=None,
        help="Number of attention heads for the shared trunk transformer. Defaults to --num_heads when omitted.",
    )
    p.add_argument(
        "--num_heads_expert",
        type=int,
        default=None,
        help="Number of attention heads for the expert transformer(s). Defaults to --num_heads when omitted.",
    )
    p.add_argument("--trunk_layers", type=int, default=4, help="Number of transformer layers in the shared trunk.")
    p.add_argument("--expert_layers", type=int, default=2, help="Number of transformer layers in the expert stack(s).")
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr_warmup", type=float, default=5e-4)
    p.add_argument("--lr_inner", type=float, default=1e-3)
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
    p.add_argument(
        "--plot_unsolved_n",
        type=int,
        default=3,
        help="Per-(skill,variant) number of unsolved eval examples to render as PNG during eval (0 disables).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset_device", type=str, default="gpu", choices=["cpu", "gpu"])
    p.add_argument("--progress", action="store_true")
    p.add_argument("--out_dir", type=Path, default=Path("arc_train_runs_fewshot"), help="Where to write plots/metrics")
    p.add_argument("--no_plots", action="store_true", help="Disable saving learning-curve PNGs (CSV still written)")
    return p


def cli_main(argv: Optional[list[str]] = None) -> None:
    # Normalize argv to support both:
    # - --skill_allowed_times 0 300 400
    # - --skill_allowed_times=0 300 400
    # Argparse will not treat the latter as a multi-arg option unless we rewrite it.
    if argv is None:
        argv = sys.argv[1:]
    norm_argv: list[str] = []
    for a in argv:
        if str(a).startswith("--skill_allowed_times="):
            _, v = str(a).split("=", 1)
            norm_argv.append("--skill_allowed_times")
            if v != "":
                norm_argv.append(v)
        else:
            norm_argv.append(str(a))

    args = _build_arg_parser().parse_args(norm_argv)
    main(
        data_dir=Path(args.data_dir),
        grid_size=int(args.grid_size),
        pos_encoding=str(args.pos_encoding),
        train_skills=[int(s) for s in args.train_skills] if args.train_skills is not None else None,
        skill_allowed_times=[int(x) for x in args.skill_allowed_times] if args.skill_allowed_times is not None else None,
        curriculum_unadmitted_prob=float(args.curriculum_unadmitted_prob),
        balanced_skill_sampling=bool(args.balanced_skill_sampling),
        skill_freeze_admit_ratio_threshold=float(args.skill_freeze_admit_ratio_threshold),
        skill_freeze_min_new_tries=int(args.skill_freeze_min_new_tries),
        skill_freeze_outer_steps=int(args.skill_freeze_outer_steps),
        expert_mode=str(args.expert_mode),
        adapter_dim=int(args.adapter_dim),
        adapter_scale=float(args.adapter_scale),
        n_warmup=int(args.n_warmup),
        n_outer=int(args.n_outer),
        n_inner_steps=int(args.n_inner_steps),
        inner_batch_size=int(args.inner_batch_size),
        warmup_batch_size=int(args.warmup_batch_size),
        val_frac=float(args.val_frac),
        admit_threshold=float(args.admit_threshold),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_heads_trunk=(int(args.num_heads_trunk) if args.num_heads_trunk is not None else None),
        num_heads_expert=(int(args.num_heads_expert) if args.num_heads_expert is not None else None),
        trunk_layers=int(args.trunk_layers),
        expert_layers=int(args.expert_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        lr_warmup=float(args.lr_warmup),
        lr_inner=float(args.lr_inner),
        lr_trunk_inner=float(args.lr_trunk_inner),
        eval_batch_size=int(args.eval_batch_size),
        eval_every=int(args.eval_every),
        eval_keys_per_skill=int(args.eval_keys_per_skill),
        plot_unsolved_n=int(args.plot_unsolved_n),
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


