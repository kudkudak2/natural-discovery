from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from arc_train_utils import (
    LearningCurves,
    TensorizedDataset,
    VOCAB_SIZE,
    cap_dataset,
    concat_datasets,
    count_params,
    evaluate_accuracy,
    load_skill_split,
    maybe_load_skill_split,
    maybe_move_train_pool,
    plot_learning_curves,
    prepare_batch,
    progress as progress_iter,
    prompt_seq_len,
    show_one_example,
    split_dataset,
    write_learning_curves_csv,
)
from checkpointing import load_pretrained_weights


DEFAULT_TRAIN_SKILLS = (11, 12, 14, 15, 16)
DEFAULT_TRAIN_WITH_OOD_SKILLS = (11, 12, 14, 15, 16)


def _unique_ints(xs: list[int]) -> list[int]:
    """Stable unique (preserves first occurrence order)."""
    out: list[int] = []
    seen: set[int] = set()
    for x in xs:
        xi = int(x)
        if xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
    return out


def _curriculum_delay_from_phases(
    *,
    phase1_skills: list[int],
    phase2_skills: list[int],
    phase2_start_step: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Convert a 2-phase curriculum into (train_skills, delay_train_skills, delay_train_until_steps).

    Phase 1: train only on `phase1_skills` starting at step 0.
    Phase 2: train on `phase2_skills` starting at `phase2_start_step`.
    """
    p1 = _unique_ints([int(s) for s in phase1_skills])
    p2 = _unique_ints([int(s) for s in phase2_skills])
    if len(p1) == 0:
        raise ValueError("phase1_skills must be non-empty")
    if len(p2) == 0:
        raise ValueError("phase2_skills must be non-empty")

    start = int(phase2_start_step)
    if start < 0:
        raise ValueError(f"phase2_start_step must be >= 0, got {start}")

    train_skills = _unique_ints(p1 + p2)
    phase2_only = [s for s in p2 if s not in set(p1)]
    delay_skills = _unique_ints(phase2_only)
    delay_steps = [int(start) for _ in delay_skills]
    return train_skills, delay_skills, delay_steps
class ARCTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = VOCAB_SIZE,
        grid_size: int = 5,
        pos_encoding: str = "2d",
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        max_len: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        if self.grid_size <= 0:
            raise ValueError(f"grid_size must be >= 1, got {self.grid_size}")
        self.grid_tokens = self.grid_size * self.grid_size

        self.pos_encoding = str(pos_encoding).lower()
        if self.pos_encoding not in {"2d", "1d"}:
            raise ValueError(f"pos_encoding must be one of {{'2d','1d'}}, got {pos_encoding!r}")

        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Always include a *global* 1D positional encoding so the model can distinguish
        # "demo1 input" vs "demo1 output" vs "test input" even when (row, col) repeats.
        self.global_pos_enc = nn.Parameter(torch.randn(1, int(max_len), embed_dim) * 0.02)

        # Optional *local* 2D positional encoding (row + col) to restore spatial inductive bias.
        # Note: SEP tokens (between grids) will receive a 0 2D positional embedding.
        if self.pos_encoding == "2d":
            self.row_embed = nn.Embedding(self.grid_size, embed_dim)
            self.col_embed = nn.Embedding(self.grid_size, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        _b, t = x.shape
        if t > int(self.global_pos_enc.shape[1]):
            raise ValueError(
                f"Sequence too long: t={t} > max_len={int(self.global_pos_enc.shape[1])}. "
                "Increase max_len."
            )

        emb = self.embed(x) + self.global_pos_enc[:, :t, :]

        if self.pos_encoding == "2d":
            # NOTE: if we move to non-square grids, we need to adjust the positional embeddings.
            # Prompt layout is a repetition of: [grid_tokens] + [SEP]
            # So we can compute 2D positions by position-in-block and automatically reset per grid.
            block = self.grid_tokens + 1
            pos = torch.arange(t, device=x.device)
            within = pos % block  # [0..grid_tokens] where grid_tokens is the SEP position
            is_sep = within == self.grid_tokens
            cell = torch.clamp(within, max=self.grid_tokens - 1)

            row = (cell // self.grid_size).to(torch.long)
            col = (cell % self.grid_size).to(torch.long)
            pos_emb_2d = self.row_embed(row) + self.col_embed(col)  # (T, D)
            pos_emb_2d = pos_emb_2d.masked_fill(is_sep.unsqueeze(-1), 0.0)
            emb = emb + pos_emb_2d.unsqueeze(0)  # (B, T, D)

        h = self.transformer(emb)
        return self.fc_out(h)  # (B, T, vocab)


def main(
    data_dir: Path = Path("tmp"),
    grid_size: int = 5,
    pos_encoding: str = "2d",
    pretrained: Optional[Path] = None,
    train_skills: Optional[list[int]] = None,
    delay_train_skill: Optional[int] = None,
    delay_train_until_step: int = 0,
    delay_train_skills: Optional[list[int]] = None,
    delay_train_until_steps: Optional[list[int]] = None,
    probe_skill: int = 8,
    cap_train_skill3: Optional[int] = None,
    cap_train_skill: Optional[int] = None,
    cap_train_n: Optional[int] = None,
    train_with_ood_skills: Optional[list[int]] = None,
    ood_train_frac: float = 0.5,
    test_frac: float = 0.2,
    steps: int = 3000,
    batch_size: int = 32,
    lr: float = 5e-4,
    lr_decay: str = "cosine",
    min_lr: float = 0.0,
    weight_decay: float = 0.01, # 0.1 seems too high actually
    seed: int = 0,
    device: str = "cuda", # if torch.cuda.is_available() else "cpu",
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    ff_dim: int = 256,
    dropout: float = 0.0,
    eval_every: int = 500,
    save_every: int = 500,
    eval_tasks: int = 128,
    eval_batch_size: int = 256,
    progress: bool = False,
    out_dir: Path = Path("arc_train_runs"),
    no_plots: bool = False,
    dataset_device: str = "gpu",
) -> None:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    grid_size = int(grid_size)
    grid_tokens = grid_size * grid_size
    seq_len = prompt_seq_len(grid_size=grid_size, num_demos=3)
    device = torch.device(device)
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    plots_enabled = not bool(no_plots)

    if device.type == "cuda":
        # More throughput on Ampere+; safe for this kind of toy transformer.
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    train_skills = list(DEFAULT_TRAIN_SKILLS) if train_skills is None else [int(s) for s in train_skills]
    for sid in train_skills:
        if sid < 1:
            raise ValueError(f"Invalid skill id: {sid}")

    # Delayed introduction schedule: skill -> step at which it becomes available (included when step >= until_step).
    # Back-compat: allow the old single-skill flags, but prefer the multi-skill form.
    delay_until_by_skill: dict[int, int] = {}
    if (delay_train_skills is not None) or (delay_train_until_steps is not None):
        if (delay_train_skill is not None) or (int(delay_train_until_step) != 0):
            raise ValueError(
                "Use either (--delay_train_skill, --delay_train_until_step) OR "
                "(--delay_train_skills, --delay_train_until_steps), not both."
            )
        skills_list = [] if delay_train_skills is None else [int(s) for s in delay_train_skills]
        until_list = [] if delay_train_until_steps is None else [int(s) for s in delay_train_until_steps]
        if len(skills_list) != len(until_list):
            raise ValueError(
                f"--delay_train_skills and --delay_train_until_steps must have the same length, "
                f"got {len(skills_list)} vs {len(until_list)}"
            )
        for sid, until in zip(skills_list, until_list):
            if until < 0:
                raise ValueError(f"delay step must be >= 0, got {until} for skill {sid}")
            if sid not in train_skills:
                raise ValueError(f"Delayed skill {sid} must be included in train_skills")
            if until > 0:
                delay_until_by_skill[sid] = int(until)
    else:
        if delay_train_until_step < 0:
            raise ValueError(f"delay_train_until_step must be >= 0, got {delay_train_until_step}")
        if delay_train_skill is not None:
            delay_train_skill = int(delay_train_skill)
            if delay_train_skill not in train_skills:
                raise ValueError(f"delay_train_skill={delay_train_skill} must be included in train_skills")
            if int(delay_train_until_step) > 0:
                delay_until_by_skill[int(delay_train_skill)] = int(delay_train_until_step)

    probe_skill = int(probe_skill)
    if probe_skill < 1:
        raise ValueError(f"Invalid --probe_skill: {probe_skill}")

    test_frac_f = float(test_frac)
    if not (0.0 <= test_frac_f < 1.0):
        raise ValueError(f"--test_frac must be in [0,1), got {test_frac_f}")
    if test_frac_f > 0.5:
        raise ValueError(f"--test_frac too large ({test_frac_f}); pick <= 0.5 for a meaningful train split.")
    train_frac_f = 1.0 - test_frac_f

    # Back-compat: if --cap_train_skill3 is set and the generalized flags aren't, map it.
    cap_skill: Optional[int] = int(cap_train_skill) if cap_train_skill is not None else None
    cap_n: Optional[int] = int(cap_train_n) if cap_train_n is not None else None
    if cap_train_skill3 is not None and cap_skill is None:
        cap_skill = 3
        cap_n = int(cap_train_skill3)
    if cap_skill is not None and cap_n is None:
        raise ValueError("--cap_train_n must be provided when --cap_train_skill is set.")

    # Load datasets from disk (no on-the-fly generation) and do a deterministic train/test split.
    #
    # Important: reported accuracies are computed on the held-out "test" portions to avoid leakage.
    train_sets: dict[int, TensorizedDataset] = {}
    eval_id_sets: dict[int, TensorizedDataset] = {}
    ood_train_pools: dict[int, TensorizedDataset] = {}
    eval_ood_sets: dict[int, TensorizedDataset] = {}

    for sid in train_skills:
        ds_train_full = load_skill_split(data_dir=data_dir, skill_id=sid, split="train")
        ds_train, ds_train_test = split_dataset(ds_train_full, train_frac=train_frac_f, rng=rng)
        train_sets[sid] = ds_train
        eval_id_sets[sid] = ds_train_test

        ds_ood_full = load_skill_split(data_dir=data_dir, skill_id=sid, split="ood")
        ds_ood_train, ds_ood_test = split_dataset(ds_ood_full, train_frac=train_frac_f, rng=rng)
        ood_train_pools[sid] = ds_ood_train
        eval_ood_sets[sid] = ds_ood_test

    # Include OOD examples for selected skills in training (from their OOD-train pool).
    # OOD test remains disjoint and is what we report in the printed metrics.
    if train_with_ood_skills is None:
        train_with_ood_skills = list(DEFAULT_TRAIN_WITH_OOD_SKILLS)
    train_with_ood = {int(s) for s in train_with_ood_skills} & set(train_skills)
    # Never include probe skill OOD in training (strict OOD probe).
    train_with_ood.discard(probe_skill)
    for sid in sorted(train_with_ood):
        ood_pool = ood_train_pools[sid]
        ood_train = ood_pool
        if float(ood_train_frac) < 1.0:
            ood_train, _ood_unused = split_dataset(ood_pool, train_frac=float(ood_train_frac), rng=rng)
        if ood_train.n > 0:
            train_sets[sid] = concat_datasets(
                [train_sets[sid], ood_train],
                skill_id=sid,
                split=f"train+ood{sid}",
                grid_size=grid_size,
            )

    # Optional artificial cap for any skill: reduce training data to force learning only when possible.
    if cap_skill is not None and cap_skill in train_sets:
        train_sets[cap_skill] = cap_dataset(train_sets[cap_skill], cap=int(cap_n), rng=rng)

    # Always report the strict OOD probe (held-out OOD test), even if probe_skill is not in train_skills.
    probe_ood_full = maybe_load_skill_split(data_dir=data_dir, skill_id=probe_skill, split="ood")
    eval_probe_ood = None
    probe_ood_train: Optional[TensorizedDataset] = None
    if probe_ood_full is not None:
        probe_ood_train, probe_ood_test = split_dataset(probe_ood_full, train_frac=train_frac_f, rng=rng)
        eval_probe_ood = probe_ood_test
    if eval_probe_ood is None:
        raise FileNotFoundError(f"Missing required eval set: {data_dir / f'skill_{probe_skill}' / 'ood.json'}")

    # Sanity: ensure grid_size matches.
    for ds in list(train_sets.values()) + list(eval_id_sets.values()) + list(eval_ood_sets.values()) + [eval_probe_ood]:
        if ds.grid_size != grid_size:
            raise ValueError(f"Dataset grid_size={ds.grid_size} != --grid_size={grid_size}")

    # Build mixed training pools. Optionally delay multiple skills until specified steps.
    train_src_all = torch.cat([train_sets[sid].src for sid in train_skills], dim=0)
    train_tgt_all = torch.cat([train_sets[sid].tgt for sid in train_skills], dim=0)
    train_pool_all = TensorizedDataset(
        skill_id=-1, split="train_mix", grid_size=grid_size, src=train_src_all, tgt=train_tgt_all
    )

    def build_pool(active_skills: list[int], *, split: str) -> TensorizedDataset:
        train_src = torch.cat([train_sets[sid].src for sid in active_skills], dim=0)
        train_tgt = torch.cat([train_sets[sid].tgt for sid in active_skills], dim=0)
        return TensorizedDataset(skill_id=-1, split=split, grid_size=grid_size, src=train_src, tgt=train_tgt)

    # Precompute phase pools keyed by the step at which that pool becomes active.
    phase_starts = [0]
    if len(delay_until_by_skill) > 0:
        phase_starts += sorted(set(int(v) for v in delay_until_by_skill.values() if int(v) > 0))

    train_pool_phases: list[tuple[int, TensorizedDataset]] = []
    for start in phase_starts:
        active_skills = [sid for sid in train_skills if int(delay_until_by_skill.get(sid, 0)) <= int(start)]
        if len(active_skills) == 0:
            raise ValueError("Delaying all training skills at step 0 would leave an empty training pool.")
        if len(active_skills) == len(train_skills):
            pool = train_pool_all
        else:
            excluded = [sid for sid in train_skills if sid not in active_skills]
            excluded_s = "_".join(f"s{sid}" for sid in excluded)
            pool = build_pool(active_skills, split=f"train_mix_excl_{excluded_s}_from_{int(start)}")
        train_pool_phases.append((int(start), pool))

    # Ensure training pools won't bottleneck on CPU (avoid moving duplicate references twice).
    moved_cache: dict[int, TensorizedDataset] = {}

    def move_pool(pool: TensorizedDataset) -> TensorizedDataset:
        k = id(pool)
        if k not in moved_cache:
            moved_cache[k] = maybe_move_train_pool(pool, device=device, dataset_device=str(dataset_device))
        return moved_cache[k]

    train_pool_phases = [(start, move_pool(pool)) for start, pool in train_pool_phases]

    model = ARCTransformer(
        vocab_size=VOCAB_SIZE,
        grid_size=grid_size,
        pos_encoding=str(pos_encoding),
        embed_dim=int(embed_dim),
        num_heads=int(num_heads),
        num_layers=int(num_layers),
        ff_dim=int(ff_dim),
        max_len=seq_len,
        dropout=float(dropout),
    ).to(device)

    if pretrained is not None:
        report = load_pretrained_weights(model, pretrained)
        print(
            "Loaded pretrained weights:"
            f" loaded={report.loaded}"
            f" skipped_unexpected={report.skipped_unexpected}"
            f" skipped_shape_mismatch={report.skipped_shape_mismatch}"
            f" missing_after_load={report.missing_after_load}",
            flush=True,
        )

    total_params, trainable_params = count_params(model)
    print(f"Model params: total={total_params:,} trainable={trainable_params:,}")

    # AdamW + fairly high weight decay. We avoid decaying biases and normalization weights.
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        name_l = name.lower()
        if p.ndim == 1 or name_l.endswith(".bias") or "norm" in name_l:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    opt = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": float(weight_decay)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(lr),
    )
    lr_decay = str(lr_decay).lower().strip()
    if lr_decay not in {"cosine", "none"}:
        raise ValueError(f"--lr_decay must be one of {{'cosine','none'}}, got {lr_decay!r}")
    if float(min_lr) < 0.0:
        raise ValueError(f"--min_lr must be >= 0, got {min_lr}")
    if lr_decay == "cosine":
        # Decay from --lr to --min_lr over the full training horizon.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=int(steps),
            eta_min=float(min_lr),
        )
    else:
        scheduler = None
    loss_fn = nn.CrossEntropyLoss()

    def save_latest_checkpoint(*, step: int) -> None:
        ckpt = {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "seed": int(seed),
            "grid_size": int(grid_size),
            "seq_len": int(seq_len),
            "train_skills": [int(s) for s in train_skills],
            "delay_until_by_skill": {int(k): int(v) for k, v in delay_until_by_skill.items()},
        }
        torch.save(ckpt, out_dir / "checkpoints" / "latest.pt")

    def save_best_val_checkpoint(*, step: int, val_score: float) -> None:
        ckpt = {
            "step": int(step),
            "val_score": float(val_score),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "seed": int(seed),
            "grid_size": int(grid_size),
            "seq_len": int(seq_len),
            "train_skills": [int(s) for s in train_skills],
            "delay_until_by_skill": {int(k): int(v) for k, v in delay_until_by_skill.items()},
        }
        torch.save(ckpt, out_dir / "checkpoints" / "best_val.pt")

    # Baseline training: mixed skills, only ID prompts (no OOD in train; notably no Skill 3 OOD).
    model.train()
    best_val = float("-inf")
    curves = LearningCurves(
        steps=[],
        loss=[],
        acc_train={},
        acc_id={},
        acc_ood={},
        probe_train_ood=[],
        probe_fully_heldout_ood=[],
    )
    gen_cpu = torch.Generator().manual_seed(int(seed))
    steps_iter = progress_iter(range(int(steps)), total=int(steps), desc="train", enabled=bool(progress))
    phase_idx = 0
    for step in steps_iter:
        while (phase_idx + 1) < len(train_pool_phases) and int(step) >= int(train_pool_phases[phase_idx + 1][0]):
            phase_idx += 1
        active_pool = train_pool_phases[phase_idx][1]
        batch = prepare_batch(
            batch_size=int(batch_size),
            train_pool=active_pool,
            device=device,
            cpu_generator=gen_cpu,
        )
        src = batch.src
        tgt = batch.tgt  # (B, grid_tokens)

        opt.zero_grad(set_to_none=True)
        logits = model(src)  # (B, T, V)
        pred_logits = logits[:, -(grid_tokens + 1) : -1, :]  # predict from test-x positions

        loss = loss_fn(pred_logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if (int(save_every) > 0) and ((step % int(save_every) == 0) or (step == int(steps) - 1)):
            save_latest_checkpoint(step=int(step))

        if (step % int(eval_every) == 0) or (step == int(steps) - 1):
            model.eval()
            eval_ids = sorted(eval_id_sets.keys())
            acc_train = {}
            acc_id = {}
            acc_ood = {}

            for sid in eval_ids:
                acc_train[sid] = evaluate_accuracy(
                    model=model,
                    rng=rng,
                    n_tasks=int(eval_tasks),
                    device=device,
                    grid_tokens=grid_tokens,
                    dataset=train_sets[sid],
                    eval_batch_size=int(eval_batch_size),
                )
                acc_id[sid] = evaluate_accuracy(
                    model=model,
                    rng=rng,
                    n_tasks=int(eval_tasks),
                    device=device,
                    grid_tokens=grid_tokens,
                    dataset=eval_id_sets[sid],
                    eval_batch_size=int(eval_batch_size),
                )
                acc_ood[sid] = evaluate_accuracy(
                    model=model,
                    rng=rng,
                    n_tasks=int(eval_tasks),
                    device=device,
                    grid_tokens=grid_tokens,
                    dataset=eval_ood_sets[sid],
                    eval_batch_size=int(eval_batch_size),
                )

            # Ensure the strict OOD probe is always visible.
            acc_probe_ood = evaluate_accuracy(
                model=model,
                rng=rng,
                n_tasks=int(eval_tasks),
                device=device,
                grid_tokens=grid_tokens,
                dataset=eval_probe_ood,
                eval_batch_size=int(eval_batch_size),
            )
            acc_probe_train = (
                evaluate_accuracy(
                    model=model,
                    rng=rng,
                    n_tasks=int(eval_tasks),
                    device=device,
                    grid_tokens=grid_tokens,
                    dataset=probe_ood_train,
                    eval_batch_size=int(eval_batch_size),
                )
                if probe_ood_train is not None and probe_ood_train.n > 0
                else 0.0
            )

            def fmt(acc: dict[int, float], skills: list[int]) -> str:
                return " ".join(f"s{sid}={acc[sid]:.3f}" for sid in skills)

            print(f"step={step:5d} loss={loss.item():.4f}")
            print(f"  trn: {fmt(acc_train, eval_ids)}")
            print(f"  id : {fmt(acc_id, eval_ids)}")
            print(
                f"  ood: {fmt(acc_ood, eval_ids)}  "
                f"(probe: s{probe_skill} train-ood={acc_probe_train:.3f} fully-heldout-ood={acc_probe_ood:.3f})"
            )
            if scheduler is not None:
                # All param groups share the same LR schedule (we only vary weight_decay).
                print(f"  lr : {opt.param_groups[0]['lr']:.6g}")

            # Track and plot learning curves
            curves.steps.append(int(step))
            curves.loss.append(float(loss.item()))
            curves.probe_train_ood.append(float(acc_probe_train))
            curves.probe_fully_heldout_ood.append(float(acc_probe_ood))
            for sid in eval_ids:
                curves.ensure_skill(sid)
                curves.acc_train[sid].append(float(acc_train[sid]))
                curves.acc_id[sid].append(float(acc_id[sid]))
                curves.acc_ood[sid].append(float(acc_ood[sid]))

            # Save metrics CSV next to the plot output (even if --no_plots).
            metrics_csv = out_dir / "plots" / "learning_curves_latest.csv"
            write_learning_curves_csv(curves=curves, skills=sorted(eval_ids), out_path=metrics_csv)

            # "Val" model selection: mean ID accuracy across eval_id splits.
            if len(eval_ids) > 0:
                val_score = float(sum(float(acc_id[sid]) for sid in eval_ids)) / float(len(eval_ids))
                if val_score > best_val:
                    best_val = val_score
                    save_best_val_checkpoint(step=int(step), val_score=float(val_score))

            if plots_enabled:
                delay_s = (
                    "none"
                    if len(delay_until_by_skill) == 0
                    else " ".join(f"s{sid}@{until}" for sid, until in sorted(delay_until_by_skill.items()))
                )
                title = (
                    "ARC skill learning curves (exact-match acc)\n"
                    f"train_skills={train_skills} | ood_in_train={sorted(train_with_ood)} | "
                    f"probe_skill={probe_skill} | cap_skill={cap_skill}:{cap_n} | "
                    f"delay_skills={delay_s} | eval_tasks={int(eval_tasks)}"
                )
                latest = out_dir / "plots" / "learning_curves_latest.png"
                plot_learning_curves(curves=curves, skills=sorted(eval_ids), out_path=latest, title=title)

            model.train()

    # Qualitative examples
    model.eval()
    if 2 in eval_id_sets:
        show_one_example(model=model, dataset=eval_id_sets[2], device=device, grid_size=grid_size)
    if 3 in eval_id_sets:
        show_one_example(model=model, dataset=eval_id_sets[3], device=device, grid_size=grid_size)
    show_one_example(model=model, dataset=eval_probe_ood, device=device, grid_size=grid_size)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a simple Transformer on synthetic ARC skills.")
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("tmp"),
        help="Directory produced by arc_dataset_generator.py (contains skill_<id>/{train,ood}.json).",
    )
    p.add_argument("--grid_size", type=int, default=5)
    p.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Optional path to pretrained weights (.pt checkpoint with `model` or a raw state_dict). "
        "Weights are loaded permissively (matching names+shapes); new layers stay randomly initialized.",
    )
    p.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=None,
        help="Alias for --train_skills (skill IDs to load for training/eval). Example: --tasks 14 15",
    )
    p.add_argument(
        "--train_skills",
        type=int,
        nargs="*",
        default=None,
        help=f"Skill IDs to load for training/eval. Default: {list(DEFAULT_TRAIN_SKILLS)}",
    )
    p.add_argument(
        "--phase1_skills",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional 2-phase curriculum. Phase 1 trains ONLY on these skills. "
            "Phase 2 adds skills (see --phase2_skills) at --phase2_start_step / --phase2_start_frac. "
            "Mutually exclusive with --delay_train_skills/--delay_train_until_steps."
        ),
    )
    p.add_argument(
        "--phase2_skills",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional 2-phase curriculum. Phase 2 skill set (joint training pool after curriculum switch). "
            "If omitted, defaults to phase1_skills (no-op)."
        ),
    )
    p.add_argument(
        "--phase2_start_step",
        type=int,
        default=None,
        help="Optional 2-phase curriculum: step at which to start Phase 2 (adding phase2-only skills).",
    )
    p.add_argument(
        "--phase2_start_frac",
        type=float,
        default=None,
        help=(
            "Optional 2-phase curriculum: fraction of total --steps at which to start Phase 2. "
            "Example: 0.5 means switch at half the steps. Ignored if --phase2_start_step is set."
        ),
    )
    p.add_argument(
        "--delay_train_skill",
        "--delay_train_skills",
        dest="delay_train_skills",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Exclude these skills from the mixed training pool initially (hard switch). "
            "Must be paired 1:1 with --delay_train_until_step/--delay_train_until_steps. "
            "Example: --delay_train_skills 13 14 --delay_train_until_steps 1000 5000"
        ),
    )
    p.add_argument(
        "--delay_train_until_step",
        "--delay_train_until_steps",
        dest="delay_train_until_steps",
        type=int,
        nargs="*",
        default=None,
        help=(
            "For each skill in --delay_train_skill/--delay_train_skills, the step at which that skill is introduced. "
            "Example: --delay_train_skills 13 14 --delay_train_until_steps 1000 5000"
        ),
    )
    p.add_argument(
        "--probe_skill",
        type=int,
        default=16,
        help=(
            "A *strict* OOD generalization probe: this skill's OOD split is never used for training, "
            "even if you mix OOD data into training for other skills via --train_with_ood_skills. "
            "Probe accuracy is reported on the held-out OOD test portion (controlled by --test_frac)."
        ),
    )
    p.add_argument(
        "--cap_train_skill3",
        type=int,
        default=None,
        help="If set, limits the number of Skill 3 training tasks (from skill_3/train.json) used in the mixed pool.",
    )
    p.add_argument(
        "--cap_train_skill",
        type=int,
        default=None,
        help="If set, limits the number of training tasks for this skill in the training pool (applies after any OOD mixing).",
    )
    p.add_argument(
        "--cap_train_n",
        type=int,
        default=None,
        help="Number of training tasks to keep for --cap_train_skill.",
    )
    p.add_argument(
        "--train_with_ood_skills",
        type=int,
        nargs="*",
        default=list(DEFAULT_TRAIN_WITH_OOD_SKILLS),
        help=(
            "Skills whose training pool should also include (a subset of) their OOD split. "
            "OOD examples are taken only from the OOD-train portion (OOD-test is always held out via --test_frac). "
            "The --probe_skill OOD split is excluded from this mixing."
        ),
    )
    p.add_argument(
        "--ood_train_frac",
        type=float,
        default=0.5,
        help=(
            "Fraction of the OOD-train portion to include in training when that skill is in --train_with_ood_skills "
            "(the remainder of OOD-train is unused; OOD-test remains held out via --test_frac)."
        ),
    )
    p.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help=(
            "Held-out test fraction for every loaded dataset split (both ID 'train' and OOD 'ood'). "
            "Reported accuracies (id/ood/probe) are computed on these held-out test portions."
        ),
    )
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument(
        "--lr_decay",
        type=str,
        default="cosine",
        choices=["cosine", "none"],
        help="LR schedule. Default is cosine decay over --steps; set 'none' for constant LR.",
    )
    p.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="Minimum LR for cosine decay (eta_min). Ignored when --lr_decay=none.",
    )
    p.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay (L2).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--pos_encoding",
        type=str,
        default="2d",
        choices=["2d", "1d"],
        help="Positional encoding scheme. '2d' (default) uses row+col embeddings per grid; '1d' uses the old absolute learned positions.",
    )
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="How often to update checkpoints/latest.pt (in steps). Set 0 to disable.",
    )
    p.add_argument("--eval_tasks", type=int, default=128)
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation (bigger is faster; uses more VRAM).",
    )
    p.add_argument("--progress", action="store_true", help="Show tqdm progress if installed")
    p.add_argument("--out_dir", type=Path, default=Path("arc_train_runs"), help="Where to write plots/metrics")
    p.add_argument("--no_plots", action="store_true", help="Disable saving learning-curve PNGs")
    p.add_argument(
        "--dataset_device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Where to keep the training pool tensors. 'gpu' avoids CPU bottlenecks; 'cpu' pins memory for async H2D.",
    )
    return p


def cli_main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.tasks is not None and args.train_skills is not None:
        raise ValueError("Use either --tasks or --train_skills (alias), not both.")

    # Resolve the "base" training skill list (before curriculum/delay tweaks).
    train_skills = args.tasks if args.tasks is not None else args.train_skills

    # Optional 2-phase curriculum, compiled down to the existing delay mechanism.
    if args.phase1_skills is not None:
        if (args.delay_train_skills is not None) or (args.delay_train_until_steps is not None):
            raise ValueError("--phase1_skills cannot be combined with --delay_train_skills/--delay_train_until_steps.")
        phase1 = [int(s) for s in args.phase1_skills]
        phase2 = phase1 if args.phase2_skills is None else [int(s) for s in args.phase2_skills]

        if args.phase2_start_step is not None:
            phase2_start_step = int(args.phase2_start_step)
        else:
            frac = 0.5 if args.phase2_start_frac is None else float(args.phase2_start_frac)
            if not (0.0 <= frac <= 1.0):
                raise ValueError(f"--phase2_start_frac must be in [0,1], got {frac}")
            phase2_start_step = int(round(frac * float(int(args.steps))))

        train_skills, delay_skills, delay_steps = _curriculum_delay_from_phases(
            phase1_skills=phase1,
            phase2_skills=phase2,
            phase2_start_step=int(phase2_start_step),
        )
        args.delay_train_skills = delay_skills
        args.delay_train_until_steps = delay_steps
    main(
        data_dir=Path(args.data_dir),
        grid_size=int(args.grid_size),
        pos_encoding=str(args.pos_encoding),
        pretrained=Path(args.pretrained) if args.pretrained is not None else None,
        train_skills=[int(s) for s in train_skills] if train_skills is not None else None,
        delay_train_skills=[int(s) for s in args.delay_train_skills] if args.delay_train_skills is not None else None,
        delay_train_until_steps=[int(s) for s in args.delay_train_until_steps] if args.delay_train_until_steps is not None else None,
        probe_skill=int(args.probe_skill),
        cap_train_skill3=int(args.cap_train_skill3) if args.cap_train_skill3 is not None else None,
        cap_train_skill=int(args.cap_train_skill) if args.cap_train_skill is not None else None,
        cap_train_n=int(args.cap_train_n) if args.cap_train_n is not None else None,
        train_with_ood_skills=[int(s) for s in args.train_with_ood_skills] if args.train_with_ood_skills is not None else None,
        ood_train_frac=float(args.ood_train_frac),
        test_frac=float(args.test_frac),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        lr_decay=str(args.lr_decay),
        min_lr=float(args.min_lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(args.device),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        eval_every=int(args.eval_every),
        save_every=int(args.save_every),
        eval_tasks=int(args.eval_tasks),
        eval_batch_size=int(args.eval_batch_size),
        progress=bool(args.progress),
        out_dir=Path(args.out_dir),
        no_plots=bool(args.no_plots),
        dataset_device=str(args.dataset_device),
    )


if __name__ == "__main__":
    cli_main()


