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


DEFAULT_TRAIN_SKILLS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
DEFAULT_TRAIN_WITH_OOD_SKILLS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

DEFAULT_TRAIN_SKILLS = (11, 12, 14, 15, 16)
DEFAULT_TRAIN_WITH_OOD_SKILLS = (11, 12, 14, 15, 16)
class ARCTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        max_len: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
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
        b, t = x.shape
        emb = self.embed(x) + self.pos_enc[:, :t, :]
        h = self.transformer(emb)
        return self.fc_out(h)  # (B, T, vocab)


def main(
    data_dir: Path = Path("tmp"),
    grid_size: int = 5,
    train_skills: Optional[list[int]] = None,
    delay_train_skill: Optional[int] = None,
    delay_train_until_step: int = 0,
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
    weight_decay: float = 0.01, # 0.1 seems too high actually
    seed: int = 0,
    device: str = "cuda", # if torch.cuda.is_available() else "cpu",
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    ff_dim: int = 256,
    dropout: float = 0.0,
    eval_every: int = 500,
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
    out_dir.mkdir(parents=True, exist_ok=True)
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

    if delay_train_until_step < 0:
        raise ValueError(f"delay_train_until_step must be >= 0, got {delay_train_until_step}")
    if delay_train_skill is not None:
        delay_train_skill = int(delay_train_skill)
        if delay_train_skill not in train_skills:
            raise ValueError(f"delay_train_skill={delay_train_skill} must be included in train_skills")

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

    # Build mixed training pools. Optionally delay one skill until a given step.
    train_src_all = torch.cat([train_sets[sid].src for sid in train_skills], dim=0)
    train_tgt_all = torch.cat([train_sets[sid].tgt for sid in train_skills], dim=0)
    train_pool_all = TensorizedDataset(
        skill_id=-1, split="train_mix", grid_size=grid_size, src=train_src_all, tgt=train_tgt_all
    )

    train_pool_pre = train_pool_all
    if delay_train_skill is not None and int(delay_train_until_step) > 0:
        pre_skills = [sid for sid in train_skills if sid != int(delay_train_skill)]
        if len(pre_skills) == 0:
            raise ValueError("Delaying the only training skill would leave an empty training pool.")
        train_src_pre = torch.cat([train_sets[sid].src for sid in pre_skills], dim=0)
        train_tgt_pre = torch.cat([train_sets[sid].tgt for sid in pre_skills], dim=0)
        train_pool_pre = TensorizedDataset(
            skill_id=-1,
            split=f"train_mix_no_s{int(delay_train_skill)}_until_{int(delay_train_until_step)}",
            grid_size=grid_size,
            src=train_src_pre,
            tgt=train_tgt_pre,
        )

    # Ensure training pools won't bottleneck on CPU.
    train_pool_all = maybe_move_train_pool(train_pool_all, device=device, dataset_device=str(dataset_device))
    train_pool_pre = maybe_move_train_pool(train_pool_pre, device=device, dataset_device=str(dataset_device))

    model = ARCTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=int(embed_dim),
        num_heads=int(num_heads),
        num_layers=int(num_layers),
        ff_dim=int(ff_dim),
        max_len=seq_len,
        dropout=float(dropout),
    ).to(device)

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
    loss_fn = nn.CrossEntropyLoss()

    # Baseline training: mixed skills, only ID prompts (no OOD in train; notably no Skill 3 OOD).
    model.train()
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
    for step in steps_iter:
        active_pool = train_pool_pre if (delay_train_skill is not None and step < int(delay_train_until_step)) else train_pool_all
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

            if plots_enabled:
                title = (
                    "ARC skill learning curves (exact-match acc)\n"
                    f"train_skills={train_skills} | ood_in_train={sorted(train_with_ood)} | "
                    f"probe_skill={probe_skill} | cap_skill={cap_skill}:{cap_n} | "
                    f"delay_skill={delay_train_skill}@{int(delay_train_until_step)} | eval_tasks={int(eval_tasks)}"
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
    p.add_argument("--train_skills", type=int, nargs="*", default=list(DEFAULT_TRAIN_SKILLS))
    p.add_argument(
        "--delay_train_skill",
        type=int,
        default=None,
        help="Exclude this skill from the mixed training pool until --delay_train_until_step (hard switch).",
    )
    p.add_argument(
        "--delay_train_until_step",
        type=int,
        default=0,
        help="Step at which --delay_train_skill is introduced into training.",
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
    p.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay (L2).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--eval_every", type=int, default=500)
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
    main(
        data_dir=Path(args.data_dir),
        grid_size=int(args.grid_size),
        train_skills=[int(s) for s in args.train_skills] if args.train_skills is not None else None,
        delay_train_skill=int(args.delay_train_skill) if args.delay_train_skill is not None else None,
        delay_train_until_step=int(args.delay_train_until_step),
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
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(args.device),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        eval_every=int(args.eval_every),
        eval_tasks=int(args.eval_tasks),
        eval_batch_size=int(args.eval_batch_size),
        progress=bool(args.progress),
        out_dir=Path(args.out_dir),
        no_plots=bool(args.no_plots),
        dataset_device=str(args.dataset_device),
    )


if __name__ == "__main__":
    cli_main()


