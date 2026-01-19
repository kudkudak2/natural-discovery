from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Optional (but expected) deps for plotting/TSNE.
from sklearn.manifold import TSNE  # type: ignore[import-not-found]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

from arc_train_utils import progress  # noqa: E402


# Import training code as a module (safe: cli_main is guarded by __main__).
from arc_train_transformer_fewshot import (  # noqa: E402
    TrunkPlusExperts,
    VariantKey,
    load_variant_pools,
    prompt_seq_len,
    VOCAB_SIZE,
)


def _variant_keys_from_state_dict(state_dict: dict[str, torch.Tensor]) -> list[VariantKey]:
    """
    Extract VariantKey entries referenced by dynamically-created expert modules.

    - expert_mode="variant": weights live under "experts.<expert_key>...."
    - expert_mode="skill_adapter": adapters live under "adapters.<expert_key>...."
    """
    expert_keys: set[str] = set()
    for full in state_dict.keys():
        s = str(full)
        if s.startswith("experts."):
            # experts.<expert_key>.<param_name>
            rest = s.split("experts.", 1)[1]
            expert_keys.add(rest.split(".", 1)[0])
        elif s.startswith("adapters."):
            # adapters.<expert_key>.<layer_idx>.<param_name>
            rest = s.split("adapters.", 1)[1]
            expert_keys.add(rest.split(".", 1)[0])

    out: list[VariantKey] = []
    for ek in sorted(expert_keys):
        if not str(ek).startswith("s") or "__v" not in str(ek):
            raise ValueError(f"Unexpected expert_key format in checkpoint: {ek!r}")
        sid_s, var = str(ek).split("__v", 1)
        sid = int(sid_s[1:])
        out.append(VariantKey(skill_id=int(sid), variant=str(var)))
    return out


@torch.no_grad()
def trunk_embed_mean(*, model: TrunkPlusExperts, x: torch.Tensor) -> torch.Tensor:
    """
    Compute trunk-only embeddings for tokenized sequences x.
    Returns (B,D) embeddings by mean pooling over time.
    """
    _b, t = x.shape
    if t > int(model.global_pos_enc.shape[1]):
        raise ValueError(
            f"Sequence too long: t={t} > max_len={int(model.global_pos_enc.shape[1])}. Increase max_len."
        )

    emb = model.embed(x) + model.global_pos_enc[:, :t, :]
    if model.pos_encoding == "2d":
        block = model.grid_tokens + 1
        pos = torch.arange(t, device=x.device)
        within = pos % block
        is_sep = within == model.grid_tokens
        cell = torch.clamp(within, max=model.grid_tokens - 1)
        row = (cell // model.grid_size).to(torch.long)
        col = (cell % model.grid_size).to(torch.long)
        pos_emb_2d = model.row_embed(row) + model.col_embed(col)  # type: ignore[attr-defined]
        pos_emb_2d = pos_emb_2d.masked_fill(is_sep.unsqueeze(-1), 0.0)
        emb = emb + pos_emb_2d.unsqueeze(0)

    h = model.trunk(emb)  # (B,T,D)
    return h.mean(dim=1)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load trained model, embed with trunk, and plot t-SNE colored by skill id.")

    p.add_argument("--weights_path", type=Path, required=True, help="Path to weights_*.pt produced by training script.")
    p.add_argument(
        "--run_config",
        type=Path,
        default=None,
        help="Optional path to run_config.json written by training script. If omitted, auto-detect next to weights.",
    )
    p.add_argument("--data_dir", type=Path, default=None, help="Dataset dir. If omitted, will use run_config.json when available.")
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "ood"],
        help="Which on-disk dataset JSON to load (<data_dir>/skill_<id>/<split>.json).",
    )
    p.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help='Which deterministic subset to embed after loading the JSON. "test" is an alias for "val".',
    )
    p.add_argument("--grid_size", type=int, default=6)
    p.add_argument("--pos_encoding", type=str, default="2d", choices=["2d", "1d"])
    p.add_argument("--train_skills", type=int, nargs="*", default=None)

    # Model architecture (must match the checkpoint).
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument(
        "--num_heads",
        type=int,
        default=4,
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
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--trunk_layers", type=int, default=4)
    p.add_argument("--expert_layers", type=int, default=2)
    p.add_argument("--expert_mode", type=str, default="skill_adapter", choices=["skill_adapter", "variant"])
    p.add_argument("--adapter_dim", type=int, default=32)
    p.add_argument("--adapter_scale", type=float, default=1.0)

    # Sampling for embedding.
    p.add_argument(
        "--max_points",
        type=int,
        default=200,
        help="Global cap on total points embedded across the entire split. Set <=0 to disable.",
    )
    p.add_argument("--max_per_variant", type=int, default=64, help="Max examples per (skill,variant) to embed.")
    p.add_argument(
        "--progress_bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set (default), show a progress bar for embedding collection when tqdm is installed.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Fraction used to create the deterministic held-out subset returned by load_variant_pools.",
    )

    # t-SNE params.
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iter", type=int, default=1000)
    p.add_argument(
        "--tsne_verbose",
        type=int,
        default=1,
        help="sklearn TSNE verbosity (0=silent, 1=per-iteration progress, 2=more).",
    )

    p.add_argument("--out_dir", type=Path, default=Path("arc_tsne_runs"), help="Output directory for plots/CSV.")
    return p


def main(
    *,
    weights_path: Path,
    data_dir: Path,
    split: str,
    subset: str,
    grid_size: int,
    pos_encoding: str,
    train_skills: Optional[list[int]],
    embed_dim: int,
    num_heads: int,
    num_heads_trunk: Optional[int],
    num_heads_expert: Optional[int],
    ff_dim: int,
    dropout: float,
    trunk_layers: int,
    expert_layers: int,
    expert_mode: str,
    adapter_dim: int,
    adapter_scale: float,
    max_points: int,
    max_per_variant: int,
    progress_bar: bool,
    perplexity: float,
    tsne_iter: int,
    tsne_verbose: int,
    seed: int,
    device: str,
    val_frac: float,
    out_dir: Path,
) -> None:
    rng = np.random.default_rng(int(seed))
    device_s = str(device).strip().lower()
    if device_s in ("auto", ""):
        device_s = "cuda" if torch.cuda.is_available() else "cpu"
    if device_s.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU for embedding.")
        device_s = "cpu"
    device_t = torch.device(device_s)
    if device_t.type == "cuda":
        print(f"Embedding on GPU: {torch.cuda.get_device_name(device_t.index or 0)}")
    else:
        print("Embedding on CPU")

    skills = [14, 15, 16] if train_skills is None else [int(s) for s in train_skills]
    for sid in skills:
        if int(sid) < 1:
            raise ValueError(f"Invalid skill id: {sid}")

    # Load pools to get tokenized sequences.
    train_pools, val_pools = load_variant_pools(
        data_dir=Path(data_dir),
        skill_ids=skills,
        split=str(split),
        rng=rng,
        val_frac=float(val_frac),
    )
    subset_s = str(subset).strip().lower()
    if subset_s == "test":
        subset_s = "val"
    if subset_s == "train":
        pools = train_pools
    elif subset_s == "val":
        pools = val_pools
    else:
        raise ValueError(f"Unexpected --subset: {subset!r}")

    seq_len = prompt_seq_len(grid_size=int(grid_size), num_demos=3)
    trunk_heads = int(num_heads) if num_heads_trunk is None else int(num_heads_trunk)
    expert_heads = int(num_heads) if num_heads_expert is None else int(num_heads_expert)
    model = TrunkPlusExperts(
        grid_size=int(grid_size),
        max_len=int(seq_len),
        pos_encoding=str(pos_encoding),
        embed_dim=int(embed_dim),
        num_heads_trunk=int(trunk_heads),
        num_heads_expert=int(expert_heads),
        trunk_layers=int(trunk_layers),
        expert_layers=int(expert_layers),
        ff_dim=int(ff_dim),
        dropout=float(dropout),
        vocab_size=int(VOCAB_SIZE),
        expert_mode=str(expert_mode),
        adapter_dim=int(adapter_dim),
        adapter_scale=float(adapter_scale),
    ).to(device_t)

    ckpt = torch.load(Path(weights_path), map_location=device_t)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Unexpected checkpoint format at {weights_path}")
    state_dict = ckpt["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unexpected model_state_dict type in checkpoint: {type(state_dict)}")

    # Experts/adapters are created dynamically; pre-create the exact keys referenced by the checkpoint
    # so strict loading works (and so we don't accidentally create extra experts from the embedding split).
    for vk in _variant_keys_from_state_dict(state_dict):
        model.ensure_expert(vk)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    all_keys = sorted(pools.keys(), key=lambda kk: kk.to_str())

    # Collect embeddings.
    feats: list[np.ndarray] = []
    labels: list[int] = []
    variants: list[str] = []
    keys: list[str] = []

    max_n = max(1, int(max_per_variant))

    # Compute how many points we could embed if we took up to `max_per_variant` from every (skill,variant).
    # This lets `--max_points` behave like a true cap: if it's >= available, we embed everything (no sampling).
    total_available = 0
    for kk in all_keys:
        pool = pools[kk]
        n = int(pool.n)
        if n <= 0:
            continue
        total_available += min(max_n, n)

    remaining = int(max_points)
    cap_total = remaining > 0 and remaining < int(total_available)

    if not cap_total:
        # Full (uncapped) embedding: shuffle pool order to avoid sorted-key bias.
        shuffled_keys = list(all_keys)
        rng.shuffle(shuffled_keys)
        keys_iter = progress(shuffled_keys, total=len(shuffled_keys), desc="embed", enabled=bool(progress_bar))
        for kk in keys_iter:
            pool = pools[kk]
            n = int(pool.n)
            if n <= 0:
                continue
            take = min(max_n, n)
            if take <= 0:
                continue
            idx = rng.choice(n, size=take, replace=False)
            xb = pool.src[idx].to(device_t)
            z = trunk_embed_mean(model=model, x=xb).detach().cpu().numpy()
            feats.append(z)
            labels += [int(kk.skill_id)] * int(z.shape[0])
            variants += [str(kk.variant)] * int(z.shape[0])
            keys += [kk.to_str()] * int(z.shape[0])
    else:
        # Capped embedding: randomize across skills/variants, but keep the sample roughly balanced per skill.
        keys_by_skill: dict[int, list[VariantKey]] = {}
        for kk in all_keys:
            keys_by_skill.setdefault(int(kk.skill_id), []).append(kk)
        skill_ids = sorted(keys_by_skill.keys())
        if len(skill_ids) == 0:
            raise ValueError("No skills found in pools (unexpected).")

        # Shuffle skill order + per-skill variant order to make the sample random.
        rng.shuffle(skill_ids)
        for sid in skill_ids:
            rng.shuffle(keys_by_skill[sid])

        # Even per-skill budget, with remainder distributed randomly across skills.
        nskills = len(skill_ids)
        base = remaining // nskills
        rem = remaining % nskills
        quotas: dict[int, int] = {sid: int(base) for sid in skill_ids}
        if rem > 0:
            # Give +1 to the first `rem` skills in randomized order.
            for sid in skill_ids[:rem]:
                quotas[sid] += 1

        per_skill_pos: dict[int, int] = {sid: 0 for sid in skill_ids}
        counts_by_skill: dict[int, int] = {sid: 0 for sid in skill_ids}

        def _take_from_skill(sid: int, want: int) -> int:
            """Consume up to `want` points from this skill, moving forward through its shuffled variants."""
            if want <= 0:
                return 0
            taken_total = 0
            var_list = keys_by_skill[sid]
            pos = per_skill_pos[sid]
            while taken_total < want and pos < len(var_list):
                kk = var_list[pos]
                pos += 1
                pool = pools[kk]
                n = int(pool.n)
                if n <= 0:
                    continue
                # Take up to the per-variant cap (max_n). Do NOT artificially limit to a tiny "chunk",
                # otherwise we can end up massively under-filling `max_points` even when data exists.
                take = min(max_n, n, want - taken_total)
                if take <= 0:
                    continue
                idx = rng.choice(n, size=take, replace=False)
                xb = pool.src[idx].to(device_t)
                z = trunk_embed_mean(model=model, x=xb).detach().cpu().numpy()
                feats.append(z)
                labels.extend([int(kk.skill_id)] * int(z.shape[0]))
                variants.extend([str(kk.variant)] * int(z.shape[0]))
                keys.extend([kk.to_str()] * int(z.shape[0]))
                taken = int(z.shape[0])
                taken_total += taken
                counts_by_skill[sid] += taken
            per_skill_pos[sid] = pos
            return taken_total

        # Pass 1: hit per-skill quotas as evenly as possible.
        for sid in skill_ids:
            got = _take_from_skill(sid, int(quotas[sid]))
            remaining -= got

        # Pass 2: redistribute any shortfall (skills with fewer available points) across remaining skills.
        # Do this round-robin across shuffled skills to keep it fair/random-ish.
        while remaining > 0:
            progressed = 0
            for sid in skill_ids:
                if remaining <= 0:
                    break
                got = _take_from_skill(sid, min(remaining, max_n))
                if got > 0:
                    remaining -= got
                    progressed += got
            if progressed == 0:
                break

        total = sum(int(z.shape[0]) for z in feats)
        if total > 0:
            counts_str = ", ".join([f"s{sid}={counts_by_skill[sid]}" for sid in sorted(counts_by_skill.keys())])
            print(f"Sampled {total} points (balanced by skill where possible): {counts_str}")

    if len(feats) == 0:
        raise ValueError("No embeddings collected (no non-empty pools?).")

    X = np.concatenate(feats, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    v = np.asarray(variants, dtype=object)

    if X.shape[0] < 5:
        raise ValueError(f"Too few points for t-SNE: n={X.shape[0]}")

    # t-SNE
    eff_perp = float(perplexity)
    if eff_perp >= float(X.shape[0]):
        eff_perp = float(max(1.0, float(X.shape[0]) - 1.0))
        print(f"Perplexity too high for n={X.shape[0]}; using perplexity={eff_perp}")
    print(f"Running t-SNE: n={X.shape[0]} dim={X.shape[1]} perplexity={eff_perp} iters={int(tsne_iter)} verbose={int(tsne_verbose)}")
    tsne = TSNE(
        n_components=2,
        perplexity=float(eff_perp),
        n_iter=int(tsne_iter),
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
        verbose=int(tsne_verbose),
    )
    XY = tsne.fit_transform(X)

    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot: color by skill id.
    skill_ids = sorted({int(v) for v in y.tolist()})
    colors = {sid: i for i, sid in enumerate(skill_ids)}
    c = np.asarray([colors[int(s)] for s in y.tolist()], dtype=np.int64)
    cmap = plt.get_cmap("tab20", max(1, len(skill_ids)))
    rgba = cmap(c)

    def _variant_sort_key(s: str) -> tuple[int, int, str]:
        ss = str(s)
        # Numeric variants first (by int value) to keep marker assignments stable and intuitive.
        neg = ss.startswith("-")
        digits = ss[1:] if neg else ss
        if digits.isdigit():
            return (0, -int(digits) if neg else int(digits), "")
        return (1, 0, ss)

    # Assign markers by variant label; marker mapping is shared across skills.
    marker_cycle = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "H", "p", "8", "d"]
    uniq_variants = sorted({str(x) for x in v.tolist()}, key=_variant_sort_key)
    variant_marker = {vv: marker_cycle[i % len(marker_cycle)] for i, vv in enumerate(uniq_variants)}

    plt.figure(figsize=(10, 8))
    # Plot one layer per variant to allow per-point markers (while keeping skill-based colors).
    for vv in uniq_variants:
        mask = v == vv
        if not np.any(mask):
            continue
        plt.scatter(
            XY[mask, 0],
            XY[mask, 1],
            s=10,
            color=rgba[mask],
            alpha=0.85,
            linewidths=0.0,
            marker=variant_marker[vv],
        )
    plt.title(f"t-SNE of trunk embeddings (color=skill, marker=variant) | split={split} | n={X.shape[0]}")
    # Build legend from skill ids (discrete).
    handles = []
    for sid in skill_ids:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=mcolors.to_hex(cmap(colors[sid])),
                markeredgewidth=0.0,
                label=f"s{sid}",
            )
        )
    plt.legend(handles=handles, title="skill", loc="best", frameon=False, fontsize=9)
    plt.tight_layout()
    out_png = plots_dir / "tsne_trunk_by_skill.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    # Save coords to CSV for downstream analysis.
    out_csv = plots_dir / "tsne_trunk_by_skill.csv"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("x,y,skill_id,key\n")
        for i in range(XY.shape[0]):
            f.write(f"{float(XY[i,0])},{float(XY[i,1])},{int(y[i])},{keys[i]}\n")

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_csv}")


def cli_main(argv: Optional[list[str]] = None) -> None:
    if argv is None:
        import sys

        argv = sys.argv[1:]
    # Track which flags were explicitly provided, so run_config can act as defaults only.
    explicit: set[str] = set()
    for a in argv:
        s = str(a)
        if s.startswith("--"):
            explicit.add(s.split("=", 1)[0])

    args = _build_arg_parser().parse_args(argv)

    # Load run_config.json if present.
    run_cfg_path: Optional[Path] = Path(args.run_config) if args.run_config is not None else None
    if run_cfg_path is None:
        # Training saves weights in out_dir/plots/*.pt; run_config is out_dir/plots/run_config.json
        run_cfg_path = Path(args.weights_path).parent / "run_config.json"
    run_cfg: dict[str, object] = {}
    if run_cfg_path.exists():
        run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))

    def pick(name: str, current, flag: str):
        # If user passed the flag explicitly, keep CLI value; otherwise prefer run_config if present.
        if flag in explicit:
            return current
        if name in run_cfg and run_cfg[name] is not None:
            return run_cfg[name]
        return current

    data_dir = pick("data_dir", args.data_dir, "--data_dir")
    if data_dir is None:
        raise ValueError("Missing --data_dir and no run_config.json found (or it lacked data_dir).")

    picked_num_heads = int(pick("num_heads", args.num_heads, "--num_heads"))
    picked_num_heads_trunk_raw = pick("num_heads_trunk", args.num_heads_trunk, "--num_heads_trunk")
    picked_num_heads_expert_raw = pick("num_heads_expert", args.num_heads_expert, "--num_heads_expert")
    picked_num_heads_trunk = int(picked_num_heads_trunk_raw) if picked_num_heads_trunk_raw is not None else None
    picked_num_heads_expert = int(picked_num_heads_expert_raw) if picked_num_heads_expert_raw is not None else None

    main(
        weights_path=Path(args.weights_path),
        data_dir=Path(str(data_dir)),
        split=str(pick("split", args.split, "--split")),
        subset=str(pick("subset", args.subset, "--subset")),
        grid_size=int(pick("grid_size", args.grid_size, "--grid_size")),
        pos_encoding=str(pick("pos_encoding", args.pos_encoding, "--pos_encoding")),
        train_skills=(
            [int(s) for s in args.train_skills]
            if "--train_skills" in explicit
            else ([int(s) for s in (run_cfg.get("train_skills") or [])] if run_cfg.get("train_skills") is not None else None)
        ),
        embed_dim=int(pick("embed_dim", args.embed_dim, "--embed_dim")),
        num_heads=int(picked_num_heads),
        num_heads_trunk=picked_num_heads_trunk,
        num_heads_expert=picked_num_heads_expert,
        ff_dim=int(pick("ff_dim", args.ff_dim, "--ff_dim")),
        dropout=float(pick("dropout", args.dropout, "--dropout")),
        trunk_layers=int(pick("trunk_layers", args.trunk_layers, "--trunk_layers")),
        expert_layers=int(pick("expert_layers", args.expert_layers, "--expert_layers")),
        expert_mode=str(pick("expert_mode", args.expert_mode, "--expert_mode")),
        adapter_dim=int(pick("adapter_dim", args.adapter_dim, "--adapter_dim")),
        adapter_scale=float(pick("adapter_scale", args.adapter_scale, "--adapter_scale")),
        max_points=int(args.max_points),
        max_per_variant=int(args.max_per_variant),
        progress_bar=bool(args.progress_bar),
        perplexity=float(args.perplexity),
        tsne_iter=int(args.tsne_iter),
        tsne_verbose=int(args.tsne_verbose),
        seed=int(args.seed),
        device=str(args.device),
        val_frac=float(args.val_frac),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    cli_main()


