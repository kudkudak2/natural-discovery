### Quick run: synthetic ARC data generation + transformer training

This folder contains a **small synthetic ARC-style benchmark** and a minimal **Transformer trainer**.
The goal is to study **skill learning + OOD generalization**: train on in-distribution (“easy”) instances and evaluate on harder **OOD** instances that follow the *same underlying rule*.
---

### 0) Where to run commands from

All commands below assume you are in this directory:

```bash
cd my_research/natural_discovery
```

---

### 1) Generate data on disk (JSON + a few PNG previews)

The generator writes a folder that looks like:

- `<out_dir>/skill_<id>/train.json`
- `<out_dir>/skill_<id>/ood.json`
- `<out_dir>/skill_<id>/png/{train,ood}/*.png` (small visual samples)

For the 6×6, 400-tasks-per-skill dataset referenced by the training commands below:

```bash
python arc_dataset_generator.py \
  --out_dir 6x6_400 \
  --grid_size 6 \
  --n_tasks 400 \
  --skills 11 12 14 15 16 \
  --png_per_skill 4 \
  --seed 0
```

Notes:
- **`train.json` vs `ood.json`**: both contain the same task format (3 demos + 1 test); `ood` is generated with harder settings.
- **PNG rendering** requires `matplotlib`. If you don’t have it, install it (e.g. `pip install matplotlib`), or reduce friction by generating fewer PNGs via `--png_per_skill 0`.
- The generator also creates `*.meta.json` sidecars and (by default) a `6x6_400.zip` archive alongside the folder.

---

### 2) Train the Transformer (AdamW + high weight decay)

`arc_train_transformer.py` trains a small Transformer encoder to predict the test output grid tokens given the prompt.

Important details:
- The script loads `<data_dir>/skill_<id>/{train,ood}.json`.
- It then does an internal deterministic split using `--test_frac` to produce held-out evaluation sets (so printed metrics are on held-out portions).
- Optimizer is **AdamW** with **`--weight_decay` defaulting to `0.1`**.

#### Baseline run

```bash
CUDA_VISIBLE_DEVICES=0 python arc_train_transformer.py \
  --out_dir=2025_12_26_6x6_400_baseline_wd \
  --data_dir=6x6_400 \
  --grid_size=6 \
  --steps=100000
```

#### “Wait” run (delay introducing skill 16 until step 60k)

```bash
CUDA_VISIBLE_DEVICES=0 python arc_train_transformer.py \
  --out_dir=2025_12_25_6x6_400_wait_wd \
  --data_dir=6x6_400 \
  --grid_size=6 \
  --steps=100000 \
  --delay_train_skill=16 \
  --delay_train_until_step=60000
```

---

### Outputs

Training writes to `--out_dir`, including:
- `plots/learning_curves_latest.png` (unless `--no_plots` is set)
- printed accuracy metrics for ID / OOD splits and the strict OOD probe skill


