### Quick run: generate data + train Transformer (+ external ARC datasets)

This folder contains:
- **Synthetic ARC-style dataset generation** (`arc_dataset_generator.py`)
- **Transformer trainer** (`arc_train_transformer.py`) for skill learning + OOD generalization
- Optional **external dataset loading** (ARC-AGI / ConceptARC-style layouts)

The training script is built around a fixed prompt layout and self-attention, so memory scales roughly like \(T^2\).
To prevent OOMs from rare very-long sequences, the trainer supports `--max_seq_len` (default **500**), which drops examples that would exceed the cap and prints drop statistics.

---

### 0) Where to run commands from

All commands below assume:

```bash
cd my_research/natural_discovery
```

---

### 1) Data formats (synthetic vs external)

`arc_train_transformer.py` supports two dataset styles:

#### Synthetic (skill-based) datasets

Folder layout:
- `<data_dir>/skill_<id>/train.json`
- `<data_dir>/skill_<id>/ood.json`
- optional `<data_dir>/skill_<id>/png/{train,ood}/*.png`

#### External ARC datasets (ARC-AGI / ConceptARC-like)

Folder layout auto-detection:
- **ARC-AGI style**: `<data_dir>/{training,evaluation}/*.json`
- **Generic external**: `<data_dir>/<any_subdir>/*.json` (all jsons under immediate subdirs are loaded, then split deterministically into train/evaluation)

Notes:
- You can keep `--grid_size=0 --num_demos=0` to infer from the external dataset (recommended).
- When `--max_seq_len>0`, the loader may effectively cap usable grid sizes; tasks that don’t fit are skipped and reported.

---

### 2) Quick run (newest): 7puzzle harder-OOD experiment

This is the newest “quick run” sequence.

#### 2.1 Generate the dataset

What it does:
- Writes `7puzzle_harderood/skill_<id>/{train,ood}.json` for the selected skills.
- Uses **per-skill `--n_tasks`** to allocate more tasks to key skills and fewer to others.

```bash
python arc_dataset_generator.py --out_dir=7puzzle_harderood --skills 14 15 16 24 25 26 27 29 --n_tasks 300 300 60 300 300 60 30 300 --n_jobs=15
```

#### 2.2 Train baseline (all skills)

What it does:
- Trains one model on a **mixed pool** containing all listed skills.
- Uses a fixed **token budget** implied by `--grid_size`/`--num_demos` and enforces `--max_seq_len` (default 500).
- Prints dataset filtering stats like:
  - `[max_seq_len=500] filtered tasks ... dropped=... kept=... final(... seq_len=...)`

```bash
CUDA_VISIBLE_DEVICES=0 python arc_train_transformer.py   --out_dir=2026_01_31_7puzzle_harderood_baseline   --data_dir=7puzzle_harderood   --grid_size=6   --steps=300000   --weight_decay=0.2  --lr_decay none --num_layers=10 --ff_dim=100 --train_skills 14 15 16 24 25 26 27 29
```

#### 2.3 “Pretrain” run (subset of skills)

What it does:
- Trains on a **subset** (e.g., to learn core patterns first), then you can compare learning curves / performance.

```bash
CUDA_VISIBLE_DEVICES=0 python arc_train_transformer.py   --out_dir=2026_01_31_7puzzle_harderood_pretrain   --data_dir=7puzzle_harderood   --grid_size=6   --steps=300000   --weight_decay=0.2  --lr_decay none --num_layers=10 --ff_dim=100 --train_skills 14 15 24 25 29
```

#### 2.4 Compare runs

```bash
python task_comparison_plot.py --out task_comparison_2026_01_33.png 2026_01_31_7puzzle_harderood_baseline 2026_01_31_7puzzle_harderood_pretrain
```

---

### 3) Training on ARC-AGI1 / ARC-AGI2 / ConceptARC

You don’t need `arc_dataset_generator.py` for these; just point `--data_dir` at the dataset root.

#### 3.1 ARC-AGI1

Assumes:
- `<ARC_AGI1_ROOT>/training/*.json`
- `<ARC_AGI1_ROOT>/evaluation/*.json`

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python arc_train_transformer.py \
  --out_dir=2026_02_01_arc_agi1 \
  --data_dir <ARC_AGI1_ROOT> \
  --grid_size 0 \
  --num_demos 0 \
  --max_seq_len 500 \
  --steps 100000
```

#### 3.2 ARC-AGI2

Same layout; just swap the root:

```bash
CUDA_VISIBLE_DEVICES=0 python arc_train_transformer.py \
  --out_dir=2026_02_01_arc_agi2 \
  --data_dir <ARC_AGI2_ROOT> \
  --grid_size 0 \
  --num_demos 0 \
  --max_seq_len 500 \
  --steps 100000
```

#### 3.3 ConceptARC (or other “external” sets)

If your dataset is not exactly `training/` + `evaluation/`, but has jsons under subfolders, the loader will treat it as a “generic external” dataset and split deterministically:

```bash
CUDA_VISIBLE_DEVICES=0 python arc_train_transformer.py \
  --out_dir=2026_02_01_conceptarc \
  --data_dir <CONCEPTARC_ROOT> \
  --grid_size 0 \
  --num_demos 0 \
  --max_seq_len 500 \
  --steps 100000
```

---

### 4) Useful knobs (recommended)

- **`--max_seq_len`** (default **500**): drops any examples whose prompt would exceed this token length (prevents attention OOM).
  - Set `--max_seq_len 0` to disable, but be careful (OOM risk).
- **`--print_solved_n`** (default **0**): prints up to N solved ID test examples at each eval (stdout).
- **`--plot_unsolved_n`**: saves “latest unsolved” PNGs during eval under `plots/unsolved_examples/`.
- **`--plot_solved_n`**: saves “latest solved” PNGs during eval under `plots/solved_examples/`.
- **`--plot_augmented_n`**: saves “latest augmented” PNGs during eval under `plots/augmented_examples/` (uses the same distribution as train-time `--aug_*`).

---

### Outputs (what you should see)

Training writes to `--out_dir`:
- `checkpoints/latest.pt` and (optionally) `checkpoints/best_val.pt`
- `plots/learning_curves_latest.png` (unless `--no_plots`)
- Console logs including:
  - per-split accuracies (ID / OOD / probe)
  - `max_seq_len` filtering statistics (dropped vs kept)
  - optional solved-example prints (if `--print_solved_n > 0`)
