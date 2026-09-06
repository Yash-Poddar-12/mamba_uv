# Mamba-UV — training pipeline

Implements Methodology 3 from your document (Sections 3.1–3.3, 3.5, and the
classification-accuracy part of 3.7's eval protocol): the Degradation Map
Estimator, Physics-Ordered Patch Tokenizer with turbidity register tokens,
the Mamba/SSM encoder, Inverse-Degradation Weighted Pooling, and the two
novel auxiliary losses (physics consistency + scan-order invariance).

**Note on scope:** I didn't find an existing zipped project anywhere in
this conversation — only the architecture diagram image and the pasted
methodology text — so this is a fresh build, not an edit of something you
uploaded. If you do have an older zip from another session, tell me what's
in it and I'll merge the two.

**Deliberately deferred** (you asked to exclude edge-device code for now):
Section 3.4's DepGraph structured pruning, Section 3.6's Raspberry
Pi/thermal validation, and any quantization/ONNX/TFLite export. The
architecture itself already uses Ghost depthwise-separable convs in the
patch embedding (cheap and correct to include from the start), so the
PC-trained model stays close to the eventual edge target. Ablation flags
for Section 3.7's ablation table are exposed in the config (see below) so
you don't need separate scripts per row.

---

## 1. One-time setup

```bash
# from wherever you want the project to live
cd /path/to/your/workspace

# create and activate the venv (name: y_env, as requested)
python -m venv y_env

# Windows (PowerShell):
y_env\Scripts\Activate.ps1
# Windows (cmd.exe):
y_env\Scripts\activate.bat
# Linux / macOS:
source y_env/bin/activate

# install dependencies
pip install -r requirements.txt
```

Project layout after you unzip:

```
mamba-uv/
├── configs/config.yaml          <- all hyperparameters live here
├── src/
│   ├── data/                    <- AQUA20 loading, class-imbalance handling
│   ├── models/                  <- DME, tokenizer, SSM block, full model
│   ├── losses.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── .env.example                 <- only needed for the optional OpenAI step
```

Run every command below **from the `mamba-uv/` root**, with `y_env` active.

---

## 2. Dataset — no manual download step

`src/data/dataset.py` pulls `taufiktrf/AQUA20` straight from the
HuggingFace Hub the first time you run training, using the `datasets`
library — there's nothing to place by hand. By default it caches to your
global `~/.cache/huggingface` folder. If you'd rather it land inside your
project's `root/datasets` folder (as you described), set this **before**
running anything:

```bash
# Linux/macOS
export HF_HOME=./datasets
# Windows PowerShell
$env:HF_HOME = ".\datasets"
```

The dataset is ~200MB (8,171 images, 20 classes, official 6.56k/1.61k
train/test split) — the download happens once and is reused after that.

---

## 3. Installing `mamba-ssm` (optional but recommended)

The methodology's encoder is built on Mamba's **selective** SSM block.
`src/models/ssm_block.py` will:

1. try to import the official `mamba_ssm.Mamba` (CUDA kernel — fast), and
2. automatically fall back to a pure-PyTorch reimplementation of the same
   selective-scan recurrence if that import fails.

The fallback is **functionally correct but slow** (it loops sequentially
over the ~200-token sequence instead of using a fused CUDA kernel) — fine
for a first correctness check, not for real training runs.

`mamba-ssm` only builds on **Linux or WSL2** with a matching NVCC/CUDA
toolkit — it does not build natively on Windows. If your "PC with CUDA
cores" is Windows, either:
- run training inside WSL2 (recommended — gives you a real Linux CUDA
  environment), or
- train with the pure-PyTorch fallback (works today, no setup, just
  slower per step).

To install the real kernel (Linux/WSL2, matching CUDA toolkit + PyTorch
already installed):

```bash
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.2.0
```

If either pip install fails to build, it's almost always a CUDA/PyTorch
version mismatch — check `nvcc --version` against your installed PyTorch's
CUDA build (`python -c "import torch; print(torch.version.cuda)"`) and
match a `mamba-ssm` release that supports it. You don't need to touch any
of the model code either way — `configs/config.yaml`'s
`mamba_backend.prefer_mamba_ssm: true` already does the try/fallback for
you automatically.

---

## 4. Training

```bash
python -m src.train --config configs/config.yaml
```

This will:
- build the train/val/test split (`data.mode: "holdout"` by default — the
  official AQUA20 train/test split, with a stratified slice of `train`
  held out for validation; switch to `"kfold"` for the paper's 5-fold CV
  protocol, which merges train+test and reports mean±std across folds),
- print the parameter count (should land under the paper's 3.5M target —
  the default config comes in at **~0.98M**),
- print which SSM backend loaded,
- train with mixed precision, cosine LR schedule + warmup, gradient
  clipping, and early stopping on macro-F1,
- checkpoint the best epoch to `checkpoints/fold_0/best.pt`,
- log scalars to `runs/` (`tensorboard --logdir runs` to watch live).

## 5. Evaluation

```bash
python -m src.evaluate --config configs/config.yaml \
    --checkpoint checkpoints/fold_0/best.pt --split test
```

Prints precision/recall/F1 (macro) and Top-1/Top-3 accuracy, and saves a
confusion matrix PNG next to the checkpoint.

---

## 6. Class imbalance — what's implemented and why

AQUA20's 20 species are unevenly represented. Three complementary
mechanisms are wired in by default (toggle any of them in
`configs/config.yaml` under `imbalance:` / `augmentation:`):

1. **Inverse-frequency class weights** fed into the loss
   (`imbalance.use_class_weighted_loss`), combined with the paper's
   label-smoothed cross-entropy (`imbalance.loss_type: "ce_ls"`).
   `"focal"` is available as an alternative if you want to experiment —
   it's not the paper's default, but focal loss is a standard, suitable
   fit for heavy imbalance.
2. **`WeightedRandomSampler`** (`imbalance.use_weighted_sampler`) so
   minority-class images are drawn roughly as often as majority-class
   ones each epoch, rather than only reweighting the loss after the fact.
3. **Heavier augmentation specifically for minority-class samples**
   (`augmentation.minority_boost_threshold` and the `minority_*` keys) —
   wider rotation range, added vertical flip, stronger random erasing —
   applied only to samples from classes below a configurable fraction of
   the largest class's count.

These three stack: a minority-class image is both oversampled by the
sampler *and* gets the stronger augmentation transform, *and* still
contributes a higher-weighted loss term when it is drawn.

### Optional: OpenAI synthetic image generation (Sec 3.5's LLM+diffusion augmentation)

This is the one piece that needs an API key, and it's entirely optional —
skip it unless you specifically want the paper's synthetic-augmentation
strategy for the rarest classes.

```bash
pip install openai python-dotenv
cp .env.example .env
# open .env and paste your real key in place of the placeholder:
#   OPENAI_API_KEY=sk-...
python -m src.data.synthetic_augment_openai --images_per_class 40
```

What it does:
1. finds classes below `augmentation.minority_boost_threshold` of the
   largest class's count,
2. generates photorealistic underwater images of each species via
   `gpt-image-1`,
3. passes every generated image through the same Beer-Lambert synthetic
   underwater-formation model the DME is built on
   (`src/models/dme.py::beer_lambert_degrade`), with randomised
   depth/turbidity — so synthetic samples get physically plausible
   degradation instead of looking artificially clean,
4. saves the result to `synthetic_data/<species_name>/*.jpg`.

Then point the config at it and retrain:

```yaml
data:
  synthetic_data_dir: "synthetic_data"
```

`src/data/dataset.py` folds these in on the training split only (never
val/test), remapping folder names back onto AQUA20's label ids.

This step costs real money per generated image (OpenAI image-generation
API pricing) and needs internet access — check current pricing before
running it at scale, and adjust `--images_per_class` accordingly.
Note the OpenAI images API surface shifts occasionally; if
`client.images.generate(...)` throws, check
https://platform.openai.com/docs/guides/images and adjust
`_generate_one()` in that file — nothing else needs to change.

---

## 7. Ablation switches (Sec 3.7's ablation table)

Flip these in `configs/config.yaml -> model` / `losses` to reproduce each
row of the ablation study without touching code:

| Row                          | Config key(s)                                  |
|-------------------------------|-------------------------------------------------|
| Standard raster scan (baseline) | `model.use_physics_order: false`               |
| + physics-ordered scan        | `model.use_physics_order: true`                |
| + register tokens             | `model.use_register_tokens: true`              |
| + IDWP                        | `model.use_idwp: true` (else mean pooling)     |
| + physics consistency loss    | `losses.use_physics_consistency_loss: true`    |
| + scan invariance loss        | `losses.use_scan_invariance_loss: true`        |

Run `train.py` once per row, pointing `train.checkpoint_dir` /
`train.log_dir` at a distinct folder each time so results don't overwrite.

---

## 8. Design notes / assumptions I made

- **Images stay in `[0, 1]`** (no ImageNet mean/std normalization). The
  DME's channel-ratio and dark-channel calculations need physically
  meaningful pixel intensities — normalizing would corrupt that signal.
- **Physics-consistency decoder** (Sec 3.5) reconstructs the degradation
  score per *token* via a 1D conv over the physics-ordered sequence,
  rather than un-sorting back to a 2D grid first — the sequence order
  already matches its corresponding target values, so this is equivalent
  and simpler.
- **Turbidity register tokens** are implemented as two small learned
  `Linear(1, d_model)` layers applied to each image's mean/std DME score,
  rather than static learned vectors — this matches "initialised from the
  image-level mean and standard deviation" while still being trainable.
- **`data.mode: "holdout"`** is the default rather than `"kfold"`, since
  it's faster to iterate on and gives you a genuinely held-out test set;
  switch to `"kfold"` when you want the paper-comparable 5-fold numbers.
