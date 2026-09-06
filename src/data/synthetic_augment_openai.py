"""
OPTIONAL data-augmentation script. Not required to train Mamba-UV -- the
default pipeline already handles class imbalance via class-weighted loss +
a WeightedRandomSampler + heavier augmentation for minority classes
(see src/data/class_balance.py). Run this only if you specifically want
the paper's "LLM-prompted diffusion-generated synthetic images" strategy
(Sec 3.5) for the most severely underrepresented species.

Setup:
    1. pip install openai python-dotenv
    2. cp .env.example .env, then paste your OpenAI key into .env
    3. python -m src.data.synthetic_augment_openai --images_per_class 40

Output layout (matches configs/config.yaml -> data.synthetic_data_dir):
    synthetic_data/<class_name>/*.jpg      <- degraded, training-ready
    synthetic_data_raw/<class_name>/*.jpg  <- clean generations, kept for inspection

After this finishes, set data.synthetic_data_dir: "synthetic_data" in
configs/config.yaml and re-run training -- src/data/dataset.py will fold
these images into the training set automatically (never into val/test).

Note: the OpenAI image-generation API surface changes from time to time.
If `client.images.generate(...)` errors, check
https://platform.openai.com/docs/guides/images for the current signature
and adjust the `_generate_one` function below -- everything else in this
file (prompt templates, Beer-Lambert pass, saving) is unaffected.
"""
import argparse
import base64
import io
import os

import numpy as np
import torch
import yaml
from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv

from src.data.class_balance import compute_class_counts
from src.models.dme import beer_lambert_degrade

PROMPT_TEMPLATE = (
    "A realistic underwater photograph of a {species}, natural ocean habitat, "
    "shot on a scientific dive camera, slightly hazy water, natural light rays, "
    "documentary marine biology style, no text or watermark"
)


def _generate_one(client, species_name: str, size: str = "1024x1024") -> Image.Image:
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=PROMPT_TEMPLATE.format(species=species_name),
        size=size,
        n=1,
    )
    img_bytes = base64.b64decode(resp.data[0].b64_json)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--images_per_class", type=int, default=40,
                         help="Synthetic images to generate per minority class")
    parser.add_argument("--out_dir", type=str, default="synthetic_data")
    parser.add_argument("--raw_dir", type=str, default="synthetic_data_raw")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not found. Copy .env.example to .env and paste your key in, "
            "or `export OPENAI_API_KEY=...` before running this script."
        )
    from openai import OpenAI
    client = OpenAI()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ds = load_dataset(cfg["data"]["hf_dataset_id"])
    class_names = ds["train"].features["label"].names
    labels = ds["train"]["label"] + ds["test"]["label"]
    counts = compute_class_counts(labels, cfg["data"]["num_classes"])
    max_count = counts.max()
    threshold = cfg["augmentation"]["minority_boost_threshold"]

    minority_classes = [
        class_names[i] for i, c in enumerate(counts) if c < threshold * max_count
    ]
    print(f"Minority classes (< {threshold:.0%} of max count = {max_count}): {minority_classes}")

    for species in minority_classes:
        raw_dir = os.path.join(args.raw_dir, species)
        deg_dir = os.path.join(args.out_dir, species)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(deg_dir, exist_ok=True)

        for i in range(args.images_per_class):
            try:
                img = _generate_one(client, species)
            except Exception as e:
                print(f"  [warn] generation failed for {species} #{i}: {e}")
                continue

            img.save(os.path.join(raw_dir, f"{species}_{i:03d}.jpg"), quality=95)

            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

            depth = 2.0 + 4.0 * torch.rand(1).item()        # 2-6m, randomised per image
            turbidity = 0.2 + 0.6 * torch.rand(1).item()     # 0.2-0.8
            degraded = beer_lambert_degrade(tensor, depth=depth, turbidity=turbidity)

            degraded_img = Image.fromarray(
                (degraded.permute(1, 2, 0).numpy() * 255).astype("uint8")
            )
            degraded_img.save(os.path.join(deg_dir, f"{species}_{i:03d}.jpg"), quality=95)

        print(f"  {species}: {args.images_per_class} synthetic images generated + degraded")

    print(f"\nDone. Set data.synthetic_data_dir: \"{args.out_dir}\" in {args.config} to use them.")


if __name__ == "__main__":
    main()
