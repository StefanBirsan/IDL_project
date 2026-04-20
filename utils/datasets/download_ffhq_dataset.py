from pathlib import Path
import random

import numpy as np
from PIL import Image
from datasets import load_dataset

OUT_DIR = Path("datasets/data/faces")
OUT_DIR.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "marcosv/ffhq-dataset",
    split="train",
    streaming=True,
)

# Buffer-based shuffle for streaming mode
dataset = dataset.shuffle(seed=42)

for idx, sample in enumerate(dataset.take(30)):
    image = sample["image"]

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = Image.fromarray(np.asarray(image).astype(np.uint8), mode="RGB")

    out_path = OUT_DIR / f"face_{idx:03d}.png"
    image.save(out_path)

    print(f"Saved {out_path}")