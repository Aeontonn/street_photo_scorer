"""Extract CLIP image embeddings for all scraped photos."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32


def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor, device


def extract_embeddings(posts: list[dict]) -> tuple[np.ndarray, list[dict]]:
    model, processor, device = load_clip()

    valid_posts = [p for p in posts if Path(p.get("local_path", "")).exists()]
    embeddings = []
    kept_posts = []

    for i in tqdm(range(0, len(valid_posts), BATCH_SIZE), desc="Extracting embeddings"):
        batch = valid_posts[i : i + BATCH_SIZE]

        images, good = [], []
        for p in batch:
            try:
                images.append(Image.open(p["local_path"]).convert("RGB"))
                good.append(p)
            except Exception:
                pass  # skip corrupt/unreadable files

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
        kept_posts.extend(good)

    valid_posts = kept_posts
    return np.vstack(embeddings), valid_posts


def run():
    with open(DATA_DIR / "raw/posts.json") as f:
        posts = json.load(f)

    embeddings, valid_posts = extract_embeddings(posts)

    np.save(PROCESSED_DIR / "embeddings.npy", embeddings)
    with open(PROCESSED_DIR / "posts_filtered.json", "w") as f:
        json.dump(valid_posts, f, indent=2)

    print(f"Saved embeddings {embeddings.shape} for {len(valid_posts)} posts")


if __name__ == "__main__":
    run()
