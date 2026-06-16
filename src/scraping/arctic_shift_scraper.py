"""
Scrape r/streetphotography via Arctic Shift (https://arctic-shift.photon-reddit.com).
No Reddit account or API key needed.

Handles both direct image posts and Reddit gallery posts (reddit.com/gallery/...).
For galleries, only the first image is used but inherits the post's score.
"""

import json
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from tqdm import tqdm

SUBREDDIT = "streetphotography"
BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _is_direct_image(url: str) -> bool:
    return url.split("?")[0].lower().endswith(IMAGE_EXTENSIONS)


def _gallery_image_urls(post: dict) -> list[str]:
    """
    Extract the first valid image URL from a gallery post.
    Uses i.redd.it/{media_id}.jpg — permanent URLs that don't expire.
    preview.redd.it URLs have signed tokens that expire quickly and return 404.
    """
    metadata = post.get("media_metadata") or {}
    gallery = post.get("gallery_data") or {}
    items = gallery.get("items", [])

    for item in items:
        if item.get("is_deleted"):
            continue
        mid = item.get("media_id", "")
        meta = metadata.get(mid, {})
        if meta.get("status") != "valid" or meta.get("e") != "Image":
            continue
        mime = meta.get("m", "image/jpg")
        ext = mime.split("/")[-1].replace("jpeg", "jpg")
        return [f"https://i.redd.it/{mid}.{ext}"]

    return []


def fetch_posts(total: int = 2000, batch_size: int = 100) -> list[dict]:
    """
    Fetch posts from Arctic Shift in batches, paginating backwards by timestamp.
    Starts from 2024-01-01 and goes back to 2021-01-01 — older posts have
    stable images that haven't been deleted from Reddit's CDN.
    """
    posts = []
    # Start: 2024-01-01 00:00 UTC
    before_utc = 1704067200

    with tqdm(total=total, desc="Fetching posts") as pbar:
        while len(posts) < total:
            params = f"subreddit={SUBREDDIT}&limit={batch_size}&sort=desc&before={before_utc}"

            req = Request(
                f"{BASE_URL}?{params}",
                headers={"User-Agent": "street_photo_scorer/1.0"},
            )
            try:
                with urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
            except (URLError, HTTPError) as e:
                print(f"\nRequest error: {e}. Retrying in 10s...")
                time.sleep(10)
                continue

            batch = data.get("data", [])
            if not batch:
                print("\nNo more posts available.")
                break

            for post in batch:
                raw_url = post.get("url", "")
                base = {
                    "id": post["id"],
                    "title": post.get("title", ""),
                    "score": post.get("score", 0),
                    "upvote_ratio": post.get("upvote_ratio", 0.0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc", 0),
                    "author": post.get("author", ""),
                }

                if _is_direct_image(raw_url):
                    posts.append({**base, "url": raw_url, "source": "direct"})

                elif "reddit.com/gallery" in raw_url:
                    gallery_urls = _gallery_image_urls(post)
                    if gallery_urls:
                        posts.append({**base, "url": gallery_urls[0], "source": "gallery"})

            before_utc = int(batch[-1].get("created_utc", 0))
            pbar.update(len(batch))
            time.sleep(0.5)

    return posts[:total]


def download_images(posts: list[dict], img_dir: Path | None = None) -> list[dict]:
    if img_dir is None:
        img_dir = DATA_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    valid = []
    for post in tqdm(posts, desc="Downloading images"):
        url = post["url"]
        ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"
        dest = img_dir / f"{post['id']}.{ext}"

        if not dest.exists():
            try:
                req = Request(url, headers={"User-Agent": "street_photo_scorer/1.0"})
                with urlopen(req, timeout=8) as resp:
                    dest.write_bytes(resp.read())
                time.sleep(0.05)
            except Exception as e:
                print(f"  skip {post['id']}: {e}")
                continue

        post["local_path"] = str(dest)
        valid.append(post)

    return valid


def run(total: int = 2000):
    posts = fetch_posts(total=total)
    print(f"\nFound {len(posts)} image posts. Downloading...")
    posts = download_images(posts)

    out = DATA_DIR / "posts.json"
    with open(out, "w") as f:
        json.dump(posts, f, indent=2)
    print(f"Saved {len(posts)} posts → {out}")


if __name__ == "__main__":
    run()
