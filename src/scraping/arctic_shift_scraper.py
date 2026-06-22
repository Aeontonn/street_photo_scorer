"""
Scrape r/streetphotography via Arctic Shift (https://arctic-shift.photon-reddit.com).
No Reddit account or API key needed.

Handles both direct image posts and Reddit gallery posts (reddit.com/gallery/...).
Resumes from the oldest existing post so re-runs extend rather than restart.
"""

import argparse
import json
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from tqdm import tqdm

BASE_URL  = "https://arctic-shift.photon-reddit.com/api/posts/search"
DATA_DIR  = Path("data/raw")
IMAGE_DIR = DATA_DIR / "images"
POSTS_FILE = DATA_DIR / "posts.json"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _is_direct_image(url: str) -> bool:
    return url.split("?")[0].lower().endswith(IMAGE_EXTENSIONS)


def _gallery_image_urls(post: dict) -> list[str]:
    metadata = post.get("media_metadata") or {}
    gallery  = post.get("gallery_data") or {}
    items    = gallery.get("items", [])

    for item in items:
        if item.get("is_deleted"):
            continue
        mid  = item.get("media_id", "")
        meta = metadata.get(mid, {})
        if meta.get("status") != "valid" or meta.get("e") != "Image":
            continue
        mime = meta.get("m", "image/jpg")
        ext  = mime.split("/")[-1].replace("jpeg", "jpg")
        return [f"https://i.redd.it/{mid}.{ext}"]

    return []


def _load_existing() -> tuple[list[dict], int]:
    """Load existing posts.json and return (posts, oldest_utc)."""
    if not POSTS_FILE.exists():
        return [], 1704067200  # default: start at 2024-01-01

    with open(POSTS_FILE) as f:
        posts = json.load(f)

    if not posts:
        return [], 1704067200

    oldest_utc = min(int(p.get("created_utc", 0)) for p in posts)
    print(f"Resuming from {len(posts)} existing posts. Oldest: {oldest_utc}")
    return posts, oldest_utc


def _request_with_retry(url: str, max_retries: int = 6) -> dict:
    """GET url with exponential backoff on any network error."""
    wait = 5
    for attempt in range(max_retries):
        try:
            req = Request(url, headers={"User-Agent": "street_photo_scorer/1.0"})
            with urlopen(req, timeout=45) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"\n  Network error ({e}). Retry {attempt+1}/{max_retries} in {wait}s...")
            time.sleep(wait)
            wait = min(wait * 2, 120)
    return {}


def fetch_posts(subreddit: str, need: int, start_before_utc: int,
                checkpoint_file: Path | None = None) -> list[dict]:
    """
    Fetch `need` image posts going backwards from start_before_utc.
    Saves a checkpoint after every 500 posts so a crash loses at most one batch.
    """
    # Resume from checkpoint if it exists
    posts: list[dict] = []
    before_utc = start_before_utc
    if checkpoint_file and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            saved = json.load(f)
        posts = saved.get("posts", [])
        before_utc = saved.get("before_utc", start_before_utc)
        print(f"  Checkpoint found: resuming with {len(posts)} fetched posts, "
              f"before_utc={before_utc}")

    batch_size = 100
    consecutive_empty = 0
    last_checkpoint = len(posts)

    with tqdm(total=need, initial=len(posts), desc=f"Fetching r/{subreddit}") as pbar:
        while len(posts) < need:
            params = (
                f"subreddit={subreddit}"
                f"&limit={batch_size}"
                f"&sort=desc"
                f"&before={before_utc}"
            )
            data = _request_with_retry(f"{BASE_URL}?{params}")

            batch = data.get("data", [])
            if not batch:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    print("\nNo more posts available in archive.")
                    break
                time.sleep(5)
                continue
            consecutive_empty = 0

            added = 0
            for post in batch:
                raw_url = post.get("url", "")
                base = {
                    "id":           post["id"],
                    "title":        post.get("title", ""),
                    "score":        post.get("score", 0),
                    "upvote_ratio": post.get("upvote_ratio", 0.0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc":  post.get("created_utc", 0),
                    "author":       post.get("author", ""),
                    "subreddit":    subreddit,
                }
                if _is_direct_image(raw_url):
                    posts.append({**base, "url": raw_url, "source": "direct"})
                    added += 1
                elif "reddit.com/gallery" in raw_url:
                    urls = _gallery_image_urls(post)
                    if urls:
                        posts.append({**base, "url": urls[0], "source": "gallery"})
                        added += 1

            before_utc = int(batch[-1].get("created_utc", 0))
            pbar.update(added)
            time.sleep(0.3)

            # Save checkpoint every 500 new posts
            if checkpoint_file and len(posts) - last_checkpoint >= 500:
                with open(checkpoint_file, "w") as f:
                    json.dump({"posts": posts, "before_utc": before_utc}, f)
                last_checkpoint = len(posts)

    # Clean up checkpoint on success
    if checkpoint_file and checkpoint_file.exists():
        checkpoint_file.unlink()

    return posts[:need]


def download_images(posts: list[dict]) -> list[dict]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    valid = []
    for post in tqdm(posts, desc="Downloading images"):
        url = post["url"]
        ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"
        dest = IMAGE_DIR / f"{post['id']}.{ext}"

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


def run(total: int = 15000, subreddit: str = "streetphotography"):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing, oldest_utc = _load_existing()
    existing_ids = {p["id"] for p in existing}

    need = total - len(existing)
    if need <= 0:
        print(f"Already have {len(existing)} posts — nothing to do.")
        return

    print(f"Need {need} more posts (have {len(existing)}, target {total})")
    checkpoint = DATA_DIR / f"fetch_checkpoint_{subreddit}.json"
    new_posts = fetch_posts(subreddit=subreddit, need=need * 2,
                            start_before_utc=oldest_utc, checkpoint_file=checkpoint)

    # Deduplicate
    new_posts = [p for p in new_posts if p["id"] not in existing_ids]
    print(f"\nFetched {len(new_posts)} new unique posts. Downloading images...")

    new_posts = download_images(new_posts)

    # Re-add local_path to existing posts (in case images moved)
    for p in existing:
        if "local_path" not in p:
            dest = IMAGE_DIR / f"{p['id']}.jpg"
            if dest.exists():
                p["local_path"] = str(dest)

    all_posts = existing + new_posts
    with open(POSTS_FILE, "w") as f:
        json.dump(all_posts, f, indent=2)

    print(f"\nDone. Total saved: {len(all_posts)} posts → {POSTS_FILE}")
    print(f"New: {len(new_posts)}  |  Previously had: {len(existing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total",     type=int, default=15000,
                        help="Target total number of posts (default: 15000)")
    parser.add_argument("--subreddit", type=str, default="streetphotography",
                        help="Subreddit to scrape (default: streetphotography)")
    args = parser.parse_args()
    run(total=args.total, subreddit=args.subreddit)
