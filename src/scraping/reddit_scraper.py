"""Scrape r/streetphotography posts with image URLs and engagement metrics."""

import os
import time
import json
from pathlib import Path
from urllib.request import urlretrieve

import praw
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

SUBREDDIT = "streetphotography"
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "street_photo_scorer/1.0"),
    )


def fetch_posts(limit: int = 1000, sort: str = "top", time_filter: str = "all") -> list[dict]:
    """Fetch posts from r/streetphotography and return metadata list."""
    reddit = get_reddit_client()
    sub = reddit.subreddit(SUBREDDIT)
    getter = getattr(sub, sort)

    posts = []
    kwargs = {"limit": limit}
    if sort == "top":
        kwargs["time_filter"] = time_filter

    for post in tqdm(getter(**kwargs), total=limit, desc="Fetching posts"):
        if not post.url.endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        posts.append({
            "id": post.id,
            "title": post.title,
            "score": post.score,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "url": post.url,
            "created_utc": post.created_utc,
            "author": str(post.author),
        })

    return posts


def download_images(posts: list[dict], img_dir: Path | None = None) -> list[dict]:
    """Download images and add local path to each post dict."""
    if img_dir is None:
        img_dir = DATA_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    valid = []
    for post in tqdm(posts, desc="Downloading images"):
        ext = post["url"].rsplit(".", 1)[-1].split("?")[0]
        dest = img_dir / f"{post['id']}.{ext}"
        if not dest.exists():
            try:
                urlretrieve(post["url"], dest)
                time.sleep(0.1)
            except Exception as e:
                print(f"  skip {post['id']}: {e}")
                continue
        post["local_path"] = str(dest)
        valid.append(post)

    return valid


def run(limit: int = 2000):
    posts = fetch_posts(limit=limit)
    posts = download_images(posts)

    out = DATA_DIR / "posts.json"
    with open(out, "w") as f:
        json.dump(posts, f, indent=2)
    print(f"Saved {len(posts)} posts → {out}")


if __name__ == "__main__":
    run()
