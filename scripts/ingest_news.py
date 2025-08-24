import os, json, argparse
from datetime import datetime, timezone
import feedparser
import requests
from dotenv import load_dotenv

load_dotenv()

FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.cnn.com/rss/edition.rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
]

NEWSAPI = "https://newsapi.org/v2/top-headlines?language=en&pageSize=100"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def utcnow_iso():
    return datetime.now(timezone.utc).isoformat()

def fetch_rss():
    items = []
    for url in FEEDS:
        fp = feedparser.parse(url)
        for e in fp.entries:
            items.append({
                "source": fp.feed.get("title"),
                "title": e.get("title"),
                "link": e.get("link"),
                "published": e.get("published", utcnow_iso()),
                "ingested_at": utcnow_iso(),
                "raw_html": None
            })
    return items

def fetch_newsapi():
    if not NEWSAPI_KEY:
        return []
    r = requests.get(NEWSAPI, headers={"X-Api-Key": NEWSAPI_KEY}, timeout=30)
    r.raise_for_status()
    data = r.json()
    items = []
    for a in data.get("articles", []):
        items.append({
                "source": a.get("source", {}).get("name"),
                "title": a.get("title"),
                "link": a.get("url"),
                "published": a.get("publishedAt"),
                "ingested_at": utcnow_iso(),
                "raw_html": None
        })
    return items

def main(out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    items = fetch_rss() + fetch_newsapi()
    with open(out_path, "a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} records to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/articles.jsonl")
    args = ap.parse_args()
    main(args.out)
