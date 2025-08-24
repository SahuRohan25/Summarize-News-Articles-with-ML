import os, json, argparse
from tqdm import tqdm
from src.preprocessing import clean_html, normalize_whitespace
from scripts.dedupe_utils import fingerprint, is_duplicate

def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main(raw_path: str, out_path: str, min_len=400, max_len=4000):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen = set()
    with open(out_path, "w", encoding="utf-8") as out:
        for rec in tqdm(iter_jsonl(raw_path)):
            title = normalize_whitespace(rec.get("title"))
            body = clean_html(rec.get("raw_html", ""))
            if not body:
                continue
            if len(body) < min_len or len(body) > max_len:
                continue
            fp = fingerprint(body)
            if is_duplicate(fp, seen):
                continue
            seen.add(fp)
            first_sent = body.split(".")[:2]
            pseudo = normalize_whitespace((title or "") + ". " + ". ".join(first_sent))
            out.write(json.dumps({"article": body, "summary": pseudo}, ensure_ascii=False) + "\n")
    print(f"Built dataset → {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/articles.jsonl")
    ap.add_argument("--out", default="data/datasets/train_weak.jsonl")
    args = ap.parse_args()
    main(args.raw, args.out)
