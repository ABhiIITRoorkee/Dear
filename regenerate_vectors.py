# regenerate_vectors.py
# Field-Weighted + Task-Aware vector regeneration for APIs, mashups, and tag vocabularies.
#
# Key upgrades (novel + useful):
# 1) Task-aware embedding (e.g., Jina v3 retrieval.query vs retrieval.passage)
# 2) Field-weighted fusion (title/desc/tags/url for APIs; title/desc/categories/url for mashups)
# 3) Optional Matryoshka truncation (e.g., 768 dims) by slicing first N dims
# 4) Optional loading of existing tag mappings from dataset/fold_1 to keep IDs stable
#
# Example (Jina v3, best-practice for retrieval):
#   python regenerate_vectors.py \
#     --raw_root "$RAW_ROOT" \
#     --out_root "$OUT_ROOT" \
#     --model_name "jinaai/jina-embeddings-v3" \
#     --mashup_task "retrieval.query" \
#     --api_task "retrieval.passage" \
#     --normalize_embeddings 1 \
#     --truncate_dim 768 \
#     --w_api_title 0.6 --w_api_desc 1.0 --w_api_tags 0.4 --w_api_url 0.2 \
#     --w_m_title 0.6  --w_m_desc 1.0  --w_m_cats 0.4  --w_m_url 0.2 \
#     --use_existing_tag_mapping 1 \
#     --batch_size 32

from typing import Optional


import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# Text cleanup helpers
# -----------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")

def normalize_space(text: str) -> str:
    if text is None:
        return ""
    return " ".join(str(text).replace("\n", " ").replace("\r", " ").split()).strip()

def clean_text(text: str, strip_urls: bool, strip_html: bool) -> str:
    text = normalize_space(text)
    if strip_html:
        text = _HTML_RE.sub(" ", text)
        text = normalize_space(text)
    if strip_urls:
        text = _URL_RE.sub(" ", text)
        text = normalize_space(text)
    return text

def url_slug_tokens(url: str) -> str:
    """
    '/api/30-boxes' -> '30 boxes'
    'http://abc.com/foo-bar' -> 'foo bar'
    """
    url = (url or "").strip()
    if not url:
        return ""
    # keep last path part if possible
    last = url.split("/")[-1]
    last = last.replace("-", " ").replace("_", " ")
    last = re.sub(r"[^A-Za-z0-9 ]+", " ", last)
    return normalize_space(last)


# -----------------------------
# Field builders
# -----------------------------
def split_csv_like(s: str) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def build_api_fields(api_obj: dict, strip_urls: bool, strip_html: bool) -> dict:
    details = api_obj.get("details", {}) or {}

    title = clean_text(details.get("title", ""), strip_urls, strip_html)
    desc  = clean_text(details.get("description", ""), strip_urls, strip_html)
    tags  = ", ".join(split_csv_like(clean_text(details.get("tags", ""), strip_urls, strip_html)))
    slug  = url_slug_tokens(api_obj.get("url", ""))

    return {
        "title": title,
        "desc": desc,
        "tags": tags,
        "url": slug,
    }

def build_mashup_fields(m_obj: dict, strip_urls: bool, strip_html: bool) -> dict:
    title = clean_text(m_obj.get("title", ""), strip_urls, strip_html)
    desc  = clean_text(m_obj.get("description", ""), strip_urls, strip_html)
    cats  = ", ".join(split_csv_like(clean_text(m_obj.get("categories", ""), strip_urls, strip_html)))
    slug  = url_slug_tokens(m_obj.get("url", ""))

    return {
        "title": title,
        "desc": desc,
        "cats": cats,
        "url": slug,
    }


# -----------------------------
# Model encoding wrapper (task-aware + robust)
# -----------------------------
def encode_texts(model, texts, batch_size, normalize_embeddings, task=None):
    """
    Some models (e.g., Jina v3 with remote code) support `task=...`.
    We try task first; if not supported, fall back to standard encode.
    """
    kwargs = dict(
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    if task is not None and str(task).strip():
        try:
            return model.encode(texts, task=task, **kwargs)
        except TypeError:
            # Older sentence-transformers / models might not accept task
            return model.encode(texts, **kwargs)
    return model.encode(texts, **kwargs)

def fuse_weighted_vectors(vecs_and_weights, normalize_after=True):
    """
    vecs_and_weights: list of (vecs: np.ndarray [B,D], w: float)
    """
    out = None
    total_w = 0.0
    for v, w in vecs_and_weights:
        w = float(w)
        if w <= 0:
            continue
        total_w += w
        out = v * w if out is None else (out + v * w)

    if out is None:
        # fallback: all weights were 0 -> return zeros
        b, d = vecs_and_weights[0][0].shape
        out = np.zeros((b, d), dtype=np.float32)
        total_w = 1.0

    out = out / max(total_w, 1e-12)

    if normalize_after:
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        out = out / norms
    return out.astype(np.float32)

def maybe_truncate(vecs: np.ndarray, truncate_dim: int) -> np.ndarray:
    if truncate_dim is None or truncate_dim <= 0:
        return vecs
    d = vecs.shape[1]
    if truncate_dim >= d:
        return vecs
    return vecs[:, :truncate_dim].copy()


# -----------------------------
# Tag mapping (stable IDs)
# -----------------------------
def find_mapping_file(fold_dir: Path, candidates: list[str]) -> Optional[Path]:

    for name in candidates:
        p = fold_dir / name
        if p.exists():
            return p
    return None

def load_id_to_tag_from_csv(path: Path) -> dict[int, str]:
    id_to_tag = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # handles "id, tag" with optional spaces
            try:
                tid = int(str(row[0]).strip())
                tag = str(row[1]).strip() if len(row) > 1 else ""
                if tag:
                    id_to_tag[tid] = tag
            except Exception:
                continue
    return id_to_tag

def extract_api_tag_vocab_from_raw(apis: list[dict]) -> dict[int, str]:
    all_tags = set()
    for api in apis:
        tags_str = (api.get("details", {}) or {}).get("tags", "") or ""
        for t in split_csv_like(tags_str):
            all_tags.add(t)
    tag_to_id = {tag: i for i, tag in enumerate(sorted(all_tags))}
    return {i: tag for tag, i in tag_to_id.items()}

def extract_mashup_tag_vocab_from_raw(mashups: list[dict]) -> dict[int, str]:
    all_tags = set()
    for m in mashups:
        cats = m.get("categories", "") or ""
        for c in split_csv_like(cats):
            all_tags.add(c)
    tag_to_id = {tag: i for i, tag in enumerate(sorted(all_tags))}
    return {i: tag for tag, i in tag_to_id.items()}


# -----------------------------
# H5 writer (same format as yours)
# -----------------------------
def write_h5_by_id(out_file: Path, ids: list[int], vecs: np.ndarray) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, "w") as f:
        for _id, v in zip(ids, vecs):
            f.create_dataset(str(_id), data=v.astype(np.float32))


def main():
    parser = argparse.ArgumentParser("Regenerate vectors with task-aware, field-weighted fusion.")
    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--normalize_embeddings", type=int, default=1)
    parser.add_argument("--use_e5_prefix", type=int, default=0)

    # NEW: task adapters (useful for Jina v3)
    parser.add_argument("--mashup_task", type=str, default="")
    parser.add_argument("--api_task", type=str, default="")
    parser.add_argument("--tag_task", type=str, default="")

    # NEW: field weights
    parser.add_argument("--w_api_title", type=float, default=0.6)
    parser.add_argument("--w_api_desc",  type=float, default=1.0)
    parser.add_argument("--w_api_tags",  type=float, default=0.4)
    parser.add_argument("--w_api_url",   type=float, default=0.2)

    parser.add_argument("--w_m_title", type=float, default=0.6)
    parser.add_argument("--w_m_desc",  type=float, default=1.0)
    parser.add_argument("--w_m_cats",  type=float, default=0.4)
    parser.add_argument("--w_m_url",   type=float, default=0.2)

    # NEW: cleanup knobs
    parser.add_argument("--strip_urls", type=int, default=1)
    parser.add_argument("--strip_html", type=int, default=1)

    # NEW: matryoshka truncation (slice first N dims)
    parser.add_argument("--truncate_dim", type=int, default=0)

    # NEW: keep tag IDs stable by loading existing mapping files
    parser.add_argument("--use_existing_tag_mapping", type=int, default=1)
    parser.add_argument("--tag_mapping_fold", type=str, default="fold_1")

    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    data_dir = out_root / "data"

    api_path = raw_root / "api.json"
    mashup_path = raw_root / "mashups.json"

    if not api_path.exists():
        raise FileNotFoundError(f"Missing: {api_path}")
    if not mashup_path.exists():
        raise FileNotFoundError(f"Missing: {mashup_path}")

    with open(api_path, "r", encoding="utf-8") as f:
        apis = json.load(f)
    with open(mashup_path, "r", encoding="utf-8") as f:
        mashups = json.load(f)

    model = SentenceTransformer(args.model_name, trust_remote_code=True)

    normalize = bool(args.normalize_embeddings)
    strip_urls = bool(args.strip_urls)
    strip_html = bool(args.strip_html)
    truncate_dim = int(args.truncate_dim) if int(args.truncate_dim) > 0 else 0

    # -----------------------------
    # Tags: prefer existing mapping for stable IDs
    # -----------------------------
    api_id_to_tag = None
    mash_id_to_tag = None

    if args.use_existing_tag_mapping:
        fold_dir = out_root / "dataset" / args.tag_mapping_fold
        api_map = find_mapping_file(fold_dir, ["Api_tag_mapping.csv", "api_tag_mapping.csv"])
        mash_map = find_mapping_file(fold_dir, ["mashup_tag_mapping.csv"])

        if api_map and mash_map:
            api_id_to_tag = load_id_to_tag_from_csv(api_map)
            mash_id_to_tag = load_id_to_tag_from_csv(mash_map)

    if api_id_to_tag is None:
        api_id_to_tag = extract_api_tag_vocab_from_raw(apis)
    if mash_id_to_tag is None:
        mash_id_to_tag = extract_mashup_tag_vocab_from_raw(mashups)

    # -----------------------------
    # Encode tag vocab vectors
    # -----------------------------
    api_tag_ids = sorted(api_id_to_tag.keys())
    api_tag_texts = [normalize_space(api_id_to_tag[i]) for i in api_tag_ids]
    mash_tag_ids = sorted(mash_id_to_tag.keys())
    mash_tag_texts = [normalize_space(mash_id_to_tag[i]) for i in mash_tag_ids]

    tag_task = args.tag_task.strip() if args.tag_task else None
    api_tag_vecs = encode_texts(model, api_tag_texts, args.batch_size, normalize, task=tag_task)
    mash_tag_vecs = encode_texts(model, mash_tag_texts, args.batch_size, normalize, task=tag_task)

    api_tag_vecs = maybe_truncate(api_tag_vecs, truncate_dim)
    mash_tag_vecs = maybe_truncate(mash_tag_vecs, truncate_dim)

    write_h5_by_id(data_dir / "api_tag_vector.h5", api_tag_ids, api_tag_vecs)
    write_h5_by_id(data_dir / "mashup_tag_vector.h5", mash_tag_ids, mash_tag_vecs)

    # -----------------------------
    # Encode API vectors (field-weighted)
    # -----------------------------
    api_ids = [int(a["id"]) for a in apis]
    api_fields = [build_api_fields(a, strip_urls, strip_html) for a in apis]

    api_title = [f["title"] for f in api_fields]
    api_desc  = [f["desc"]  for f in api_fields]
    api_tags  = [f["tags"]  for f in api_fields]
    api_url   = [f["url"]   for f in api_fields]

    if args.use_e5_prefix:
        api_title = ["passage: " + t for t in api_title]
        api_desc  = ["passage: " + t for t in api_desc]
        api_tags  = ["passage: " + t for t in api_tags]
        api_url   = ["passage: " + t for t in api_url]

    api_task = args.api_task.strip() if args.api_task else None

    v_title = encode_texts(model, api_title, args.batch_size, normalize, task=api_task)
    v_desc  = encode_texts(model, api_desc,  args.batch_size, normalize, task=api_task)
    v_tags  = encode_texts(model, api_tags,  args.batch_size, normalize, task=api_task)
    v_url   = encode_texts(model, api_url,   args.batch_size, normalize, task=api_task)

    fused_api = fuse_weighted_vectors(
        [
            (v_title, args.w_api_title),
            (v_desc,  args.w_api_desc),
            (v_tags,  args.w_api_tags),
            (v_url,   args.w_api_url),
        ],
        normalize_after=normalize,
    )
    fused_api = maybe_truncate(fused_api, truncate_dim)
    write_h5_by_id(data_dir / "API_vectors.h5", api_ids, fused_api)

    # -----------------------------
    # Encode Mashup vectors (field-weighted)
    # -----------------------------
    mash_ids = [int(m["id"]) for m in mashups]
    mash_fields = [build_mashup_fields(m, strip_urls, strip_html) for m in mashups]

    m_title = [f["title"] for f in mash_fields]
    m_desc  = [f["desc"]  for f in mash_fields]
    m_cats  = [f["cats"]  for f in mash_fields]
    m_url   = [f["url"]   for f in mash_fields]

    if args.use_e5_prefix:
        m_title = ["query: " + t for t in m_title]
        m_desc  = ["query: " + t for t in m_desc]
        m_cats  = ["query: " + t for t in m_cats]
        m_url   = ["query: " + t for t in m_url]

    mash_task = args.mashup_task.strip() if args.mashup_task else None

    v_title = encode_texts(model, m_title, args.batch_size, normalize, task=mash_task)
    v_desc  = encode_texts(model, m_desc,  args.batch_size, normalize, task=mash_task)
    v_cats  = encode_texts(model, m_cats,  args.batch_size, normalize, task=mash_task)
    v_url   = encode_texts(model, m_url,   args.batch_size, normalize, task=mash_task)

    fused_mash = fuse_weighted_vectors(
        [
            (v_title, args.w_m_title),
            (v_desc,  args.w_m_desc),
            (v_cats,  args.w_m_cats),
            (v_url,   args.w_m_url),
        ],
        normalize_after=normalize,
    )
    fused_mash = maybe_truncate(fused_mash, truncate_dim)
    write_h5_by_id(data_dir / "vectors.h5", mash_ids, fused_mash)

    # -----------------------------
    # Write a small metadata file for reproducibility
    # -----------------------------
    meta = {
        "model_name": args.model_name,
        "normalize_embeddings": int(normalize),
        "truncate_dim": int(truncate_dim),
        "use_e5_prefix": int(args.use_e5_prefix),
        "tasks": {"mashup_task": mash_task, "api_task": api_task, "tag_task": tag_task},
        "weights": {
            "api": {"title": args.w_api_title, "desc": args.w_api_desc, "tags": args.w_api_tags, "url": args.w_api_url},
            "mashup": {"title": args.w_m_title, "desc": args.w_m_desc, "cats": args.w_m_cats, "url": args.w_m_url},
        },
        "cleanup": {"strip_urls": int(strip_urls), "strip_html": int(strip_html)},
        "tag_mapping": {"use_existing": int(args.use_existing_tag_mapping), "fold": args.tag_mapping_fold},
    }
    (data_dir / "vector_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Wrote vectors to: {data_dir}")
    print(f"     Model: {args.model_name}")
    print(f"     Tasks: mashup={mash_task} api={api_task} tag={tag_task}")
    print(f"     truncate_dim={truncate_dim} normalize={int(normalize)}")
    print(f"     Meta: {data_dir / 'vector_meta.json'}")


if __name__ == "__main__":
    main()
