# regenerate_vectors.py
import json
import os
import h5py
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def normalize_space(x: str) -> str:
    return " ".join((x or "").replace("\n", " ").split()).strip()

def build_api_text(api_obj):
    title = normalize_space(api_obj.get("details", {}).get("title", ""))
    desc  = normalize_space(api_obj.get("details", {}).get("description", ""))
    tags_str = normalize_space(api_obj.get("details", {}).get("tags", ""))
    # tags_str is comma-separated already
    parts = []
    if title: parts.append(title)
    if desc:  parts.append(desc)
    if tags_str: parts.append("Tags: " + tags_str)
    return "\n".join(parts).strip()

def build_mashup_text(m_obj):
    title = normalize_space(m_obj.get("title", ""))
    desc  = normalize_space(m_obj.get("description", ""))
    cats  = normalize_space(m_obj.get("categories", ""))
    parts = []
    if title: parts.append(title)
    if desc:  parts.append(desc)
    if cats:  parts.append("Categories: " + cats)
    return "\n".join(parts).strip()

def save_vectors_h5(filename: Path, ids, texts, model: SentenceTransformer, batch_size=128):
    filename.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filename, "w") as f:
        for i in range(0, len(texts), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_txt = texts[i:i+batch_size]
            vecs = model.encode(
                batch_txt,
                batch_size=min(batch_size, len(batch_txt)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ).astype(np.float32)
            for _id, v in zip(batch_ids, vecs):
                f.create_dataset(str(_id), data=v)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, required=True,
                        help="Path to 'Original Dataset' folder containing api.json and mashups.json")
    parser.add_argument("--out_root", type=str, required=True,
                        help="Output root folder containing 'data/' (and optionally 'dataset/')")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    data_dir = out_root / "data"

    with open(raw_root / "api.json", "r", encoding="utf-8") as f:
        apis = json.load(f)
    with open(raw_root / "mashups.json", "r", encoding="utf-8") as f:
        mashups = json.load(f)

    # Tag vocab (from raw json)
    api_id_to_tags = {}
    all_api_tags = set()
    for api in apis:
        api_id = api["id"]
        tags_str = api.get("details", {}).get("tags", "") or ""
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        api_id_to_tags[api_id] = tags
        all_api_tags.update(tags)

    mashup_id_to_tags = {}
    all_mashup_tags = set()
    for m in mashups:
        mid = m["id"]
        cats = [c.strip() for c in (m.get("categories", "") or "").split(",") if c.strip()]
        mashup_id_to_tags[mid] = cats
        all_mashup_tags.update(cats)

    api_tag2id = {tag: i for i, tag in enumerate(sorted(all_api_tags))}
    mashup_tag2id = {tag: i for i, tag in enumerate(sorted(all_mashup_tags))}

    # Load embedding model
    model = SentenceTransformer(args.model_name)

    # 1) Tag vectors
    api_tag_ids = list(api_tag2id.values())
    api_tag_texts = [tag for tag, tid in sorted(api_tag2id.items(), key=lambda x: x[1])]
    save_vectors_h5(data_dir / "api_tag_vector.h5", api_tag_ids, api_tag_texts, model, args.batch_size)

    mash_tag_ids = list(mashup_tag2id.values())
    mash_tag_texts = [tag for tag, tid in sorted(mashup_tag2id.items(), key=lambda x: x[1])]
    save_vectors_h5(data_dir / "mashup_tag_vector.h5", mash_tag_ids, mash_tag_texts, model, args.batch_size)

    # 2) API vectors (title + description + tags)
    api_ids = [api["id"] for api in apis]
    api_texts = [build_api_text(api) for api in apis]
    save_vectors_h5(data_dir / "API_vectors.h5", api_ids, api_texts, model, args.batch_size)

    # 3) Mashup vectors (title + description + categories)
    mash_ids = [m["id"] for m in mashups]
    mash_texts = [build_mashup_text(m) for m in mashups]
    save_vectors_h5(data_dir / "vectors.h5", mash_ids, mash_texts, model, args.batch_size)

    print("Done. Wrote vectors to:", str(data_dir))

if __name__ == "__main__":
    main()
