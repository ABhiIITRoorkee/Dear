import json
import os
import h5py
import numpy as np
from sentence_transformers import SentenceTransformer

# 加载数据
with open('Original Dataset/mashup_shuffled.json', 'r', encoding='utf-8') as f:
    mashups = json.load(f)

with open('Original Dataset/apis.json', 'r', encoding='utf-8') as f:
    apis = json.load(f)

# 构造API的id到tags映射
api_id_to_tags = {}
all_api_tags = set()
for api in apis:
    api_id = api['id']
    tags_str = api['details'].get('tags', '')
    tags = [t.strip() for t in tags_str.split(',') if t.strip()]
    api_id_to_tags[api_id] = tags
    all_api_tags.update(tags)

mashup_id_to_tags = {}
all_mashup_tags = set()
for m in mashups:
    mid = m['id']
    cats = [c.strip() for c in m['categories'].split(',') if c.strip()]
    mashup_id_to_tags[mid] = cats
    all_mashup_tags.update(cats)

all_tags = all_api_tags.union(all_mashup_tags)
tag2id = {tag: i for i, tag in enumerate(sorted(all_tags))}

# 初始化SentenceTransformer模型
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def save_vectors_h5(filename, id_to_text):
    with h5py.File(filename, 'w') as f:
        for id_, text in id_to_text.items():
            vec = model.encode(text)
            f.create_dataset(str(id_), data=vec.astype(np.float32))

os.makedirs('data', exist_ok=True)
save_vectors_h5('data/tag_vectors.h5', {tid: tag for tag, tid in tag2id.items()})

api_id_to_desc = {api['id']: api['details']['description'] for api in apis}
save_vectors_h5('data/API_vectors.h5', api_id_to_desc)


mashup_id_to_desc = {m['id']: m['description'] for m in mashups}
save_vectors_h5('data/vectors.h5', mashup_id_to_desc)

N = len(mashups)
end = round(N * 0.9)
print(end)

from math import ceil
cv_mashups = mashups[:end]
folds = 10
fold_size = ceil(len(cv_mashups) / folds)

for i in range(folds):
    fold_dir = f'dataset/fold{i + 1}'
    os.makedirs(fold_dir, exist_ok=True)

    start = i * fold_size
    end = min((i + 1) * fold_size, len(cv_mashups))

    test_set = cv_mashups[start:end]
    train_set = cv_mashups[:start] + cv_mashups[end:]

    with open(os.path.join(fold_dir, 'TE.csv'), 'w', encoding='utf-8') as f:
        for m in train_set:
            line = [str(m['id'])] + [str(api_id) for api_id in m.get('api_info', [])]
            f.write(' '.join(line) + '\n')

    with open(os.path.join(fold_dir, 'RS.csv'), 'w', encoding='utf-8') as f:
        for m in test_set:
            line = [str(m['id'])] + [str(api_id) for api_id in m.get('api_info', [])]
            f.write(' '.join(line) + '\n')

    with open(os.path.join(fold_dir, 'VA.csv'), 'w', encoding='utf-8') as f:
        for m in mashups[end:]:
            line = [str(m['id'])] + [str(api_id) for api_id in m.get('api_info', [])]
            f.write(' '.join(line) + '\n')

    with open(os.path.join(fold_dir, 'api_tags.csv'), 'w', encoding='utf-8') as f:
        for api_id, tags in api_id_to_tags.items():
            tag_ids = [str(tag2id[t]) for t in tags if t in tag2id]
            f.write(f'{api_id} ' + ' '.join(tag_ids) + '\n')

    with open(os.path.join(fold_dir, 'mashup_tags.csv'), 'w', encoding='utf-8') as f:
        for m in train_set:
            tags = mashup_id_to_tags.get(m['id'], [])
            tag_ids = [str(tag2id[t]) for t in tags if t in tag2id]
            f.write(f"{m['id']} " + ' '.join(tag_ids) + '\n')

