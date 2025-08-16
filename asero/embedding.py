#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import hashlib
import json
from .config import OPENAI_EMBEDDING_MODEL, EMBEDDING_CHUNK_SIZE

def compute_dict_checksum(d):
    data = json.dumps(d, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def get_embeddings(texts, client, model=OPENAI_EMBEDDING_MODEL):
    if not texts:
        return []
    embeddings = []
    num_chunks = (len(texts) + EMBEDDING_CHUNK_SIZE - 1) // EMBEDDING_CHUNK_SIZE
    print(f"Computing embeddings for {len(texts):,} texts in {num_chunks:,} chunks")
    for i in range(num_chunks):
        start_idx = i * EMBEDDING_CHUNK_SIZE
        end_idx = min((i + 1) * EMBEDDING_CHUNK_SIZE, len(texts))
        chunk_texts = texts[start_idx:end_idx]
        resp = client.embeddings.create(input=chunk_texts, model=model)
        embeddings.extend([np.array(d.embedding) for d in resp.data])
    return embeddings

def get_or_create_embeddings(utterances, client, cache, model=OPENAI_EMBEDDING_MODEL):
    embeddings = []
    to_fetch = []
    fetch_indices = []
    for idx, utt in enumerate(utterances):
        if utt in cache:
            embeddings.append(cache[utt])
        else:
            to_fetch.append(utt)
            fetch_indices.append(idx)
            embeddings.append(None)
    if to_fetch:
        fetched = get_embeddings(to_fetch, client, model)
        for i, utt in enumerate(to_fetch):
            cache[utt] = fetched[i]
        for i, idx in enumerate(fetch_indices):
            embeddings[idx] = fetched[i]
    return embeddings

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-16))
