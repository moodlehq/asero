#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

import logging
import yaml
import hashlib
import json
import numpy as np
import os
from asero.embedding import get_embeddings

logger = logging.getLogger(__name__)

def load_tree_from_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_tree_to_yaml(tree, yaml_file):
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(node_to_yaml_dict(tree), f, sort_keys=False, indent=2)

def node_to_yaml_dict(node):
    d = {
        'name': node.name,
    }
    if hasattr(node, 'threshold') and hasattr(node, 'parent') and node.parent:
        d['threshold'] = float(node.threshold)
    if hasattr(node, 'utterances') and node.utterances:
        d["utterances"] = node.utterances
    if node.children:
        d['children'] = [node_to_yaml_dict(child) for child in node.children]
    return d

def save_embedding_cache(cache, fname, tree_checksum):
    data = {k: v.tolist() for k, v in cache.items()}
    out = {
        "checksum": tree_checksum,
        "data": data
    }
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(out, f)

def load_embedding_cache(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    data = obj["data"]
    checksum = obj.get("checksum")
    cache = {k: np.array(v) for k, v in data.items()}
    return cache, checksum

def load_or_regenerate_embedding_cache_for_tree(tree_root, config, tree_checksum):
    all_unique_utts = set(tree_root.all_utterances())
    embedding_cache = {}
    if os.path.exists(config.cache_file):
        cache, cached_tree_checksum = load_embedding_cache(config.cache_file)
        if cached_tree_checksum == tree_checksum:
            logger.info("Embedding cache is valid and up to date.")
            embedding_cache = cache
        else:
            logger.info("Cache tree checksum mismatch. Incrementally updating embedding cache.")
            embedding_cache = {utt: cache[utt] for utt in all_unique_utts if utt in cache}
            logger.info(f"Reused {len(embedding_cache)} of {len(all_unique_utts)} required utterances.")
            missing_utts = [utt for utt in all_unique_utts if utt not in embedding_cache]
            if missing_utts:
                logger.info(f"Computing embeddings for {len(missing_utts)} new utterances.")
                new_embs = get_embeddings(missing_utts, config)
                for utt, emb in zip(missing_utts, new_embs):
                    embedding_cache[utt] = emb
            else:
                logger.info("No new utterances to embed.")
            save_embedding_cache(embedding_cache, config.cache_file, tree_checksum)
            logger.info("Cache rebuilt and saved.")
    else:
        logger.info("No embedding cache found. Building new cache...")
        new_embs = get_embeddings(list(all_unique_utts), config)
        for utt, emb in zip(all_unique_utts, new_embs):
            embedding_cache[utt] = emb
        save_embedding_cache(embedding_cache, config.cache_file, tree_checksum)
        logger.info("Cache built and saved.")
    return embedding_cache

def compute_dict_checksum(d):
    data = json.dumps(d, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()
