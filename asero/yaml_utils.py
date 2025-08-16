#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

import yaml
import json
import numpy as np
import os
from .embedding import compute_dict_checksum, get_embeddings
from .config import DEFAULT_THRESHOLD

def load_tree_from_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_node_from_yaml_dict(d, config=None):
    from .tree import SemanticRouterNode
    node = SemanticRouterNode(
        d['name'],
        d.get('utterances', []),
        [],
        None,
        config,
        d.get('threshold', DEFAULT_THRESHOLD)
    )
    node.children = [build_node_from_yaml_dict(c, config) for c in d.get('children', [])]
    for child in node.children:
        child.parent = node
        child.config = config
    return node

def load_router_from_yaml(yaml_file, config=None):
    tree_dict = load_tree_from_yaml(yaml_file)
    return build_node_from_yaml_dict(tree_dict, config), tree_dict

def node_to_yaml_dict(node):
    d = {
        'name': node.name,
    }
    if hasattr(node, 'threshold'):
        d['threshold'] = float(node.threshold)
    d["utterances"] = node.utterances
    if node.children:
        d['children'] = [node_to_yaml_dict(child) for child in node.children]
    return d

def save_router_to_yaml_file(root, yaml_file):
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(node_to_yaml_dict(root), f, sort_keys=False, indent=2)

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
            print("Embedding cache is valid and up to date.")
            embedding_cache = cache
        else:
            print("Cache tree checksum mismatch. Incrementally updating embedding cache.")
            embedding_cache = {utt: cache[utt] for utt in all_unique_utts if utt in cache}
            print(f"Reused {len(embedding_cache)} of {len(all_unique_utts)} required utterances.")
            missing_utts = [utt for utt in all_unique_utts if utt not in embedding_cache]
            if missing_utts:
                print(f"Computing embeddings for {len(missing_utts)} new utterances.")
                new_embs = get_embeddings(missing_utts, config.client, config.model)
                for utt, emb in zip(missing_utts, new_embs):
                    embedding_cache[utt] = emb
            else:
                print("No new utterances to embed.")
            save_embedding_cache(embedding_cache, config.cache_file, tree_checksum)
            print("Cache rebuilt and saved.")
    else:
        print("No embedding cache found. Building new cache...")
        new_embs = get_embeddings(list(all_unique_utts), config.client, config.model)
        for utt, emb in zip(all_unique_utts, new_embs):
            embedding_cache[utt] = emb
        save_embedding_cache(embedding_cache, config.cache_file, tree_checksum)
        print("Cache built and saved.")
    return embedding_cache
