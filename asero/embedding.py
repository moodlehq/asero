#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_embeddings(texts, config):
    """
    Compute embeddings for a list of texts using the OpenAI API.

    Args:
        texts (list[str]): List of text strings to embed.
        config (SemanticRouterConfig): Configuration object.

    Returns:
        list[np.ndarray]: List of embedding vectors.
    """
    if not texts:
        return []
    embeddings = []
    num_chunks = (len(texts) + config.embedding_chunk_size - 1) // config.embedding_chunk_size
    logger.info(f"Computing embeddings for {len(texts):,} texts in {num_chunks:,} chunks")
    for i in range(num_chunks):
        start_idx = i * config.embedding_chunk_size
        end_idx = min((i + 1) * config.embedding_chunk_size, len(texts))
        chunk_texts = texts[start_idx:end_idx]
        resp = config.client.embeddings.create(input=chunk_texts, model=config.embedding_model)
        embeddings.extend([np.array(d.embedding) for d in resp.data])
    return embeddings


def get_or_create_embeddings(utterances, config, cache):
    """
    Retrieve embeddings for a list of utterances, fetching missing ones from the API.

    Args:
        utterances (list[str]): List of utterances to get embeddings for.
        config (SemanticRouterConfig): Configuration object.
        cache (dict): Dictionary of existing utterance embeddings.

    Returns:
        list[np.ndarray]: List of embedding vectors in the same order as input utterances.
    """
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
        fetched = get_embeddings(to_fetch, config)
        for i, utt in enumerate(to_fetch):
            cache[utt] = fetched[i]
        for i, idx in enumerate(fetch_indices):
            embeddings[idx] = fetched[i]
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """
    Compute cosine similarity between two vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score between -1 and 1.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-16))
