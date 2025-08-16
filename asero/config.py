#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

import os
from openai import OpenAI
from dataclasses import dataclass

# --- OpenAI Embedding helpers --- #
os.environ["OPENAI_API_KEY"] = "sk-JsnDAbVmdorAQ8wsXS-hSg"
os.environ["OPENAI_BASE_URL"] = "http://0.0.0.0:4000"
OPENAI_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_CHUNK_SIZE = 128  # Number of utterances to embed in one OpenAI API call

# --- File paths --- #
YAML_TREE_PATH = "semantic_router_tree.yaml"
CACHE_JSON_PATH = "semantic_router_cache.json"

# --- Other constants --- #
DEFAULT_THRESHOLD = 0.5

@dataclass
class SemanticRouterConfig:
    """
    Dataclass for semantic router configuration.

    Attributes:
        client (any): OpenAI API client instance for embedding queries.
        model (str): Embedding model name.
        yaml_file (str): Path to tree YAML definition.
        cache_file (str): Path to embedding cache JSON file.
    """
    client: OpenAI
    model: str
    yaml_file: str
    cache_file: str

client = OpenAI()
