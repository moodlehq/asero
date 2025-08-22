#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Configuration for asero semantic router."""

import logging
import os

from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from asero import ROOT_DIR

logger = logging.getLogger(__name__)


@dataclass
class SemanticRouterConfig:
    """Dataclass for semantic router configuration.

    Attributes:
        client (OpenAI): OpenAI API client instance for embedding queries.
        embedding_model (str): Name of the embedding model to use.
        embedding_dimensions (int): Dimensionality of embeddings.
        embedding_chunk_size (int): Number of texts to process in one batch.
        threshold (float): Similarity threshold for routing.
        yaml_file (str): Path to the YAML file defining the tree structure.
        cache_file (str): Path to the JSON file for embedding cache.

    """

    client: OpenAI
    embedding_model: str
    embedding_dimensions: int
    embedding_chunk_size: int
    threshold: float
    yaml_file: str
    cache_file: str


def get_config(yaml_tree_path: str | None = None) -> SemanticRouterConfig:
    """Get the semantic router configuration.

    Args:
        yaml_tree_path (str | None): Optional path to the YAML file defining the tree structure.
            If None, it will use the path defined in the environment variable `ROUTER_YAML_FILE`.
            If the environment variable is not set, it defaults to `router_example.yaml`

    Returns:
        SemanticRouterConfig: The configuration instance.

    """
    # Load environment variables from .env file.
    load_dotenv()

    # OpenAI Embedding helpers.
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")  # Default to OpenAI's URL

    if not api_key:
        msg = "OPENAI_API_KEY environment variable is not set."
        raise ValueError(msg)

    # Embedding model, chunk size, and dimensions.
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    embedding_chunk_size = int(os.getenv("EMBEDDING_CHUNK_SIZE", "128"))
    default_threshold = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))

    # File paths.
    if yaml_tree_path is None:
        # If no path is provided, use the environment variable or default.
        yaml_tree_path = os.getenv("ROUTER_YAML_FILE", "router_example.yaml")
        # If file is relative, make it relative to the project base dir.
        if not os.path.isabs(yaml_tree_path):
            yaml_tree_path = os.path.join(ROOT_DIR, yaml_tree_path)
    cache_json_path = f"{os.path.splitext(yaml_tree_path)[0]}_cache.json"

    return SemanticRouterConfig(
        client=OpenAI(api_key=api_key, base_url=base_url),
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        embedding_chunk_size=embedding_chunk_size,
        threshold=default_threshold,
        yaml_file=yaml_tree_path,
        cache_file=cache_json_path,
    )
