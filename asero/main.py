#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

import logging
from .config import SemanticRouterConfig, client, OPENAI_EMBEDDING_MODEL, YAML_TREE_PATH, CACHE_JSON_PATH
from .yaml_utils import load_router_from_yaml, compute_dict_checksum, load_or_regenerate_embedding_cache_for_tree

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    config = SemanticRouterConfig(
        client=client,
        model=OPENAI_EMBEDDING_MODEL,
        yaml_file=YAML_TREE_PATH,
        cache_file=CACHE_JSON_PATH
    )
    root, tree_dict = load_router_from_yaml(config.yaml_file, config)
    tree_checksum = compute_dict_checksum(tree_dict)
    embedding_cache = load_or_regenerate_embedding_cache_for_tree(root, config, tree_checksum)
    root.compute_embedding_indices(embedding_cache)
    logger.info("\nType a query to see top-3 semantic routes (ctrl-C to exit):")
    while True:
        try:
            q = input("You: ").strip()
            matches = root.top_n_routes(q, embedding_cache, top_n=7)
            logger.info("")
            logger.info(f"Query: {q}")
            logger.info("Top nodes:")
            for route, score, depth, is_leave in matches:
                logger.info(f"  {route:<55} {score:.7f} (depth={depth}, is_leave={is_leave})")
        except KeyboardInterrupt:
            logger.info("\nExiting.")
            break
        except Exception as e:
            logger.warning(f"Error: {e}")

if __name__ == "__main__":
    main()
