#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

import logging
from asero import LOG_LEVEL
from asero.config import config
from asero.router import SemanticRouterNode, SemanticRouter
from asero.yaml_utils import compute_dict_checksum, load_or_regenerate_embedding_cache_for_tree


def main():
    router = SemanticRouter()
    router.logger.info("\nType a query to see top-3 semantic routes (ctrl-C to exit):")
    while True:
        try:
            q = input("You: ").strip()
            matches = router.root.top_n_routes(q, router.embedding_cache, top_n=7)
            router.logger.info("")
            router.logger.info(f"Query: {q}")
            router.logger.info("Top nodes:")
            for route, score, depth, is_leaf in matches:
                router.logger.info(f"  {route:<55} {score:.7f} (depth={depth}, is_leaf={is_leaf})")
        except KeyboardInterrupt:
            router.logger.info("\nExiting.")
            break


if __name__ == "__main__":
    main()
