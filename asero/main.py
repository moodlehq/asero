#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main module, demonstration purposes, for asero semantic router."""

from asero.router import SemanticRouter


def main():
    """Demonstrate the SemanticRouter functionality."""
    router = SemanticRouter()  # Defaults to router_example.yaml
    top = 3

    # Let's play with the router.
    while True:
        try:
            print(f"Type a query to see top-{top} semantic routes (ctrl-C to exit):")
            q = input("You: ").strip()
            matches = router.top_n_routes(q, router.embedding_cache, top_n=top)
            print("")
            print(f"Query: {q}")
            print("Top nodes:")
            for route, score, depth, is_leaf in matches:
                print(f"  {route:<55} {score:.7f} (depth={depth}, is_leaf={is_leaf})")
            if not matches:
                print("No matches (over threshold) found.")
            print("===== ===== ===== ===== =====")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
