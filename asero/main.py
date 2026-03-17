#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main module, demonstration purposes, evaluate and optimise, for asero semantic router."""

import argparse
import asyncio
import sys
import traceback

from asero.config import get_config
from asero.eval import evaluate, optimise
from asero.router import SemanticRouter


async def run():
    """Demonstrate the SemanticRouter functionality."""
    config = get_config()
    router = SemanticRouter(config)  # Defaults to router_example.yaml
    top = 3

    # Let's play with the router.
    while True:
        try:
            print(f"Type a query to see top-{top} semantic routes (ctrl-C to exit):")
            q = (await asyncio.to_thread(input, "You: ")).strip()
            matches = await router.atop_n_routes(q, top_n=top)
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


def main():
    """Parse CLI args and dispatch to run() or evaluate()."""
    parser = argparse.ArgumentParser(description="Asero semantic-router demo")
    parser.add_argument(
        "--evaluate",
        metavar="<path to eval file>",
        help="Run evaluation on the specified eval file instead of the REPL",
    )
    parser.add_argument(
        "--optimise",
        metavar="<path to eval file>",
        help="Run threshold optimisation on the specified eval file",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        default=False,
        help="Write optimised thresholds back to the YAML file (only used with --optimise)",
    )
    parser.add_argument(
        "--metric",
        choices=["top1", "top2", "top3", "top4", "top5"],
        default="top1",
        help="Metric for --evaluate and --optimise (default: top1)",
    )
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    exitcode = 0
    try:
        if args.evaluate:
            evaluate(args.evaluate, metric=args.metric)  # Let's execute the evaluation.
        elif args.optimise:
            optimise(args.optimise, metric=args.metric, write=args.write)  # Run threshold optimisation.
        else:
            loop.run_until_complete(run())  # Run the interactive demo.
    except Exception:
        traceback.print_exc()
        exitcode = 1
    finally:
        loop.close()
        sys.exit(exitcode)


if __name__ == "__main__":
    main()
