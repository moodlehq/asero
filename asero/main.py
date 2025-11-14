#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main module, demonstration purposes, for asero semantic router."""
import asyncio
import sys
import traceback

from asero.config import get_config
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
    """Prepare the async loop for operation and graceful shutdown, then run()."""
    # Create the event loop, set it as current and add the signal handlers.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    exitcode = 0
    try:
        loop.run_until_complete(run())  # Run the main loop.
    except Exception:
        traceback.print_exc()
        exitcode = 1
    finally:
        loop.close()
        sys.exit(exitcode)


if __name__ == "__main__":
    main()
