#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Paraphrase utterances via OpenAI and save JSON to a *required* output file.

Generate augmented data for testing and optimising the semantic router. This uses
the configured LLM, accessed via OpenAI API, to synthetically create a number of
virtual queries and their matching results towards using the evaluate and optimise
options of the semantic router.

The input is a router YAML file (same format as router_example.yaml). If no input
file is provided as an argument, the script falls back to the path set in the
ROUTER_YAML_FILE environment variable.

The output is a JSON list of dictionaries, each with keys: "utterance" (query),
"match" (route path) and "source" (informative, how the record was generated),
ready to be used both to evaluate and optimise the router thresholds.
"""

import argparse
import json
import os
import re
import sys

import yaml

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
load_dotenv()  # loads OPENAI_API_KEY / OPENAI_BASE_URL from .env file

_missing = [v for v in ("OPENAI_API_KEY", "OPENAI_BASE_URL") if not os.getenv(v)]
if _missing:
    sys.exit(f"Error: required environment variable(s) not set: {', '.join(_missing)}")

client = OpenAI()

# These can be adapted to the placeholder keys that are used in your router file.
RANDOM_FILLERS = {
    "room_name": [
        "Math 101", "Staff Lounge", "Project Phoenix", "Daily Stand-up",
        "Customer Success", "Random", "Coffee Chat", "Moodle All Hands",
        "!EwNZinBOJkaumGIRyI:moodle.com", "!fsKzaSUlHzxhJcwWpN:matrix.org"
    ],
    "user_id": [
        "u12345", "alice", "bob_smith", "98765", "moodle_admin", "moodler",
        "@user.something:moodle.com", "@pepeluis:matrix.org"
    ],
    "course_id": [
        "CSE101", "HIST-240", "PHYS_770", "MBA-Fall", "OOP_with_Python",
        "Test Course", "ALL CAPS COURSE", "all low course"
    ],
    "generic": [
        "news", "updates", "alerts", "notifications", "stream", "important",
        "not a verb", "white", "red", "chorizo", "moodle", "Moodle",
        "Navarrete", "Paris", "eat_potatoes", "eat potatoes", "raining",
    ],
}


def _extract_utterances(node: dict, path: list[str]) -> list[dict[str, str]]:
    """Recursively collect all utterances from a router YAML node and its descendants."""
    current_path = path + [node["name"]]
    results = []
    for utt in node.get("utterances", []):
        results.append({"utterance": utt, "match": "/".join(current_path)})
    for child in node.get("children", []):
        results.extend(_extract_utterances(child, current_path))
    return results


def load_router_yaml(input_file: str) -> list[dict[str, str]]:
    """Load a router YAML file and return a flat list of utterance/match pairs.

    Args:
        input_file (str): Path to the router YAML file.

    Returns:
        List of dicts with ``"utterance"`` and ``"match"`` keys.

    """
    with open(input_file, encoding="utf-8") as f:
        tree = yaml.safe_load(f)
    return _extract_utterances(tree, [])


PLACEHOLDER_RE = re.compile(r"<<[^>]+>>")


def random_fill(placeholder: str) -> str:
    """Return a random filler value for ``placeholder``."""
    import random

    key = placeholder.strip("<>").lower()
    pool = RANDOM_FILLERS.get(key, RANDOM_FILLERS.get("generic", ["unknown"]))
    return random.choice(pool)


def build_prompt(original: str, n: int) -> str:
    """Build the LLM prompt requesting ``n`` paraphrases of ``original``."""
    return f"""You are a data-augmentation assistant.
Generate {n} diverse alternative sentence of the following sentence.
Rules:
- Keep the intent identical.
- Vary vocabulary and sentence structure very slightly on each alternative.
- Replace some (not all)  placeholder <<placeholder>> with a realistic (multi-word allowed) value.
- Replaced placeholders can be quoted (single and double quotes) or not quoted. Apply for any randomly.
- Output ONLY the paraphrases, one per line.

Sentence:
{original}""".strip()


def call_openai(prompt: str, model: str) -> list[str]:
    """Send ``prompt`` to the OpenAI endpoint and return the response lines."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=600,
        n=1,
        stop=None,
    )
    content = resp.choices[0].message.content or ""
    return [ln.strip() for ln in content.splitlines() if ln.strip()]


def augment_record(rec: dict[str, str], variations: int, model: str) -> list[dict[str, str]]:
    """Generate augmented rows for a single utterance record."""
    original = rec["utterance"]
    match = rec["match"]
    rows = [{"utterance": original, "match": match, "source": "original"}]

    prompt = build_prompt(original, variations)
    candidates = call_openai(prompt, model)

    original_filled = PLACEHOLDER_RE.sub(lambda m: random_fill(rec["utterance"]), original)
    if original_filled != original:
        rows.append({"utterance": original_filled, "match": match, "source": "original_filled"})

    for cand in candidates[:variations]:
        cand = PLACEHOLDER_RE.sub(lambda m: random_fill(m.group(0)), cand)
        rows.append({"utterance": cand, "match": match, "source": "generated"})
    return rows


def check_model_available(model: str) -> None:
    """Exit with an error if ``model`` is not available at the configured endpoint.

    Args:
        model (str): Model name to verify.

    """
    try:
        available = [m.id for m in client.models.list().data]
    except Exception as exc:
        sys.exit(f"Error: could not retrieve model list from the API endpoint: {exc}")
    if model not in available:
        sys.exit(
            f"Error: model '{model}' is not available at {os.getenv('OPENAI_BASE_URL')}.\n"
            f"Available models: {', '.join(sorted(available))}"
        )


def show_config(args: argparse.Namespace, input_file: str, output_file: str) -> None:
    """Print all configuration settings before processing."""
    print("==== Configuration =========================================")
    print(f"Router YAML   : {input_file}")
    print(f"Output file   : {output_file}")
    print(f"OpenAI model  : {args.model}")
    print(f"Variations    : {args.variations}")
    print(f"Limit         : {args.limit if args.limit else 'all'}")
    print(f"API base URL  : {os.getenv('OPENAI_BASE_URL')}")
    print("API key set   : yes")
    print("===========================================================")
    print()


def main():
    """Parse CLI arguments and run the augmentation pipeline."""
    class _Formatter(argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_Formatter,
    )
    parser.add_argument(
        "--input-file", metavar="YAML_FILE",
        help="Router YAML file to read utterances from (default: $ROUTER_YAML_FILE)",
    )
    parser.add_argument(
        "--output-file", metavar="JSON_FILE",
        help="Destination JSON file (default: <input-file>_eval.json)",
    )
    parser.add_argument(
        "--model", default="llama3.3-70b",
        help="OpenAI model name (default: llama3.3-70b)",
    )
    parser.add_argument(
        "--variations", type=int, default=5,
        help="Paraphrases per utterance (default: 5)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Process only the first N utterances (default: 0 = all)",
    )

    args = parser.parse_args()

    input_file = args.input_file or os.getenv("ROUTER_YAML_FILE")
    if not input_file:
        parser.error("provide --input-file or set ROUTER_YAML_FILE")

    output_file = args.output_file or os.path.splitext(input_file)[0] + "_eval.json"

    check_model_available(args.model)
    show_config(args, input_file, output_file)

    data_in = load_router_yaml(input_file)
    if args.limit:
        data_in = data_in[:args.limit]

    data_out = []
    for rec in tqdm(data_in, desc="Augmenting"):
        data_out.extend(augment_record(rec, args.variations, args.model))

    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(data_out, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
