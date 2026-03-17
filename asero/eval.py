#  Copyright (c) 2026, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Evaluation module for asero semantic router."""

import json
import logging
import os

from collections import defaultdict
from dataclasses import dataclass

from tqdm import tqdm

from asero import ROOT_DIR
from asero.config import get_config
from asero.router import SemanticRouter, SemanticRouterNode
from asero.util import (
    compute_embedding_config_checksum,
    load_embedding_cache,
    save_embedding_cache,
)

logger = logging.getLogger(__name__)


def _load_query_cache(eval_file: str, config) -> tuple[dict, str, str]:
    """Load (or initialise) the query embedding cache for an eval file.

    Args:
        eval_file (str): Absolute path to the eval JSON file.
        config: Router configuration object.

    Returns:
        Tuple of (query_cache, query_cache_path, expected_checksum).

    """
    query_cache_path = os.path.splitext(eval_file)[0] + "_query_cache.json"
    expected_checksum = compute_embedding_config_checksum(config)
    cached_data, cached_checksum = load_embedding_cache(query_cache_path)
    query_cache = cached_data if cached_checksum == expected_checksum else {}
    return query_cache, query_cache_path, expected_checksum


def _load_eval_data(eval_file: str) -> list | None:
    """Load and return eval cases from a JSON file.

    Args:
        eval_file (str): Absolute path to the eval JSON file.

    Returns:
        Parsed list of eval cases, or ``None`` if the file is not valid JSON.

    """
    with open(eval_file, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.decoder.JSONDecodeError:
            logger.error(f"Eval file '{eval_file}' is not a valid JSON file.")
            return None


@dataclass
class EvaluationResult:
    """A single evaluation result entry."""

    route: str
    score: float
    depth: int
    is_leaf: bool


def evaluate(eval_file: str, metric: str = "top1"):
    """Run evaluation on the supplied eval file instead of the interactive loop.

    Args:
        eval_file (str): Path to the JSON eval file.
        metric (str): Evaluation metric — ``"top1"`` through ``"top5"``.
            Defaults to ``"top1"``.

    """
    if not os.path.isabs(eval_file):
        eval_file = os.path.join(ROOT_DIR, eval_file)

    if not os.path.isfile(eval_file) or not os.access(eval_file, os.R_OK):
        logger.error(f"Eval file '{eval_file}' does not exist or is not readable.")
        return

    top_k = int(metric[3:])
    mrr_k = top_k + 1  # MRR@(K+1): always one position beyond top-K so it carries extra information
    top_n = mrr_k

    # We are going to load the router, then the eval file, and compute metrics
    config = get_config()
    router = SemanticRouter(config, eval_mode=False)  # Defaults to router_example.yaml

    query_cache, query_cache_path, expected_checksum = _load_query_cache(eval_file, config)

    eval_data = _load_eval_data(eval_file)
    if eval_data is None:
        return

    # Iterate over eval cases, invoking the router and collecting metrics.
    topk_hits = 0          # cases where expected appears in the top-K results
    reciprocal_ranks = []  # 1/rank if found within mrr_k (= top_k+1), else 0.0
    y_score = []           # best similarity score
    details = []           # store information about each case, so we can inspect/adjust stuff later

    # Evaluation loop
    # get the last 100 elements for quick testing
    # eval_data = eval_data[-100:] if len(eval_data) > 100 else eval_data

    for case in tqdm(eval_data, desc="Evaluating"):

        utterance = case["utterance"]
        expected = case["match"]

        router_matches = router.top_n_routes(utterance, top_n=top_n, query_cache=query_cache)
        matches = [
            EvaluationResult(
                route=router_match[0],
                score=router_match[1],
                depth=router_match[2],
                is_leaf=router_match[3]
            )
            for router_match in router_matches
        ]

        # Find rank of expected route in the returned list (1-indexed).
        # top-K hit: expected must be within the first top_k positions.
        # MRR@(K+1): rr is 0 for any rank beyond mrr_k, giving one extra position of signal.
        returned_routes = [m.route for m in matches]
        if expected in returned_routes:
            rank = returned_routes.index(expected) + 1
            if rank <= top_k:
                topk_hits += 1
            rr = 1.0 / rank if rank <= mrr_k else 0.0
        else:
            rank = None
            rr = 0.0

        reciprocal_ranks.append(rr)

        best_score = matches[0].score if matches else 0.0
        y_score.append(best_score)

        details.append({
            "utterance": utterance,
            "expected": expected,
            "returned": returned_routes,
            "rank": rank,
            "best_score": best_score,
            "scores": [(m.route, m.score) for m in matches],
        })

    # Persist the query embedding cache for future runs.
    save_embedding_cache(query_cache, query_cache_path, expected_checksum)

    # Calculate metrics.
    total = len(reciprocal_ranks)
    if not total:
        logger.warning("No eval cases to evaluate.")
        return

    topk_accuracy = topk_hits / total
    mrr = sum(reciprocal_ranks) / total

    print("Evaluation finished.")
    print(f"Evaluation metric     : {metric}")
    print(f"Total eval cases      : {total}")
    print(f"Top-{top_k} Accuracy  : {topk_accuracy:0.7f}")
    print(f"MRR@{mrr_k}                  : {mrr:0.7f}")


def _compute_metric_for_threshold(query_results: list, route: str, threshold: float, metric: str) -> int:
    """Simulate applying ``threshold`` to a single ``route`` and compute the target metric.

    All other routes remain unfiltered (threshold 0).  Returns the raw hit count.

    Args:
        query_results: Per-query result list produced by ``optimise()``.
        route: The route whose threshold is being evaluated.
        threshold: Candidate threshold value for ``route``.
        metric: One of ``"top1"`` through ``"top5"``.

    Returns:
        Number of queries where the expected route appears in the top-K visible routes.

    """
    top_k = int(metric[3:])
    hits = 0
    for q in query_results:
        visible = {r: s for r, s in q["scores"].items() if r != route or s >= threshold}
        if not visible:
            continue
        ranked = sorted(visible, key=lambda r: visible[r], reverse=True)[:top_k]
        if q["expected"] in ranked:
            hits += 1
    return hits


def optimise(eval_file: str, metric: str = "top1", write: bool = False):
    """Run threshold optimisation on the supplied eval file.

    Loads the router in eval mode (thresholds disabled), collects raw similarity
    scores for every eval case, then sweeps candidate thresholds per route to
    maximise the chosen metric.  Recommended thresholds are printed as a table
    and, when ``write=True``, written back to the YAML file.

    Args:
        eval_file (str): Path to the JSON eval file.
        metric (str): Optimisation objective — ``"top1"`` through ``"top5"``.
            Defaults to ``"top1"``.
        write (bool): If True, write optimised thresholds back to the YAML file.

    """
    if not os.path.isabs(eval_file):
        eval_file = os.path.join(ROOT_DIR, eval_file)

    if not os.path.isfile(eval_file) or not os.access(eval_file, os.R_OK):
        logger.error(f"Eval file '{eval_file}' does not exist or is not readable.")
        return

    print(
        "WARNING: Optimising thresholds on the same data used for testing may overfit.\n"
        "Recommendation: use separate calibration and test splits.\n"
    )

    config = get_config()
    # eval_mode=True disables all thresholds so every route is considered.
    router = SemanticRouter(config, eval_mode=True)

    query_cache, query_cache_path, expected_checksum = _load_query_cache(eval_file, config)

    eval_data = _load_eval_data(eval_file)
    if eval_data is None:
        return

    # Collect per-query full score vectors (top_n=1000 ≈ unlimited).
    query_results: list[dict] = []

    for case in tqdm(eval_data, desc="Collecting scores"):

        utterance = case["utterance"]
        expected = case["match"]

        router_matches = router.top_n_routes(utterance, top_n=1000, query_cache=query_cache)
        scores = {m[0]: m[1] for m in router_matches}
        query_results.append({"expected": expected, "scores": scores})

    # Persist the query embedding cache for future runs.
    save_embedding_cache(query_cache, query_cache_path, expected_checksum)

    # --- Compute optimal threshold per route via metric-aware sweep ---
    all_routes: set[str] = set()
    for q in query_results:
        all_routes.update(q["scores"].keys())

    optimal_thresholds: dict[str, float] = {}

    for route in all_routes:
        # Skip routes that have no positive examples.
        if not any(q["expected"] == route for q in query_results):
            continue

        best_value = -1.0
        best_t = 0.0
        for step in range(101):
            t = step / 100.0
            value = _compute_metric_for_threshold(query_results, route, t, metric)
            if value >= best_value:  # >= prefers the highest threshold that achieves the best value
                best_value = value
                best_t = t
        optimal_thresholds[route] = best_t

    if not optimal_thresholds:
        print("No routes with sufficient data to optimise.")
        return

    # --- Enforce parent ≤ min(children) constraint on optimal thresholds ---
    # Process deepest routes first (post-order) so each parent is capped after
    # all its children have already been capped.
    for route in sorted(optimal_thresholds, key=lambda r: r.count("/"), reverse=True):
        parent = "/".join(route.split("/")[:-1])
        if parent in optimal_thresholds and optimal_thresholds[parent] > optimal_thresholds[route]:
            optimal_thresholds[parent] = optimal_thresholds[route]

    # --- Derive TP / FP counts from query_results for the output table ---
    tp_counts: dict[str, int] = defaultdict(int)
    fp_counts: dict[str, int] = defaultdict(int)
    for q in query_results:
        tp_counts[q["expected"]] += 1
        if q["scores"]:
            top1 = max(q["scores"], key=lambda r: q["scores"][r])
            if top1 != q["expected"]:
                fp_counts[top1] += 1

    # --- Load a second router (eval_mode=False) to read current thresholds ---
    router_orig = SemanticRouter(config, eval_mode=False)

    # --- Print results table ---
    col_route = 60
    print(f"\nOptimisation metric: {metric}")
    print(f"\n{'Route':<{col_route}} {'Current':>8} {'Optimal':>8} {'TPs':>6} {'FPs':>6}")
    print("-" * (col_route + 32))
    for route in sorted(optimal_thresholds.keys()):
        path = route.split("/")
        node = router_orig.root.find_node(path)
        current = node.threshold if node else 0.0
        opt = optimal_thresholds[route]
        n_tp = tp_counts[route]
        n_fp = fp_counts[route]
        print(f"{route:<{col_route}} {current:>8.3f} {opt:>8.3f} {n_tp:>6} {n_fp:>6}")

    if write:
        # Apply optimised leaf thresholds then propagate upwards via the existing
        # validate_and_fix_nodes logic (parent ≤ min(children)).
        for route, opt_threshold in optimal_thresholds.items():
            path = route.split("/")
            node = router_orig.root.find_node(path)
            if node is not None:
                node.threshold = opt_threshold

        SemanticRouterNode.validate_and_fix_nodes(router_orig.root)
        router_orig.root.save(config)
        print(f"\nThresholds written to {config.yaml_file}")
    else:
        print("\nRun with --write to apply these thresholds to the YAML file.")
