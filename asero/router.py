#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""Main asero semantic router classes."""

import logging
import re

from typing import Self

import numpy as np

from asero import LOG_LEVEL, __version__
from asero.config import SemanticRouterConfig
from asero.embedding import cosine_similarity, get_embeddings, get_or_create_embeddings
from asero.logger import setup_logging
from asero.util import (
    compute_dict_checksum,
    load_or_regenerate_embedding_cache_for_tree,
    load_tree_from_yaml,
    save_embedding_cache,
    save_tree_to_yaml,
)

logger = logging.getLogger(__name__)


class SemanticRouterNode:
    """Node in the semantic router hierarchy/tree.

    Each node contains:
      - a name,
      - a list of utterances (sample texts),
      - children nodes,
      - a parent pointer,
      - configuration and similarity threshold.
    """

    def __init__(
        self,
        name: str,
        utterances: list[str],
        children: list[Self],
        config: SemanticRouterConfig,
        parent: Self | None = None,
        threshold: float | None = None,
    ):
        """Initialize a SemanticRouterNode.

        Args:
            name (str): Node name.
            utterances (list[str]): Utterances describing this node.
            children (list[SemanticRouterNode] | None): Child nodes.
            config (SemanticRouterConfig): Config object.
            parent (SemanticRouterNode | None): Parent node. Set by parent, or None for root.
            threshold (float): Similarity threshold for routing.

        """
        self.name = name
        self.utterances = utterances
        self.children = children or []
        self.config = config
        self.parent = parent
        self.threshold = threshold or config.threshold
        # Propagate config to children:
        for child in self.children:
            child.parent = self
            child.config = self.config
        self.embedding_indices = None

    @classmethod
    def load(cls, config: SemanticRouterConfig) -> tuple["SemanticRouterNode", dict]:
        """Load the semantic router tree from a YAML file.

        Args:
            config (SemanticRouterConfig): Configuration object containing the YAML file path.

        Returns:
            tuple[SemanticRouterNode, dict]: A tuple containing the root node and the parsed YAML dictionary.

        """
        d = load_tree_from_yaml(config.yaml_file)
        return cls.build(d, config), d

    @classmethod
    def build(
        cls,
        d: dict,
        config: SemanticRouterConfig
    ) -> "SemanticRouterNode":
        """Build a SemanticRouterNode from a dictionary structure.

        Args:
            d (dict): Dictionary representing the node and its children.
            config (SemanticRouterConfig): Configuration object.

        Returns:
            SemanticRouterNode: The constructed node.

        """
        node = SemanticRouterNode(
            name=d["name"],
            utterances=d.get("utterances", []),
            children=[],
            config=config,
            parent=None,
            threshold=d.get("threshold", config.threshold),
        )
        node.children = [
            cls.build(c, config) for c in d.get("children", [])
        ]
        for child in node.children:
            child.parent = node
            child.config = config
        return node

    def save(self, config: SemanticRouterConfig) -> None:
        """Save the current node and its subtree to a YAML file.

        Args:
            config (SemanticRouterConfig): Configuration object containing the YAML file path.

        """
        save_tree_to_yaml(self, config.yaml_file)

    def find_node(self, path: list[str]) -> Self | None:
        """Recursively find a node matching the given path (list of names).

        Args:
            path (list[str]): Sequence of node names [root, ..., leaf].

        Returns:
            SemanticRouterNode | None: The node for the path, or None if not found.

        """
        if not path:
            return None
        if self.name != path[0]:
            return None
        if len(path) == 1:
            return self
        for child in self.children:
            found = child.find_node(path[1:])
            if found:
                return found
        return None

    def all_utterances(self) -> list[str]:
        """Recursively gather all utterances in this node and its children.

        Returns:
            list[str]: All utterances below (and including) this node.

        """
        utt = list(self.utterances)
        for child in self.children:
            utt.extend(child.all_utterances())
        return utt

    def clone_with_parents(self, parent: Self | None = None) -> "SemanticRouterNode":
        """Deep copy this subtree, updating parent pointers and propagating config.

        Args:
            parent (SemanticRouterNode | None): Parent reference for clone.

        Returns:
            SemanticRouterNode: New tree/subtree, identical structure.

        """
        node = SemanticRouterNode(
            name=self.name,
            utterances=list(self.utterances),
            children=[child.clone_with_parents() for child in self.children],
            config=self.config,
            parent=parent,
            threshold=self.threshold or self.config.threshold,
        )
        return node

    def compute_embedding_indices(
        self,
        embedding_cache: dict[str, np.ndarray],
    ) -> None:
        """Set the embedding indices for this node and its children.

        Args:
            embedding_cache (dict[str, np.ndarray]): {utterance: embedding array}

        """
        if self.parent is None:
            self.embedding_indices = []
        else:
            utts = self.all_utterances()
            self.embedding_indices = [u for u in utts if u in embedding_cache]
        for c in self.children:
            c.compute_embedding_indices(embedding_cache)

    def top_n_routes(
        self,
        query: str,
        embedding_cache: dict[str, np.ndarray],
        top_n: int = 3,
        only_leaves: bool = True,
        allowed_paths: list[str] | None = None
    ) -> list[tuple[str, float, int, bool]]:
        """For a given query, return the top-N most similar semantic routes in the hierarchy.

        Args:
            query (str): User query string.
            embedding_cache (dict[str, np.ndarray]): {utterance: embedding}
            top_n (int): Number of top routes to return.
            only_leaves (bool): If True, only return leaf nodes.
            allowed_paths (list[str]): List of allowed paths (regexp) to filter results.

        Returns:
            list[tuple[str, float, int, bool]]: List of tuples:
                (route_path, similarity_score, depth, is_leaf)

        """
        embedding = get_embeddings([query], self.config)
        sim_cache = {}
        results = {}

        if not query.strip() or not embedding:
            logger.warning("Empty query or embedding, returning empty results.")
            return []

        query_embedding = embedding[0]

        def visit(node: Self, path: list[str], parent_score: float = 0) -> None:
            path_str = "/".join(path + [node.name])
            logger.debug(f"Visiting {path_str}")
            if node.parent is None:
                for child in node.children:
                    visit(child, path + [node.name])
                return
            if node.embedding_indices and len(node.embedding_indices) > 0:
                scores = []
                for utt in node.embedding_indices:
                    if utt not in sim_cache:
                        sim_cache[utt] = cosine_similarity(query_embedding, embedding_cache[utt])
                    scores.append(sim_cache[utt])
                max_score = max(scores)  # We can change this to averages or others in the future.
                logger.debug(f"  Max score is {max_score:.7f}")
            else:
                max_score = float("-inf")
            threshold = getattr(node, "threshold", self.config.threshold)
            if max_score < threshold:
                logger.debug(f"  Skipping {path_str} with max score {max_score:.7f} < threshold {threshold:.7f}")
                return  # Below threshold, skip this node (branch finished).
            if max_score < parent_score:  # Note this optimisation only works when using max scores.
                logger.debug(f"  Skipping {path_str} with max score {max_score:.7f} < parent score {parent_score:.7f}")
                return  # Below parent score, skip this node (branch finished). No way it will become better.
            for child in node.children:
                visit(child, path + [node.name], max_score)

            results[path_str] = (max_score, len(path) + 1, not node.children)

        visit(self, [])
        candidates = [(path, score, depth, is_leaf) for path, (score, depth, is_leaf) in results.items()]
        candidates.sort(key=lambda tup: tup[1], reverse=True)

        if only_leaves:  # Filter to only include leaf nodes.
            candidates = [c for c in candidates if c[3]]

        if allowed_paths:  # Filter candidates by allowed paths (regexp).
            regexes = [re.compile(ap) for ap in allowed_paths]
            candidates = [
                c for c in candidates if any(rx.search(c[0]) for rx in regexes)
            ]

        return candidates[:top_n]

    def persist_tree_and_update_cache(
        self,
        tree_copy: "SemanticRouterNode",
        embedding_cache: dict[str, np.ndarray]
    ) -> None:
        """Save a tree to YAML, purge unused embeddings, and update cache.

        Args:
            tree_copy (SemanticRouterNode): Tree to persist.
            embedding_cache (dict[str, np.ndarray]): Embedding cache to trim/save.

        """
        tree_copy.save(self.config)
        all_utts = set(tree_copy.all_utterances())
        cache_keys_to_remove = set(embedding_cache.keys()) - all_utts
        for k in cache_keys_to_remove:
            del embedding_cache[k]
        new_tree_dict = load_tree_from_yaml(self.config.yaml_file)
        new_tree_checksum = compute_dict_checksum(new_tree_dict)
        save_embedding_cache(embedding_cache, self.config.cache_file, new_tree_checksum)

    def add_utterance_transactional(
        self,
        path: list[str],
        new_utt: str,
        embedding_cache: dict[str, np.ndarray]
    ) -> "SemanticRouterNode":
        """Add a new utterance to a node (by path), updating cache/YAML.

        Args:
            path (list[str]): Path to node where to add.
            new_utt (str): New utterance to add.
            embedding_cache (dict[str, np.ndarray]): Embedding cache, updated as needed.

        Returns:
            SemanticRouterNode: Updated root node (possibly reloaded from YAML).

        Raises:
            ValueError: If node path not found or root.

        """
        if path == [self.name]:
            msg = "Utterances at root node are not allowed"
            raise ValueError(msg)
        tree_copy = self.clone_with_parents()
        target = tree_copy.find_node(path)
        if target is None:
            msg = f"Node path {path} not found for add_utterance"
            raise ValueError(msg)
        if new_utt not in target.utterances:
            target.utterances.append(new_utt)
            _ = get_or_create_embeddings([new_utt], self.config, embedding_cache)
        tree_copy.compute_embedding_indices(embedding_cache)
        self.persist_tree_and_update_cache(tree_copy, embedding_cache)
        return tree_copy

    def remove_utterance_transactional(
        self,
        path: list[str],
        utt_to_remove: str,
        embedding_cache: dict[str, np.ndarray]
    ) -> "SemanticRouterNode":
        """Remove an utterance from given node (by path), updating cache/YAML.

        Args:
            path (list[str]): Path to node for utterance removal.
            utt_to_remove (str): The utterance to remove.
            embedding_cache (dict[str, np.ndarray]): Embedding cache.

        Returns:
            SemanticRouterNode: Updated root node.

        Raises:
            ValueError: If node/path not found or root.

        """
        if path == [self.name]:
            msg = "Utterances at root node are not allowed"
            raise ValueError(msg)
        tree_copy = self.clone_with_parents()
        target = tree_copy.find_node(path)
        if target is None:
            msg = f"Node path {path} not found for remove_utterance"
            raise ValueError(msg)
        target.utterances = [utt for utt in target.utterances if utt != utt_to_remove]
        tree_copy.compute_embedding_indices(embedding_cache)
        self.persist_tree_and_update_cache(tree_copy, embedding_cache)
        return tree_copy

    def replace_utterances_transactional(
        self,
        path: list[str],
        new_utterances: list[str],
        embedding_cache: dict[str, np.ndarray]
    ) -> "SemanticRouterNode":
        """Replace all utterances of a node (by path), updating cache/YAML.

        Args:
            path (list[str]): Node path.
            new_utterances (list[str]): New utterances list.
            embedding_cache (dict[str, np.ndarray]): Embedding cache.

        Returns:
            SemanticRouterNode: Updated root node.

        Raises:
            ValueError: If node/path not found or is root.

        """
        if path == [self.name]:
            msg = "Utterances at root node are not allowed"
            raise ValueError(msg)
        tree_copy = self.clone_with_parents()
        target = tree_copy.find_node(path)
        if target is None:
            msg = f"Node path {path} not found for replace_utterances"
            raise ValueError(msg)
        target.utterances = list(new_utterances)
        _ = get_or_create_embeddings(target.utterances, self.config, embedding_cache)
        tree_copy.compute_embedding_indices(embedding_cache)
        self.persist_tree_and_update_cache(tree_copy, embedding_cache)
        return tree_copy


class SemanticRouter:
    """Main class for the semantic router, managing the tree and embedding cache.

    Attributes:
        root (SemanticRouterNode): Root node of the semantic router tree.
        tree_dict (dict): Parsed YAML tree structure.
        tree_checksum (str): Checksum of the current tree structure.
        embedding_cache (dict[str, np.ndarray]): Cache of utterance embeddings.

    """

    def __init__(
        self,
        config: SemanticRouterConfig
    ):
        """Initialize the SemanticRouter, loading the tree and embedding cache."""
        setup_logging(level=LOG_LEVEL)
        logger.info("Another Semantic Router (asero) starting up...")
        logger.info(f"Version: {__version__}")
        logger.info(f"Using router YAML file: {config.yaml_file}")

        self.root, self.tree_dict = SemanticRouterNode.load(config)
        self.tree_checksum = compute_dict_checksum(self.tree_dict)
        self.embedding_cache = load_or_regenerate_embedding_cache_for_tree(self.root, config, self.tree_checksum)
        self.root.compute_embedding_indices(self.embedding_cache)

    def top_n_routes(
        self,
        query: str,
        top_n: int = 3,
        only_leaves: bool = True,
        allowed_paths: list[str] | None = None
    ) -> list[tuple[str, float, int, bool]]:
        """For a given query, return the top-N most similar semantic routes in the hierarchy.

        Args:
            query (str): User query string.
            top_n (int): Number of top routes to return.
            only_leaves (bool): If True, only return leaf nodes.
            allowed_paths (list[str]): List of allowed paths to filter results.

        Returns:
            list[tuple[str, float, int, bool]]: List of tuples:
                (route_path, similarity_score, depth, is_leaf)

        """
        return self.root.top_n_routes(  # Just wrap the call to the root node's method.
            query,
            self.embedding_cache,
            top_n=top_n,
            only_leaves=only_leaves,
            allowed_paths=allowed_paths,
        )

    def add_utterance(
        self,
        path: list[str],
        new_utt: str,
    ) -> "SemanticRouterNode":
        """Add a new utterance to a node (by path), updating cache/YAML.

        Args:
            path (list[str]): Path to node where to add.
            new_utt (str): New utterance to add.

        Returns:
            SemanticRouterNode: Updated root node (possibly reloaded from YAML).

        Raises:
            ValueError: If node path not found or root.

        """
        return self.root.add_utterance_transactional(path, new_utt, self.embedding_cache)

    def remove_utterance(
        self,
        path: list[str],
        utt_to_remove: str,
    ) -> "SemanticRouterNode":
        """Remove an utterance from given node (by path), updating cache/YAML.

        Args:
            path (list[str]): Path to node for utterance removal.
            utt_to_remove (str): The utterance to remove.

        Returns:
            SemanticRouterNode: Updated root node.

        Raises:
            ValueError: If node/path not found or root.

        """
        return self.root.remove_utterance_transactional(path, utt_to_remove, self.embedding_cache)

    def replace_utterances(
        self,
        path: list[str],
        new_utterances: list[str],
    ) -> "SemanticRouterNode":
        """Replace all utterances of a node (by path), updating cache/YAML.

        Args:
            path (list[str]): Node path.
            new_utterances (list[str]): New utterances list.

        Returns:
            SemanticRouterNode: Updated root node.

        Raises:
            ValueError: If node/path not found or is root.

        """
        return self.root.replace_utterances_transactional(path, new_utterances, self.embedding_cache)
