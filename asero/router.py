#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

import logging
from typing import List, Dict, Tuple, Optional, Self

from asero.config import config, SemanticRouterConfig
from asero.embedding import get_or_create_embeddings, cosine_similarity
from asero.yaml_utils import compute_dict_checksum, load_tree_from_yaml, save_embedding_cache
from asero.logger import setup_logging
from yaml_utils import save_tree_to_yaml

logger = logging.getLogger(__name__)


class SemanticRouterNode:
    """
    Node in the semantic router hierarchy/tree.

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
        parent: Self | None = None,
        config: SemanticRouterConfig = config,
        threshold: float = config.threshold,
    ):
        """
        Initialize a SemanticRouterNode.

        Args:
            name (str): Node name.
            utterances (List[str]): Utterances describing this node.
            children (List[SemanticRouterNode], optional): Child nodes.
            parent (SemanticRouterNode, optional): Parent node. Set by parent, or None for root.
            config (SemanticRouterConfig, optional): Config object.
            threshold (float): Similarity threshold for routing.
        """
        self.name = name
        self.utterances = utterances
        self.children = children or []
        self.parent = parent
        self.config = config
        self.threshold = threshold
        # Propagate config to children:
        for child in self.children:
            child.parent = self
            child.config = self.config
        self.embedding_indices = None

    @classmethod
    def load(cls, config: SemanticRouterConfig) -> Tuple[Self, Dict]:
        d = load_tree_from_yaml(config.yaml_file)
        return cls.build(d, config), d

    @classmethod
    def build(
        cls,
        d: Dict,
        config: Optional[SemanticRouterConfig] = None
    ) -> Self:
        node = SemanticRouterNode(
            d["name"],
            d.get("utterances", []),
            [],
            None,
            config,
            d.get("threshold", config.threshold),
        )
        node.children = [
            cls.build(c, config) for c in d.get("children", [])
        ]
        for child in node.children:
            child.parent = node
            child.config = config
        return node

    def save(self, config: SemanticRouterConfig) -> None:
        save_tree_to_yaml(self, config.yaml_file)

    def find_node(self, path: List[str]) -> Optional[Self]:
        """
        Recursively find a node matching the given path (list of names).

        Args:
            path (List[str]): Sequence of node names [root, ..., leaf].

        Returns:
            Optional[SemanticRouterNode]: The node for the path, or None if not found.
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

    def all_utterances(self) -> List[str]:
        """
        Recursively gather all utterances in this node and its children.

        Returns:
            List[str]: All utterances below (and including) this node.
        """
        utt = list(self.utterances)
        for child in self.children:
            utt.extend(child.all_utterances())
        return utt

    def clone_with_parents(self, parent: Optional[Self] = None) -> Self:
        """
        Deep copy this subtree, updating parent pointers and propagating config.

        Args:
            parent (SemanticRouterNode, optional): Parent reference for clone.

        Returns:
            SemanticRouterNode: New tree/subtree, identical structure.
        """
        node = SemanticRouterNode(
            self.name,
            list(self.utterances),
            [child.clone_with_parents() for child in self.children],
            parent,
            self.config,
            self.threshold
        )
        return node

    def compute_embedding_indices(
        self,
        embedding_cache: Dict[str, List[float]]
    ) -> None:
        """
        For this node (and descendants), set .embedding_indices to all utterances
        present in the cache from this node up.

        Args:
            embedding_cache (Dict[str, List[float]]): {utterance: embedding array}

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
        embedding_cache: Dict[str, List[float]],
        top_n: int = 3
    ) -> List[Tuple[str, float, int, bool]]:
        """
        For a given query, return the top-N most similar semantic routes in the hierarchy.

        Args:
            query (str): User query string.
            embedding_cache (Dict[str, List[float]]): {utterance: embedding}
            top_n (int): Number of top routes to return.

        Returns:
            List[Tuple[str, float, int, bool]]: List of tuples:
                (route_path, similarity_score, depth, is_leaf)
        """
        query_embedding = get_or_create_embeddings([query], self.config, embedding_cache)[0]
        sim_cache = {}
        results = {}

        def visit(node: Self, path: List[str]) -> None:
            path_str = '/'.join(path + [node.name])
            # logger.info(f"Visiting: {path_str}")
            if node.parent is None:
                # logger.info(f"Skipping top node {node.name}.")
                for child in node.children:
                    visit(child, path + [node.name])
                return
            if node.embedding_indices and len(node.embedding_indices) > 0:
                # logger.info(f"Found {len(node.embedding_indices)} embeddings for {path_str}.")
                scores = []
                for utt in node.embedding_indices:
                    if utt not in sim_cache:
                        sim_cache[utt] = cosine_similarity(query_embedding, embedding_cache[utt])
                    scores.append(sim_cache[utt])
                max_score = max(scores)
            else:
                max_score = float('-inf')
                # logger.info(f"Max score for {path_str}: {max_score:.7f}")
            threshold = getattr(node, "threshold", config.threshold)
            if max_score < threshold:
                # If the max score is below required threshold, skip node and all children.
                # logger.info(f"Score {max_score:.7f} below threshold {threshold}, branch terminated.")
                return  # We are done for this branch.
            # Visit children and collect their best scores too.
            children_best = []
            for child in node.children:
                visit(child, path + [node.name])
                child_path = '/'.join(path + [node.name, child.name])
                if child_path in results:
                    children_best.append(results[child_path][0])
            if children_best and max(children_best) >= max_score:
                # logger.info(f"Gor a child (or children) with a better score than current path {path_str}.")
                # Got a child with a better score than current path remove current (if it was there).
                if path_str in results:
                    del results[path_str]
                return  # We are done for this branch.
            # Still here, current path is a good candidate.
            results[path_str] = (max_score, len(path) + 1, not node.children)  # (score, depth, is_leaf)

        visit(self, [])
        candidates = [(path, score, depth, is_leaf) for path, (score, depth, is_leaf) in results.items()]
        candidates.sort(key=lambda tup: tup[1], reverse=True)
        return candidates[:top_n]

    def persist_tree_and_update_cache(
        self,
        tree_copy: Self,
        embedding_cache: Dict[str, List[float]]
    ) -> None:
        """
        Save a tree to YAML, purge unused embeddings, and update cache.

        Args:
            tree_copy (SemanticRouterNode): Tree to persist.
            embedding_cache (Dict[str, List[float]]): Embedding cache to trim/save.
        """
        tree_copy.save(self.config)
        # Filter cache, removing old utterances not in the new tree.
        all_utts = set(tree_copy.all_utterances())
        cache_keys_to_remove = set(embedding_cache.keys()) - all_utts
        for k in cache_keys_to_remove:
            del embedding_cache[k]
        # Load the new tree from YAML to compute its checksum and add new cache entries if needed.
        new_tree_dict = load_tree_from_yaml(self.config.yaml_file)
        new_tree_checksum = compute_dict_checksum(new_tree_dict)
        save_embedding_cache(embedding_cache, self.config.cache_file, new_tree_checksum)

    def add_utterance_transactional(
        self,
        path: List[str],
        new_utt: str,
        embedding_cache: Dict[str, List[float]]
    ) -> Self:
        """
        Clone current tree, add a new utterance to a node (by path), update cache/YAML.

        Args:
            path (List[str]): Path to node where to add.
            new_utt (str): New utterance to add.
            embedding_cache (Dict[str, List[float]]): Embedding cache, updated as needed.

        Returns:
            SemanticRouterNode: Updated root node (possibly reloaded from YAML).
        Raises:
            ValueError: If node path not found or root.
        """
        if path == [self.name]:
            raise ValueError("Utterances at root node are not allowed")
        tree_copy = self.clone_with_parents()
        target = tree_copy.find_node(path)
        if target is None:
            raise ValueError(f"Node path {path} not found for add_utterance")
        if new_utt not in target.utterances:
            target.utterances.append(new_utt)
            _ = get_or_create_embeddings([new_utt], self.config, embedding_cache)
        tree_copy.compute_embedding_indices(embedding_cache)
        self.persist_tree_and_update_cache(tree_copy, embedding_cache)
        return tree_copy

    def remove_utterance_transactional(
        self,
        path: List[str],
        utt_to_remove: str,
        embedding_cache: Dict[str, List[float]]
    ) -> Self:
        """
        Clone current tree, remove an utterance from given node, update cache/YAML.

        Args:
            path (List[str]): Path to node for utterance removal.
            utt_to_remove (str): The utterance to remove.
            embedding_cache (Dict[str, List[float]]: Embedding cache.

        Returns:
            SemanticRouterNode: Updated root node.

        Raises:
            ValueError: If node/path not found or root.
        """
        if path == [self.name]:
            raise ValueError("Utterances at root node are not allowed")
        tree_copy = self.clone_with_parents()
        target = tree_copy.find_node(path)
        if target is None:
            raise ValueError(f"Node path {path} not found for remove_utterance")
        target.utterances = [utt for utt in target.utterances if utt != utt_to_remove]
        tree_copy.compute_embedding_indices(embedding_cache)
        self.persist_tree_and_update_cache(tree_copy, embedding_cache)
        return tree_copy

    def replace_utterances_transactional(
        self,
        path: List[str],
        new_utterances: List[str],
        embedding_cache: Dict[str, List[float]]
    ) -> Self:
        """
        Clone tree, replace all utterances of a node, update cache/YAML.

        Args:
            path (List[str]): Node path.
            new_utterances (List[str]): New utterances list.
            embedding_cache (Dict[str, List[float]]: Embedding cache.

        Returns:
            SemanticRouterNode: Updated root node.

        Raises:
            ValueError: If node/path not found or is root.
        """
        if path == [self.name]:
            raise ValueError("Utterances at root node are not allowed")
        tree_copy = self.clone_with_parents()
        target = tree_copy.find_node(path)
        if target is None:
            raise ValueError(f"Node path {path} not found for replace_utterances")
        target.utterances = list(new_utterances)
        _ = get_or_create_embeddings(target.utterances, self.config, embedding_cache)
        tree_copy.compute_embedding_indices(embedding_cache)
        self.persist_tree_and_update_cache(tree_copy, embedding_cache)
        return tree_copy


class SemanticRouter:
    def __init__(self):
        setup_logging(level=LOG_LEVEL)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Another Semantic Router starting up...")

        self.root, self.tree_dict = SemanticRouterNode.load(config)
        self.tree_checksum = compute_dict_checksum(self.tree_dict)
        self.embedding_cache = load_or_regenerate_embedding_cache_for_tree(self.root, config, self.tree_checksum)
        self.root.compute_embedding_indices(self.embedding_cache)
