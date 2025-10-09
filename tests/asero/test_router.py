#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""asero/router.py unit tests."""

import logging
import unittest

from unittest.mock import MagicMock, patch

from asero.config import SemanticRouterConfig
from asero.router import SemanticRouterNode


class TestSemanticRouterNode(unittest.TestCase):

    def setUp(self):
        # Mock SemanticRouterConfig for testing
        self.mock_config = MagicMock(spec=SemanticRouterConfig)
        self.mock_config.threshold = 0.5  # Default threshold

        # Suppress logging for asero.router during tests
        self._logger = logging.getLogger("asero.router")
        self._original_level = self._logger.level
        self._logger.setLevel(logging.CRITICAL)

    def tearDown(self):
        # Restore logging level after tests
        self._logger.setLevel(self._original_level)

    # Helper function to create a simple node, now using self.mock_config
    def create_node(self, name, threshold, children=None):
        node = SemanticRouterNode(
            name=name,
            utterances=[],
            children=children or [],
            config=self.mock_config,
            threshold=threshold if threshold is not None else self.mock_config.threshold,
        )
        # Manually set parent and config for children as init only propagates
        # config from a given parent, but here we're constructing bottom-up
        for child in node.children:
            child.parent = node
            child.config = self.mock_config
        return node

    def test_enforce_threshold_rule_leaf_node(self):
        # Test case 1: Leaf node
        leaf_node = self.create_node("Leaf", 0.7)

        # The rule should return the node's own threshold without modification
        self.assertEqual(0.7, SemanticRouterNode._enforce_threshold_rule(leaf_node))
        self.assertEqual(0.7, leaf_node.threshold)  # Ensure it wasn't changed

    def test_enforce_threshold_rule_parent_no_violation(self):
        # Test case 2: Parent node with children, no violation
        child1 = self.create_node("Child1", 0.6)
        child2 = self.create_node("Child2", 0.8)
        parent_node = self.create_node("Parent", 0.5, children=[child1, child2])

        # Parent's threshold (0.5) is <= min(child1.threshold, child2.threshold) (0.6), so no change
        self.assertEqual(0.5, SemanticRouterNode._enforce_threshold_rule(parent_node))
        self.assertEqual(0.5, parent_node.threshold)  # Ensure it wasn't changed
        self.assertEqual(0.6, child1.threshold)
        self.assertEqual(0.8, child2.threshold)

    def test_enforce_threshold_rule_parent_violation(self):
        # Test case 3: Parent node with children, violation
        child1 = self.create_node("Child1", 0.6)
        child2 = self.create_node("Child2", 0.4)  # This child has a lower threshold
        parent_node = self.create_node("Parent", 0.7, children=[child1, child2])  # Violation here.

        # Parent's threshold (0.7) is > min(child1.threshold, child2.threshold) (0.4)
        # It should be adjusted to 0.4
        self.assertEqual(0.4, SemanticRouterNode._enforce_threshold_rule(parent_node))
        self.assertEqual(0.4, parent_node.threshold)  # Ensure it was changed
        self.assertEqual(0.6, child1.threshold)
        self.assertEqual(0.4, child2.threshold)

    def test_enforce_threshold_rule_nested_nodes(self):
        # Test case 4: Deeper hierarchy
        grandchild1 = self.create_node("Grandchild1", 0.3)
        grandchild2 = self.create_node("Grandchild2", 0.5)
        child1 = self.create_node("Child1", 0.7, children=[grandchild1, grandchild2])  # Violation here.

        grandchild3 = self.create_node("Grandchild3", 0.6)
        child2 = self.create_node("Child2", 0.5, children=[grandchild3])

        root_node = self.create_node("Root", 0.8, children=[child1, child2])  # Violation here.

        expected_root_threshold = 0.3
        expected_child1_threshold = 0.3
        expected_child2_threshold = 0.5

        self.assertEqual(expected_root_threshold, SemanticRouterNode._enforce_threshold_rule(root_node))
        self.assertEqual(expected_root_threshold, root_node.threshold)
        self.assertEqual(expected_child1_threshold, child1.threshold)
        self.assertEqual(expected_child2_threshold, child2.threshold)
        self.assertEqual(0.3, grandchild1.threshold)
        self.assertEqual(0.5, grandchild2.threshold)
        self.assertEqual(0.6, grandchild3.threshold)

    def test_enforce_threshold_rule_different_threshold_values(self):
        # Test case 5: Different threshold values, including 0 and 1
        child_min = self.create_node("ChildMin", 0.0)
        child_max = self.create_node("ChildMax", 1.0)
        child_mid = self.create_node("ChildMid", 0.5)

        # Parent above min
        parent1 = self.create_node("Parent1", 0.7, children=[child_min, child_max, child_mid])
        self.assertEqual(0.0, SemanticRouterNode._enforce_threshold_rule(parent1))
        self.assertEqual(0.0, parent1.threshold)

        # Parent equal to min
        child_a = self.create_node("ChildA", 0.2)
        child_b = self.create_node("ChildB", 0.4)
        parent2 = self.create_node("Parent2", 0.2, children=[child_a, child_b])
        self.assertEqual(0.2, SemanticRouterNode._enforce_threshold_rule(parent2))
        self.assertEqual(0.2, parent2.threshold)

        # Parent below min
        child_x = self.create_node("ChildX", 0.8)
        child_y = self.create_node("ChildY", 0.9)
        parent3 = self.create_node("Parent3", 0.7, children=[child_x, child_y])
        self.assertEqual(0.7, SemanticRouterNode._enforce_threshold_rule(parent3))
        self.assertEqual(0.7, parent3.threshold)

    def test_enforce_threshold_rule_no_children_no_change(self):
        # Test a node that explicitly has no children (empty list)
        node = self.create_node("NoChildren", 0.5, children=[])
        self.assertEqual(0.5, SemanticRouterNode._enforce_threshold_rule(node))
        self.assertEqual(0.5, node.threshold)

    def test_enforce_threshold_rule_all_children_adjusted(self):
        # Test a scenario where all children also need adjustment, and then the parent
        grandchild1 = self.create_node("GC1", 0.2)
        grandchild2 = self.create_node("GC2", 0.4)

        # Child1 (0.8) > min(GC1=0.2, GC2=0.4) -> adjusts to 0.2
        child1 = self.create_node("Child1", 0.8, children=[grandchild1, grandchild2])

        grandchild3 = self.create_node("GC3", 0.1)
        grandchild4 = self.create_node("GC4", 0.6)

        # Child2 (0.7) > min(GC3=0.1, GC4=0.6) -> adjusts to 0.1
        child2 = self.create_node("Child2", 0.7, children=[grandchild3, grandchild4])

        # Root (0.9) > min(adjusted Child1=0.2, adjusted Child2=0.1) -> adjusts to 0.1
        root_node = self.create_node("Root", 0.9, children=[child1, child2])

        # Expected adjustments:
        # GC1: 0.2
        # GC2: 0.4
        # Child1: 0.2 (from 0.8)
        # GC3: 0.1
        # GC4: 0.6
        # Child2: 0.1 (from 0.7)
        # Root: 0.1 (from 0.9)

        self.assertEqual(0.1, SemanticRouterNode._enforce_threshold_rule(root_node))
        self.assertEqual(0.1, root_node.threshold)
        self.assertEqual(0.2, child1.threshold)
        self.assertEqual(0.1, child2.threshold)
        self.assertEqual(0.2, grandchild1.threshold)
        self.assertEqual(0.4, grandchild2.threshold)
        self.assertEqual(0.1, grandchild3.threshold)
        self.assertEqual(0.6, grandchild4.threshold)

    def test_enforce_threshold_rule_logs_warning(self):
        # Test case for logging a warning when a parent's threshold is adjusted
        child1 = self.create_node("Child1", 0.6)
        child2 = self.create_node("Child2", 0.4)
        parent_node = self.create_node("Parent", 0.7, children=[child1, child2])

        # Use patch.object to mock the logger.warning method
        # This allows us to check if it was called without actually printing to console
        with patch.object(logging.getLogger("asero.router"), "warning") as mock_warning:
            SemanticRouterNode._enforce_threshold_rule(parent_node)
            mock_warning.assert_called_once()
            self.assertIn("adjusting", mock_warning.call_args[0][0].lower())
            self.assertEqual(0.4, parent_node.threshold)


if __name__ == "__main__":
    unittest.main()
