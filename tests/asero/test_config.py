#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

"""asero/config.py unit tests."""

import importlib
import sys
import unittest

from unittest.mock import patch


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Ensure the module is not already loaded
        if "asero.config" in sys.modules:
            del sys.modules["asero.config"]

    def test_missing_openai_api_key_raises_error(self):
        with patch("os.getenv", return_value=None):
            with self.assertRaises(ValueError):
                importlib.import_module("asero.config")

    def test_default_embedding_model_used(self):
        with patch("os.getenv") as mock_getenv:
            with patch("os.getenv") as mock_getenv:
                mock_getenv.side_effect = {
                    "OPENAI_API_KEY": "test-key",  # pragma: allowlist-secret
                }.get
                with patch("asero.ROOT_DIR", new="/mock/root"):
                    config_module = importlib.import_module("asero.config")
                    self.assertEqual("nomic-embed-text", config_module.embedding_model)

    def test_yaml_path_resolves_to_absolute(self):
        with patch("os.getenv") as mock_getenv:
            mock_getenv.side_effect = {
                "OPENAI_API_KEY": "test-key",  # pragma: allowlist-secret
                "ROUTER_YAML_FILE": "relative/path.yaml",
            }.get
            with patch("asero.ROOT_DIR", new="/mock/root"):
                config_module = importlib.import_module("asero.config")
                self.assertEqual("/mock/root/relative/path.yaml", config_module.yaml_tree_path)

    def test_cache_file_name_is_correct(self):
        with patch("os.getenv") as mock_getenv:
            mock_getenv.side_effect = {
                "OPENAI_API_KEY": "test-key",  # pragma: allowlist-secret
                "ROUTER_YAML_FILE": "/absolute/path.yaml",
            }.get
            with patch("asero.ROOT_DIR", new="/mock/root"):
                config_module = importlib.import_module("asero.config")
                self.assertEqual("/absolute/path_cache.json", config_module.cache_json_path)

    def test_config_initialization(self):
        with patch("os.getenv") as mock_getenv:
            mock_getenv.side_effect = {
                "OPENAI_API_KEY": "test-key",  # pragma: allowlist-secret
                "EMBEDDING_MODEL": "test-model",
                "EMBEDDING_DIMENSIONS": 100,
                "EMBEDDING_CHUNK_SIZE": 50,
                "DEFAULT_THRESHOLD": 0.7,
                "ROUTER_YAML_FILE": "test.yaml"
            }.get
            with patch("asero.ROOT_DIR", new="/mock/root"):
                config_module = importlib.import_module("asero.config")
                self.assertEqual("test-model", config_module.config.embedding_model)
                self.assertEqual(100, config_module.config.embedding_dimensions)
                self.assertEqual(0.7, config_module.config.threshold)
                self.assertEqual("/mock/root/test.yaml", config_module.config.yaml_file)


if __name__ == "__main__":
    unittest.main()
