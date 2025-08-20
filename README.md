# asero - Semantic Router for Intent Classification

A(nother) semantic routing system that classifies user queries into hierarchical categories using OpenAI embeddings and cosine similarity.

## Features

- Hierarchical intent routing with configurable similarity thresholds
- Automatic embedding caching for performance optimization
- YAML-based configuration for routing tree structure
- Transactional updates to routing configuration
- OpenAI embedding model integration

## Requirements
- Python 3.12 and up.
- Access to any Open AI compatible endpoint (Ollama, LiteLLM, ...) with some embeddings model available.

## Quick start

1. Install it with `pip install .`
2. Setup the `.env` file (start with the provided `env_template`one).
3. Optionally, edit the `router_example.yaml` to define your routes.
4. Play with the routes using the `asero` CLI command.
5. That's all!

## Development

1. Install development dependencies: `pip install .[dev]`
2. Enable up pre-commit hooks: `pre-commit install`
3. Setup the `.env` file (start with the provided `env_template`one).
4. Hack, hack, hack (the `asero` CLI command, that runs `main.py`, should be enough)
5. Test, test, test. Try to cover as much as possible always.

## Use as library
<<<coming soon>>>

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for more information.

----
© 2025 Moodle Research Team
