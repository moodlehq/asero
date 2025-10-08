# Asero - Semantic Router for Intent Classification

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

(see the [Contributing](#Contributing) section for more details)

## Use as library

Don't forget to configure the `.env` file (see "Quick start" above).

```python
from asero import config as asero_config
from asero.router import SemanticRouter

# Load the configuration from a YAML file (see router_example.yaml)
config = asero_config.get_config("path/to/valid/router.yaml")

# Create a router instance
router = SemanticRouter(config)

# Start making queries (explore options in the SemanticRouter class)
matches = router.top_n_routes(query="Hi!")

# Or, for async contexts:
matches = await router.atop_n_routes(query="Hi!")

# Print the top matches (route, score, depth, leaf)
[print(match) for match in matches]
```

Will output something like:
```
('router_example/greetings/neutral', 0.9999999999999999, 3, True)
('router_example/chitchat', 0.48668323180162615, 2, True)
```


## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for more information.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## Code of Conduct

Please note that this project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

----
© 2025 Moodle Research Team
