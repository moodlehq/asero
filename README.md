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
2. Setup the `.env` file (start with the provided `dotenv.template`one).
3. Optionally, edit the `router_example.yaml` to define your routes.
4. Play with the routes using the `asero` CLI command.
5. That's all!

## Development

1. Install development dependencies: `pip install .[dev]`
2. Enable up pre-commit hooks: `pre-commit install`
3. Setup the `.env` file (start with the provided `dotenv.template`one).
4. Hack, hack, hack (the `asero` CLI command, that runs `main.py`, should be enough)
5. Test, test, test. Try to cover as much as possible always.

(see the [Contributing](#Contributing) section for more details)

## CLI commands

### `augment` — generate an evaluation dataset via LLM

Uses a configured LLM (accessed through any OpenAI-compatible endpoint) to synthetically paraphrase every utterance in your router YAML, producing a JSON file ready for `asero --evaluate` or `asero --optimise`.

```
augment [--input-file <router.yaml>] [--output-file <out.json>] [--model <name>] [--variations N] [--limit N]
```

- **`--input-file`** — router YAML to read utterances from (defaults to `$ROUTER_YAML_FILE`).
- **`--output-file`** — destination JSON file (defaults to `<input-stem>_eval.json`).
- **`--model`** — LLM model name at the configured endpoint (default: `llama3.3-70b`).
- **`--variations`** — number of paraphrases to generate per utterance (default: `5`).
- **`--limit`** — process only the first N utterances; `0` means all (default: `0`).

Requires `OPENAI_API_KEY` and `OPENAI_BASE_URL` to be set (e.g. via `.env`). The script validates that the requested model is available at the endpoint before starting.

Run `augment --help` for full details.

### `asero --evaluate` — measure top-K accuracy and MRR on a dataset

Runs the current router against a JSON eval file and reports top-K accuracy and MRR(K+1).

```
asero --evaluate <path to eval file> [--metric top1|top2|top3|top4|top5]
```

- **`--evaluate`** — path to the JSON evaluation file (produced by `augment`).
- **`--metric`** — which top-K level to report as the headline accuracy figure (default: `top1`).

The evaluation runs in *eval mode*: all routing thresholds are disabled so every query always returns scores, giving a true picture of the embedding quality independent of threshold tuning.

### `asero --optimise` — tune per-route similarity thresholds

Searches for the threshold values that maximise the chosen accuracy metric on a dataset, then optionally writes the results back to the router YAML.

```
asero --optimise <path to eval file> [--metric top1|top2|top3|top4|top5] [--write]
```

- **`--optimise`** — path to the JSON evaluation file (produced by `augment`).
- **`--metric`** — accuracy metric to maximise during the search (default: `top1`).
- **`--write`** — when set, the optimised thresholds are written back to the router YAML file; without this flag the results are printed but not persisted.

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
