# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
and commits should be formatted using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

## [0.4.1] - 2025-10-09

### Added

- Router: Validate tree thresholds are correct, auto-fix and warn by @stronk7 ([32b20cf](https://github.com/moodlehq/asero/commit/32b20cf3b640a4b0ce9934f0b500c6b0e0705521))

### Fixed

- GitHub: Add coverage information to the test workflow by @stronk7 ([5df76c2](https://github.com/moodlehq/asero/commit/5df76c293243135945dfee06a227b5ff46bd658a))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.4.0...v0.4.1

## [0.4.0] - 2025-10-08

### Added

- Async: Provide a new `atop_n_routes()` function for async contexts by @stronk7 ([d3d707e](https://github.com/moodlehq/asero/commit/d3d707e448801e8f749f999d815683bce74d63ed))

### Changed

- Dependencies: Bump various libraries by @stronk7 ([4342f2a](https://github.com/moodlehq/asero/commit/4342f2af949d0770692ad95f1f841b212ac2e6fb))

### Fixed

- Docs: Solve a configuration problem with CHANGELOG generation by @stronk7 ([f7d4032](https://github.com/moodlehq/asero/commit/f7d403253f96e8eafe091441832c5d26a6402d58))
- CLI: Support the new async functions from the CLI by @stronk7 ([1761144](https://github.com/moodlehq/asero/commit/17611442d6898c64fda6f893336fa7d2e099a50b))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.3.0...v0.4.0

## [0.3.0] - 2025-09-15

### Added

- Test: Support for pre-commit, git-cliff and others added by @stronk7 ([d19b88d](https://github.com/moodlehq/asero/commit/d19b88d28f8086629f6c6b31245fb3c302edec65))
- Docs: Add main documentation pages by @stronk7 ([9702680](https://github.com/moodlehq/asero/commit/9702680a51c55ad610483c535c6f652b2b41f1cd))

### Changed

- GitHub: Move from GitLab to GitHub workflows by @stronk7 ([faa6100](https://github.com/moodlehq/asero/commit/faa6100b7235edc8b01c0aa55a343fee83ebed26))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.14...v0.3.0

## [0.2.14] - 2025-09-11

### Changed

- Bump dependencies by @stronk7 ([deef30c](https://github.com/moodlehq/asero/commit/deef30cb02754fba0cd66e45642e6e56e9ccdfe9))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.13...v0.2.14

## [0.2.13] - 2025-09-09

### Changed

- Pass the embedding dimensions by @stronk7 ([4eaaa56](https://github.com/moodlehq/asero/commit/4eaaa56f1a00572f924bc0f84bb874cfeee2965b))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.12...v0.2.13

## [0.2.12] - 2025-08-22

### Added

- Added the "Use as library" section to docs by @stronk7 ([e7a4926](https://github.com/moodlehq/asero/commit/e7a49266dad84264989207900b8065933ac80b08))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.11...v0.2.12

## [0.2.11] - 2025-08-22

### Changed

- Allow get_config() to specify the path to tree.yml file by @stronk7 ([12cc936](https://github.com/moodlehq/asero/commit/12cc936a3b8487c3b85f1f72f98a12286ae1cc24))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.10...v0.2.11

## [0.2.10] - 2025-08-22

### Changed

- Move candidates filtering to separate method by @stronk7 ([e5532ab](https://github.com/moodlehq/asero/commit/e5532abdcbd3df3f2a7bc0aae1190b4e8ca18d47))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.9...v0.2.10

## [0.2.9] - 2025-08-22

### Changed

- Allow top_n_routes() allowed paths to be regexes by @stronk7 ([25ecd34](https://github.com/moodlehq/asero/commit/25ecd34ba809d8cb2f4f55eca4621a3d0a11427a))
- Move from imported config to proper get_config() by @stronk7 ([60494bb](https://github.com/moodlehq/asero/commit/60494bbe25334b1895318f05fce523eefe62241e))
- Improve behaviour on empty queries or embeddings by @stronk7 ([be8d38a](https://github.com/moodlehq/asero/commit/be8d38ace35c9d8313d4d90c04f307439e071ad4))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.8...v0.2.9

## [0.2.8] - 2025-08-21

### Added

- Add info about the router yaml file being used by @stronk7 ([36cc346](https://github.com/moodlehq/asero/commit/36cc346194734695619394034064c2f26eeb1a8e))

### Changed

- Better finding ROOT_DIR up to the dir having an .env file by @stronk7 ([a9e8369](https://github.com/moodlehq/asero/commit/a9e8369907002ca4b60044cfdc004c0d94461ed3))
- Ensure that SemanticRouter uses self embeddings cache by @stronk7 ([b6c4a14](https://github.com/moodlehq/asero/commit/b6c4a145db3824eb3973c1113cf2a59e6f044f1f))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.7...v0.2.8

## [0.2.7] - 2025-08-20

### Added

- Add standard gitlab-ci pipeline (basically run pre-commit) by @stronk7 ([75f5d47](https://github.com/moodlehq/asero/commit/75f5d47b7a3b3a118479b1d313f7ce3473a000c5))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.6...v0.2.7

## [0.2.6] - 2025-08-20

### Added

- Add readme with basic docs, env template, license and a few links by @stronk7 ([18eec11](https://github.com/moodlehq/asero/commit/18eec11ec75cf2b8375316b909421aa4dd98633e))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.5...v0.2.6

## [0.2.5] - 2025-08-20

### Changed

- Lower requirements to python 3.12 by @stronk7 ([9669d5f](https://github.com/moodlehq/asero/commit/9669d5fb3a5f85aad9f21abbb28d2817c6988281))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.4...v0.2.5

## [0.2.4] - 2025-08-18

### Changed

- Wrap public api methods of SemanticRouterNode in SemanticRouter by @stronk7 ([3c152e7](https://github.com/moodlehq/asero/commit/3c152e7ed17adb6e705ac1b3aab44b62b6a21532))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.3...v0.2.4

## [0.2.3] - 2025-08-18

### Changed

- Improve algo to avoid visiting low score branches by @stronk7 ([50a3e36](https://github.com/moodlehq/asero/commit/50a3e36d153e714d94d757ade11a1305867a8742))
- Filter by only-leaves and allowed-paths by @stronk7 ([e824ba7](https://github.com/moodlehq/asero/commit/e824ba7edcce0b2cea587d7227d7067cb31e0342))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.2...v0.2.3

## [0.2.2] - 2025-08-18

### Added

- Add unittests for asero/config.py covering env vars, defaults, and paths by @stronk7 ([37e370a](https://github.com/moodlehq/asero/commit/37e370a7a1a2a367bad458e49d4c598b00533436))

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.1...v0.2.2

## [0.2.1] - 2025-08-18

### Added

- Add and fix docstrings in asero/router.py by @stronk7 ([ab41cd6](https://github.com/moodlehq/asero/commit/ab41cd6dbd5c87a3b6a6a9512372a1e15aee9f58))
- Add and fix docstrings in the remaining modules by @stronk7 ([79d6976](https://github.com/moodlehq/asero/commit/79d6976151cf82f8225a235b3600795b2e54f2bd))

### Changed

- Initial commit by @stronk7 ([e1f5c57](https://github.com/moodlehq/asero/commit/e1f5c57066b5fed5b1dc453db0afd916960f7b7e))
- Re-organise stuff (parameters, files, pre-commit, logging, ...) by @stronk7 ([d646eae](https://github.com/moodlehq/asero/commit/d646eaed5ceede60ef18db0e1b3bd0b76870bd0d))
- Keep a small utility to play in CLI by @stronk7 ([4109642](https://github.com/moodlehq/asero/commit/410964298bec6850ce7d2df725106ea580523a81))
- Towards linters (ruff, pyright) compliance by @stronk7 ([010dfd5](https://github.com/moodlehq/asero/commit/010dfd5af21ae245ad5ac4e9608e1af09583f0e7))

### New Contributors 🧡:

- @stronk7 made their first contribution

**Full Changelog**: https://github.com/moodlehq/asero/compare/v0.2.0...v0.2.1

[0.4.1]: https://github.com/moodlehq/asero/compare/v0.4.0..v0.4.1
[0.4.0]: https://github.com/moodlehq/asero/compare/v0.3.0..v0.4.0
[0.3.0]: https://github.com/moodlehq/asero/compare/v0.2.14..v0.3.0
[0.2.14]: https://github.com/moodlehq/asero/compare/v0.2.13..v0.2.14
[0.2.13]: https://github.com/moodlehq/asero/compare/v0.2.12..v0.2.13
[0.2.12]: https://github.com/moodlehq/asero/compare/v0.2.11..v0.2.12
[0.2.11]: https://github.com/moodlehq/asero/compare/v0.2.10..v0.2.11
[0.2.10]: https://github.com/moodlehq/asero/compare/v0.2.9..v0.2.10
[0.2.9]: https://github.com/moodlehq/asero/compare/v0.2.8..v0.2.9
[0.2.8]: https://github.com/moodlehq/asero/compare/v0.2.7..v0.2.8
[0.2.7]: https://github.com/moodlehq/asero/compare/v0.2.6..v0.2.7
[0.2.6]: https://github.com/moodlehq/asero/compare/v0.2.5..v0.2.6
[0.2.5]: https://github.com/moodlehq/asero/compare/v0.2.4..v0.2.5
[0.2.4]: https://github.com/moodlehq/asero/compare/v0.2.3..v0.2.4
[0.2.3]: https://github.com/moodlehq/asero/compare/v0.2.2..v0.2.3
[0.2.2]: https://github.com/moodlehq/asero/compare/v0.2.1..v0.2.2
[0.2.1]: https://github.com/moodlehq/asero/compare/v0.2.0..v0.2.1

<!-- generated by git-cliff -->
