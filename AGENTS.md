# Contributor and AI agent guidance

This repository is the Elixir half of a historical capstone comparison. Read `README.md`, `docs/historical-environment.md`, and the [comparison overview](https://github.com/sallaumen/elixir_vs_python_nn_performance_comparison) before changing experiment behavior.

- Keep the Axon MNIST and CIFAR-10 models distinct from the legacy manual Nx trainers. Do not silently replace one with the other.
- The maintained environment is pinned in `.tool-versions` and `mix.lock`. `docs/historical/` preserves the old lockfile. Update documentation when changing runtime, dependencies, batch size, model structure, or evaluation data.
- Preserve dataset shapes, normalization, one-hot encoding, channels-first CIFAR-10 format, and held-out evaluation unless a change explicitly documents the experimental impact.
- Label historical figures as historical. Only call a metric test accuracy when it uses Scidata's test split. Do not claim speed parity with the Python model without the shared protocol in the comparison repository.
- Add focused ExUnit tests for pure transformations and synthetic model execution. Run `mix format --check-formatted`, `mix compile --warnings-as-errors`, and `mix test`. Full training downloads datasets and should be reported separately.
- Write comments, docs, commit messages, and user-facing output in clear English.
