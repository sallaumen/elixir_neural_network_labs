# Elixir neural network experiments

Elixir/Nx implementations of the MNIST and CIFAR-10 image classifiers from Lucas Campos Tavano's Computer Engineering capstone project at UTFPR. The [comparison overview](https://github.com/sallaumen/elixir_vs_python_nn_performance_comparison) links this repository to its Python counterpart and explains why the historical results are not a controlled language benchmark.

## Current stack and status

The maintained project targets **Erlang/OTP 29.1.1** and **Elixir 1.20.4**, as pinned in [`.tool-versions`](.tool-versions). The lockfile resolves Nx 1.0.0, EXLA 1.0.0, Axon 0.9.0, and Scidata 0.1.11. Both Axon models build, predict, train, and evaluate synthetic batches on EXLA's host client. Full MNIST and CIFAR-10 training and held-out evaluation completed locally on 7 October 2026. The [validation record](https://github.com/sallaumen/elixir_vs_python_nn_performance_comparison/blob/master/docs/validation-2026-10-07.md) describes the environment and the locally served, checksum-verified CIFAR-10 archive used for that run.

| Dataset | Model | Default training |
| --- | --- | --- |
| MNIST | Dense 128 → Dropout 0.5 → Dense 10 | Adam, 3 epochs |
| CIFAR-10 | Conv 32 → BatchNorm → Pool → Conv 64 → BatchNorm → Pool → Dense 64 → Dropout 0.5 → Dense 10 | Adam, 3 epochs |

Images are normalized to `[0, 1]` and batched in groups of 16. The CIFAR-10 model uses channels-first tensors. `run_training_and_test/0` now downloads and evaluates the **separate test split**. The checked-in [MNIST figure](MNIST_results.png) remains a historical artifact, not a result produced by the current stack.

## Setup

With [asdf](https://asdf-vm.com/) and its Erlang/Elixir plugins:

```sh
asdf install
mix deps.get
mix compile
```

EXLA uses the `host` client by default so the project can run without CUDA. Set `EXLA_CLIENT=cuda` when you have a compatible CUDA installation. See the [EXLA documentation](https://github.com/elixir-nx/nx/tree/main/exla) for native backend requirements.

## Run

From the repository root:

```sh
iex -S mix
```

In IEx:

```elixir
Trainer.train(:mnist)
Dataset.Training.MNIST.run_training_and_test()
Trainer.train(:cifar10)
Dataset.Training.Cifar10.run_training_and_test()
```

Dataset downloads and full training take time. A training call does not evaluate a test split; choose `run_training_and_test/0` when you need a held-out metric. The models differ from their Python counterparts in architecture and optimizer choices.

## Checks

```sh
mix format --check-formatted
mix compile --warnings-as-errors
mix test
```

The tests use synthetic tensors; they do not download datasets. Full dataset runs should be recorded with exact hardware, backend, Git commit, and dependency versions before making performance claims.

## Historical environment

The original benchmark-era toolchain was Erlang/OTP **24.3.4.1**, Elixir **1.13.4-otp-24**, Python **3.8.0**, and Bazel **3.7.2**. Its locked dependencies included Nx **0.2.1**, EXLA **0.2.2**, Scidata **0.1.8**, and an Axon Git revision. See [historical environment details](docs/historical-environment.md) and the [archived lockfile](docs/historical/mix.lock). Those versions describe the old project; they do not describe results from the current code.

## Repository layout

- [`lib/dataset/training/`](lib/dataset/training/) contains the dataset loader and Axon training modules.
- [`lib/dataset/numerical_definition/`](lib/dataset/numerical_definition/) retains the legacy manual Nx implementation.
- [`lib/implementation_model/`](lib/implementation_model/) contains legacy model persistence code.
- [`test/`](test/) contains focused calculations, preprocessing, and synthetic model checks.
