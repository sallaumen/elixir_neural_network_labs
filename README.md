# Elixir Neural Network labs
Elixir implementation of MNIST and CIFAR10 neural networks

## Installation

```elixir
def deps do
  [
    {:elixir_neural_network_labs, "~> 0.1.0"}
  ]
end
```

## Prerequisite for EXLA

Bazel 3.1.0<br>
>asdf plugin add bazel<br>
asdf install bazel 3.1.0<br>
asdf global bazel 3.1.0<br>

Erlang OTP 24 (MacOS)<br>
>brew install erlang

Elixir 1.12<br>
>asdf plugin add elixir<br>
asdf install elixir 1.12-otp-24<br>
asdf global elixir 1.12-otp-24<br>

 Python and NumPy<br>
>sudo apt install python3-pip<br>
pip3 install numpy<br>
cd /usr/bin<br>
sudo ln -s python3 python<br>

## Starting project

Add both dependencies

    {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
    {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", branch: "main", sparse: "nx", override: true}
    {:scidata, "~> 0.1.1"}
>`mix deps.get`<br>
>`mix compile`

Be aware that exla takes a really long time to compile

## Execution

> iex -S mix
<br>
iex(1)> MNIST.Executor.execute()
<br>
iex(2)> CIFAR10.Executor.execute()

## Current status:
### - MNIST
Status: **Working**

![Diagram](MNIST_results.png)
<br>
<br>
<br>

- CIFAR10: Not Working
### - CIFAR10
Status: **Not Working**
