defmodule Trainer do
  alias Dataset.Training.Cifar10
  alias Dataset.Training.MNIST

  def train(:mnist), do: MNIST.run_training()
  def train(:cifar10), do: Cifar10.run_training()
end
