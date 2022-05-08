defmodule Trainer do
  alias DatasetTrain.Cifar10
  alias DatasetTrain.MNIST

  def train(:mnist), do: MNIST.execute()
  def train(:cifar10), do: Cifar10.execute()
end
