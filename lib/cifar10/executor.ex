defmodule CIFAR10.Executor do
  alias CIFAR10.DatasetTrain.Trainer
  alias CIFAR10.DatasetTrain.AxonTrainer

  def execute(), do: Trainer.execute()

  def execute_axon_impl(), do: AxonTrainer.execute()

end
