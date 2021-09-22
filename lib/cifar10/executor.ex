defmodule CIFAR10.Executor do
#  alias CIFAR10.DatasetTrain.Trainer
  alias CIFAR10.DatasetTrain.AxonTrainer

  def execute(), do: raise "implementation still incomplete"

  def execute_axon_impl(), do: AxonTrainer.execute()

end
