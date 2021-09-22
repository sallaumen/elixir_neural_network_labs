defmodule MNIST.Executor do
  alias MNIST.DatasetTrain.Trainer
  alias MNIST.DatasetTrain.AxonTrainer

  def execute(), do: Trainer.execute()

  def execute_axon_impl(), do: AxonTrainer.execute()

end
