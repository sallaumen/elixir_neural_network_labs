defmodule MNIST.Executor do
  alias MNIST.DatasetTrain.Trainer

  def execute(), do: Trainer.execute()

end
