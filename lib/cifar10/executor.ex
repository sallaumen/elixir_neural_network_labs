defmodule CIFAR10.Executor do
  alias CIFAR10.DatasetTrain.Trainer

  def execute(), do: Trainer.execute()

  def which_number?(image, algorithm) do
    {:ok, image, algorithm}
  end
end
