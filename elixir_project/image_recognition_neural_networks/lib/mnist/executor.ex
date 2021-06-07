defmodule MNIST.Executor do
  alias MNIST.DatasetTrain.Trainer

  def execute(), do: Trainer.execute()

  def which_number?(image, algorithm) do
    {:ok, image, algorithm}
  end
end
