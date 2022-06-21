defmodule DatasetTrain.TrainerNumericalDefinition do
  @moduledoc """
  This module is legacy, please when working with models in this project, use DatasetTrain instead
  """
  import Nx.Defn
  alias ImplementationModel.ModelPersistenceLayer

#  @default_defn_compiler EXLA

  defn init_params do
    w1 = Nx.random_normal({784, 128}, 0.0, 0.1, names: [:input, :hidden])
    b1 = Nx.random_normal({128}, 0.0, 0.1, names: [:hidden])
    w2 = Nx.random_normal({128, 10}, 0.0, 0.1, names: [:hidden, :output])
    b2 = Nx.random_normal({10}, 0.0, 0.1, names: [:output])
    {w1, b1, w2, b2}
  end

  defn predict({w1, b1, w2, b2}, batch) do
    batch
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Nx.logistic()
#    |> Nx.sigmoid()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> softmax()
  end

  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t), axes: [:output], keep_axes: true)
  end

  defn loss({w1, b1, w2, b2}, images, labels) do
    predictions = predict({w1, b1, w2, b2}, images)

    positive_loss =
      (Nx.log(predictions) * labels)
      |> Nx.mean(axes: [:output])
      |> Nx.sum()

    -positive_loss
  end

  defn update({w1, b1, w2, b2} = params, images, labels) do
    {grad_w1, grad_b1, grad_w2, grad_b2} = grad(params, fn params -> loss(params, images, labels) end)

    {w1 - grad_w1 * 0.01, b1 - grad_b1 * 0.01, w2 - grad_w2 * 0.01, b2 - grad_b2 * 0.01}
  end

  def get_or_train_neural_network(:train, path, images_zip, epochs) do
    trained_params =
      for epoch <- 1..epochs, reduce: init_params() do
        acc ->
          IO.puts("Epoch #{epoch}")

          for {{imgs, labels}, _} <- images_zip, reduce: acc do
            acc -> update(acc, imgs, labels)
          end
      end

    ModelPersistenceLayer.save_model_params(trained_params, path, :skip)
    trained_params
  end

  def get_or_train_neural_network(:get, path, _, _) do
    ModelPersistenceLayer.load_model_params(path)
  end
end
