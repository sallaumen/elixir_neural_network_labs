defmodule DatasetTrain.TrainerNumericalDefinition do
  import Nx.Defn

  @default_defn_compiler EXLA

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
    {grad_w1, grad_b1, grad_w2, grad_b2} = grad(params, loss(params, images, labels))

    {w1 - grad_w1 * 0.01, b1 - grad_b1 * 0.01, w2 - grad_w2 * 0.01, b2 - grad_b2 * 0.01}
  end

  def get_or_train_neural_network(:train, path, images_zip, epochs) do
    trained_params =
      for _ <- 1..epochs,
          {{imgs, labels}, _} <- images_zip,
          reduce: init_params() do
        params -> update(params, imgs, labels)
      end

    save_model_params(trained_params, path, false)
    trained_params
  end

  def get_or_train_neural_network(:get, path, _, _) do
    load_model_params(path)
  end

  defp save_model_params(params, path, false), do: :not_saved
  defp save_model_params(params, path, true) do
    path = "lib/numeric_recognition/trained_parameters/#{path}"
    {l1, l2, l3, l4} = params

    File.write(path <> "l1", Nx.to_binary(l1))
    File.write(path <> "l2", Nx.to_binary(l2))
    File.write(path <> "l3", Nx.to_binary(l3))
    File.write(path <> "l4", Nx.to_binary(l4))
  end

  defp load_model_params(path) do
    path = "lib/numeric_recognition/trained_parameters/#{path}"

    l1 =
      File.read!(path <> "l1")
      |> Nx.from_binary({:f, 32})

    l2 =
      File.read!(path <> "l2")
      |> Nx.from_binary({:f, 32})

    l3 =
      File.read!(path <> "l3")
      |> Nx.from_binary({:f, 32})

    l4 =
      File.read!(path <> "l4")
      |> Nx.from_binary({:f, 32})

    {l1, l2, l3, l4}
  end
end
