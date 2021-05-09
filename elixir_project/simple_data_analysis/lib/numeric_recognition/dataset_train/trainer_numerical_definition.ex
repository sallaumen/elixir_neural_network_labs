defmodule NumericRecognition.DatasetTrain.TrainerNumericalDefinition do
  import Nx.Defn

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
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> softmax()
  end

  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t), axes: [:output], keep_axes: true)
  end

  defn loss({w1, b1, w2, b2}, images, labels) do
    predictions = predict({w1, b1, w2, b2}, images)
    -Nx.sum(Nx.mean(Nx.log(predictions) * labels, axes: [:output]))
  end

  defn update({w1, b1, w2, b2} = params, images, labels) do
    {grad_w1, grad_b1, grad_w2, grad_b2} =
      grad(params, loss(params, images, labels))

    {w1 - grad_w1 * 0.01, b1 - grad_b1 * 0.01,
      w2 - grad_w2 * 0.01, b2 - grad_b2 * 0.01}
  end

end