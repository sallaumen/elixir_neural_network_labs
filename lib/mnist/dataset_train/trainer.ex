defmodule MNIST.DatasetTrain.Trainer do
  alias DatasetTrain.TrainerNumericalDefinition
  alias Dataset.Parser
  alias Dataset.Printer

  @chunk_size 0..29
  @variation_size 10

  def execute() do
    {images, labels} = Parser.get_data(:mnist)
    zip = Parser.zip_images_with_labels({images, labels})
    params = TrainerNumericalDefinition.get_or_train_neural_network(:train, "predictions_v1", zip, 3)
    result_array = predict_all(images, params)

    labels_array = Printer.print_and_get_labels(labels, @variation_size)
    IO.puts("Real output:")
    IO.inspect(result_array)
    Printer.print_and_get_success_percentage(labels_array, result_array)
    :ok
  end

  defp predict_all(images, params) do
    images
    |> Enum.flat_map(fn image_batch ->
      TrainerNumericalDefinition.predict(params, image_batch[@chunk_size])
      |> get_result_array()
    end)
  end

  defp get_result_array(predictions) do
    predictions
    |> Nx.to_flat_list()
    |> Enum.chunk_every(10)
    |> Enum.map(fn list -> list |> Enum.zip(0..9) end)
    |> Enum.map(fn list -> Enum.max(list) end)
    |> Enum.map(fn zip ->
      {_probability, index} = zip
      index
    end)
  end

end
