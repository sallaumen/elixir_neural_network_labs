defmodule Dataset.Training.Cifar10.LegacyTrainer do
  alias Dataset.Training.NumericalDefinitions
  alias Dataset.Training.Parser
  alias Dataset.Training.Loader
  alias Dataset.Training.Printer

  @variation_size 10
  @epochs 3

  @deprecated "Use Dataset.Training.Cifar10.run_training/0 instead"
  def run_training() do
    {images, labels} = Loader.get_dataset(:cifar10)
    zip = Parser.zip_images_with_labels({images, labels})

    params = NumericalDefinitions.get_or_train_neural_network(:train, "predictions_v1", zip, @epochs)

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
      NumericalDefinitions.predict(params, image_batch[Parser.get_working_batch_size()])
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
