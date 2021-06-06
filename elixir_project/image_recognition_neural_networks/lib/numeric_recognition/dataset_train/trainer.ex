defmodule NumericRecognition.DatasetTrain.Trainer do
  alias DatasetTrain.TrainerNumericalDefinition
  alias DatasetParse.Parser

  def execute() do
    analyse_size = 0..29
    {images, labels} = Parser.get_data(analyse_size, :mnist, 10)
    zip = zip_images_with_labels({images, labels})

    params =
      TrainerNumericalDefinition.get_or_train_neural_network(:train, "predictions_v1", zip, 1)

    [fbi | _] = images

    TrainerNumericalDefinition.predict(params, fbi[analyse_size])
    |> to_result_output()

    :ok
  end

  defp zip_images_with_labels({images, labels}) do
    Enum.zip(images, labels)
    |> Enum.with_index()
  end

  defp to_result_output(predictions) do
    result_array =
      predictions
      |> Nx.to_flat_list()
      |> Enum.chunk_every(10)
      |> Enum.map(fn list -> list |> Enum.zip(0..9) end)
      |> Enum.map(fn list -> Enum.max(list) end)
      |> Enum.map(fn zip ->
        {_probability, index} = zip
        index
      end)

    IO.puts("Real output:")
    IO.inspect(result_array)
  end
end
