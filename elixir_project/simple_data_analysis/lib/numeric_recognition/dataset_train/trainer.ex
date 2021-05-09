defmodule NumericRecognition.DatasetTrain.Trainer do
  alias NumericRecognition.DatasetTrain.TrainerNumericalDefinition
  alias NumericRecognition.DatasetParser.Parser

  def execute() do
    train_images = File.read!("dataset/train-images-idx3-ubyte.gz") |> :zlib.gunzip()
    expected_result = File.read!("dataset/train-labels-idx1-ubyte.gz") |> :zlib.gunzip()
    {images, labels} = Parser.get_data(train_images, expected_result)

    [fbi | _] = images
    TrainerNumericalDefinition.init_params()
    |> TrainerNumericalDefinition.predict(fbi[0..2])

    zip = Enum.zip(images, labels)
      |> Enum.with_index()
    params =
      for e <- 1..1, {{images, labels}, b} <- zip, reduce: TrainerNumericalDefinition.init_params() do
        params ->
          IO.puts "epoch #{e}, batch #{b}"
          TrainerNumericalDefinition.update(params, images, labels)
      end

#    save_model_params(params)
#    load_model_params()
    TrainerNumericalDefinition.predict(params, fbi[0..2])
  end

  def save_model_params(params) do
    "lib/animal_recognition/trained_parameters/prediction_v1"
    |> File.write(:erlang.term_to_binary(params))
  end

  def load_model_params() do
    "lib/animal_recognition/trained_parameters/prediction_v1"
    |> File.read!
    |> :erlang.binary_to_term()
  end

end
