defmodule DatasetTrain.MNIST do
  require Axon
  alias Dataset.Loader

  @epochs 3

  def execute() do
    {train_images, train_labels} = Loader.get_dataset(:mnist)

    model = create_model()
    final_training_state = train_model(model, train_images, train_labels)
    test_model(model, final_training_state, train_images, train_labels)
  end

  defp create_model() do
    model =
      Axon.input({nil, 784})
      |> Axon.dense(128, activation: :relu)
      |> Axon.dropout()
      |> Axon.dense(10, activation: :softmax)

    IO.puts(" -> Model:")
    IO.inspect(model)
    model
  end

  defp train_model(model, train_images, train_labels) do
    IO.puts(" -> Training:")

    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(train_images, train_labels), epochs: @epochs, compiler: EXLA)
  end

  defp test_model(model, final_training_state, test_images, test_labels) do
    IO.puts(" -> Testing model:")
    test_data = Stream.zip(test_images, test_labels)

    model
    |> Axon.Loop.evaluator(final_training_state)
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(test_data, compiler: EXLA)

    IO.puts("\n")
  end
end
