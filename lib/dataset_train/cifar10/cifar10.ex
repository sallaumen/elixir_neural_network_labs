defmodule DatasetTrain.Cifar10 do
  require Axon
  alias Dataset.Loader

  @epochs 3

  def execute() do
    {train_images, train_labels} = Loader.get_dataset(:cifar10)

    model = create_model()
    train_model(model, train_images, train_labels)
  end

  def execute_and_test() do
    {train_images, train_labels} = Loader.get_dataset(:cifar10)

    model = create_model()
    trained_model = train_model(model, train_images, train_labels)

    test_model(model, trained_model, train_images, train_labels)
  end

  defp create_model() do
    model =
      Axon.input({nil, 3, 32, 32}, "input")
      |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu)
      |> Axon.batch_norm()
      |> Axon.max_pool(kernel_size: {2, 2})
      |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
      |> Axon.batch_norm()
      |> Axon.max_pool(kernel_size: {2, 2})
      |> Axon.flatten()
      |> Axon.dense(64, activation: :relu)
      |> Axon.dropout(rate: 0.5)
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
    |> Axon.Loop.run(Stream.zip(train_images, train_labels), %{}, epochs: @epochs, compiler: EXLA)
  end

  def test_model(model, final_training_state, test_images, test_labels) do
    IO.puts(" -> Testing model:")
    test_data = Stream.zip(test_images, test_labels)

    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(test_data, final_training_state, compiler: EXLA)

    IO.puts("\n")
  end
end
