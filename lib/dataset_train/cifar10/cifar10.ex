defmodule DatasetTrain.Cifar10 do
  require Axon

  @epochs 1

  def execute() do
    {train_images, train_labels} = load_algorithm_dataset()

    model = create_model()
    final_training_state = train_model(model, train_images, train_labels)
    test_model(model, final_training_state, train_images, train_labels)
  end

  defp load_algorithm_dataset() do
    IO.puts(" -> Downloading dataset")
    {raw_train_images, raw_train_labels} = Scidata.CIFAR10.download()
    train_images = transform_images(raw_train_images)
    train_labels = transform_labels(raw_train_labels)

    {train_images, train_labels}
  end

  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 3, 32, 32})
    |> Nx.divide(255.0)
    |> Nx.to_batched_list(32)
  end

  defp transform_labels({bin, type, _}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    |> Nx.to_batched_list(32)
  end

  defp create_model() do
    model =
      Axon.input({nil, 3, 32, 32})
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
