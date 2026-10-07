defmodule Dataset.Training.MNIST do
  alias Dataset.Training.Loader

  @epochs 3

  def run_training() do
    {train_images, train_labels} = Loader.get_dataset(:mnist)

    model = create_model()
    train_model(model, train_images, train_labels)
  end

  def run_training_and_test() do
    {train_images, train_labels} = Loader.get_dataset(:mnist)

    model = create_model()
    trained_model = train_model(model, train_images, train_labels)

    {test_images, test_labels} = Loader.get_dataset(:mnist, :test)
    test_model(model, trained_model, test_images, test_labels)
  end

  def create_model() do
    model =
      Axon.input("input", shape: {nil, 784})
      |> Axon.dense(128, activation: :relu)
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
    |> Axon.Loop.run(Stream.zip(train_images, train_labels), Axon.ModelState.empty(), epochs: @epochs, compiler: EXLA)
  end

  defp test_model(model, final_training_state, test_images, test_labels) do
    IO.puts(" -> Testing model:")
    test_data = Stream.zip(test_images, test_labels)

    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(test_data, final_training_state.step_state.model_state, compiler: EXLA)

    IO.puts("\n")
  end
end
