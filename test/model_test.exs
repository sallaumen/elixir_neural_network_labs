defmodule Dataset.Training.ModelTest do
  use ExUnit.Case

  test "MNIST and CIFAR-10 models build, predict, and train on synthetic batches" do
    for {module, input_shape} <- [
          {Dataset.Training.MNIST, {2, 784}},
          {Dataset.Training.Cifar10, {2, 3, 32, 32}}
        ] do
      model = module.create_model()
      input = Nx.broadcast(0.0, input_shape)
      labels = Nx.tensor([0, 0]) |> Nx.new_axis(-1) |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
      {init, predict} = Axon.build(model)
      state = init.(input, Axon.ModelState.empty())
      assert Nx.shape(predict.(state, input)) == {2, 10}

      loop = model |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
      result = Axon.Loop.run(loop, [{input, labels}], Axon.ModelState.empty(), epochs: 1, compiler: EXLA)
      assert result.epoch == 1

      evaluator = model |> Axon.Loop.evaluator() |> Axon.Loop.metric(:accuracy, "Accuracy")

      evaluation =
        Axon.Loop.run(evaluator, [{input, labels}], result.step_state.model_state, compiler: EXLA)

      assert evaluation.epoch == 1
    end
  end
end
