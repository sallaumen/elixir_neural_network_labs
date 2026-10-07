defmodule Dataset.Training.Loader do
  @batch_size 16

  def batch_size, do: @batch_size

  def get_dataset(dataset, split \\ :train) do
    IO.puts(" -> Downloading #{split} dataset")

    raw_data = download_dataset(dataset, split)
    prepare_dataset(dataset, raw_data)
  end

  def prepare_dataset(dataset, {raw_images, raw_labels}) do
    {transform_images(dataset, raw_images), transform_labels(raw_labels)}
  end

  defp download_dataset(:cifar10, :train), do: Scidata.CIFAR10.download()
  defp download_dataset(:cifar10, :test), do: Scidata.CIFAR10.download_test()
  defp download_dataset(:mnist, :train), do: Scidata.MNIST.download()
  defp download_dataset(:mnist, :test), do: Scidata.MNIST.download_test()

  defp transform_images(:cifar10, {bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 3, 32, 32})
    |> Nx.divide(255.0)
    |> Nx.to_batched(@batch_size)
  end

  defp transform_images(:mnist, {bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 784})
    |> Nx.divide(255.0)
    |> Nx.to_batched(@batch_size)
  end

  defp transform_labels({bin, type, _shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    |> Nx.to_batched(@batch_size)
  end
end
