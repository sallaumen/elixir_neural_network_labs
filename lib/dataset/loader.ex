defmodule Dataset.Loader do
  def get_dataset(algorithm) do
    IO.puts(" -> Downloading dataset")
    {raw_train_images, raw_train_labels} = download_dataset(algorithm)
    train_images = transform_images(algorithm, raw_train_images)
    train_labels = transform_labels(algorithm, raw_train_labels)

    {train_images, train_labels}
  end

  defp download_dataset(:cifar10), do: Scidata.CIFAR10.download()
  defp download_dataset(:mnist), do: Scidata.MNIST.download()

  defp transform_images(:cifar10, {bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 3, 32, 32})
    |> Nx.divide(255.0)
    |> Nx.to_batched_list(32)
  end

  defp transform_images(:mnist, {bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 784})
    |> Nx.divide(255.0)
    |> Nx.to_batched_list(32)
  end

  defp transform_labels(:cifar10, {bin, type, _}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    |> Nx.to_batched_list(32)
  end

  defp transform_labels(:mnist, {bin, type, _}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    |> Nx.to_batched_list(32)
  end
end
