defmodule DatasetParse.Parser do
  def get_data(analyse_size, type, variation_size) do
    {train_images, train_labels} = get_images_and_results(type)
    print_expected_output(train_labels, analyse_size, variation_size)
    {train_images, train_labels}
  end

  defp get_images_and_results(:cifar10) do
    Scidata.CIFAR10.download(
      transform_images: &transform_images/1,
      transform_labels: &one_hot_labels/1
    )
  end

  defp get_images_and_results(:mnist) do
    Scidata.MNIST.download(
      transform_images: &transform_images/1,
      transform_labels: &one_hot_labels/1
    )
  end

  defp one_hot_labels({labels_binary, type, {n_labels}}) do
    labels_binary
    |> Nx.from_binary(type)
    |> Nx.reshape({n_labels, 1}, names: [:batch, :output])
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    |> Nx.to_batched_list(30)
  end

  defp transform_images({images_binary, type, {n_images, n_rows, n_cols}}) do
    shape_size = n_rows * n_cols

    images_binary
    |> Nx.from_binary(type)
    |> Nx.reshape({n_images, shape_size}, names: [:batch, :input])
    |> Nx.divide(255)
    |> Nx.to_batched_list(30)
  end

  defp print_expected_output(labels, analyse_size, variation_size) do
    slice =
      labels
      |> Enum.at(1)
      |> Nx.to_flat_list()
      |> Stream.chunk_every(variation_size)
      |> Stream.map(&Enum.with_index(&1))
      |> Stream.flat_map(&(&1 |> Enum.filter(fn {value, _index} -> value == 1 end)))
      |> Stream.map(fn {_, value} -> value end)
      |> Enum.slice(analyse_size)
    IO.puts("Expected output:")
    IO.inspect(slice)
  end
end
