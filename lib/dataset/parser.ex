defmodule Dataset.Parser do
  def get_working_batch_size(), do: 0..31

  def get_data(type) do
    get_images_and_results(type)
  end

  def zip_images_with_labels({images, labels}) do
    Enum.zip(images, labels)
    |> Enum.with_index()
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
    |> Nx.to_batched_list(32)
  end

  defp transform_images({images_binary, type, {n_images, n_rows, n_cols}}) do
    shape_size = n_rows * n_cols

    images_binary
    |> Nx.from_binary(type)
    |> Nx.reshape({n_images, shape_size}, names: [:batch, :input])
    |> Nx.divide(255)
    |> Nx.to_batched_list(32)
  end

  defp transform_images({images_binary, type, {n_images, _, n_rows, n_cols}}) do
    transform_images({images_binary, type, {n_images, n_rows, n_cols}})
  end
end
