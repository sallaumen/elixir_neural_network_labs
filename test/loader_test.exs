defmodule Dataset.Training.LoaderTest do
  use ExUnit.Case

  alias Dataset.Training.Loader

  test "prepares MNIST images and one-hot labels in aligned batches" do
    images = {:binary.copy(<<255>>, 16 * 784), {:u, 8}, {16, 1, 28, 28}}
    labels = {:binary.copy(<<3>>, 16), {:u, 8}, {16}}
    {image_batches, label_batches} = Loader.prepare_dataset(:mnist, {images, labels})

    [image_batch] = Enum.to_list(image_batches)
    [label_batch] = Enum.to_list(label_batches)
    assert Nx.shape(image_batch) == {16, 784}
    assert Nx.shape(label_batch) == {16, 10}
    assert hd(Nx.to_flat_list(image_batch)) == 1.0
    assert Nx.to_flat_list(label_batch[0]) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  end

  test "prepares CIFAR-10 images in channels-first order" do
    images = {:binary.copy(<<128>>, 16 * 3 * 32 * 32), {:u, 8}, {16, 3, 32, 32}}
    labels = {:binary.copy(<<1>>, 16), {:u, 8}, {16}}
    {image_batches, label_batches} = Loader.prepare_dataset(:cifar10, {images, labels})

    [image_batch] = Enum.to_list(image_batches)
    [label_batch] = Enum.to_list(label_batches)
    assert Nx.shape(image_batch) == {16, 3, 32, 32}
    assert Nx.shape(label_batch) == {16, 10}
    assert_in_delta hd(Nx.to_flat_list(image_batch)), 128 / 255, 0.00001
  end
end
