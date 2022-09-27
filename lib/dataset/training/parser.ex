defmodule Dataset.Training.Parser do
  def get_working_batch_size(), do: 0..31

  def zip_images_with_labels({images, labels}) do
    Enum.zip(images, labels)
    |> Enum.with_index()
  end
end
