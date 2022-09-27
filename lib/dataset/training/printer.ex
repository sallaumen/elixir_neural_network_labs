defmodule Dataset.Training.Printer do
  def print_and_get_labels(labels, variation_size) do
    labels_array = get_labels_array(labels, variation_size)
    IO.puts("Expected output:")
    IO.inspect(labels_array)
  end

  def print_and_get_success_percentage(expected, real) do
    batch_size = Enum.count(expected)

    successes =
      0..batch_size
      |> Enum.reduce(0, fn current_index, acc ->
        value = Enum.at(expected, current_index) == Enum.at(real, current_index)
        acc + boolean_to_integer(value)
      end)

    IO.inspect("Success rate: #{successes * 100 / batch_size}")
  end

  defp boolean_to_integer(bool) do
    (bool && 1) || 0
  end

  defp get_labels_array(labels, variation_size) do
    labels
    |> Enum.flat_map(fn label_batch -> Nx.to_flat_list(label_batch) end)
    |> Stream.chunk_every(variation_size)
    |> Stream.map(&Enum.with_index(&1))
    |> Stream.flat_map(&(&1 |> Enum.filter(fn {value, _index} -> value == 1 end)))
    |> Stream.map(fn {_, value} -> value end)
    |> Enum.map(fn entry -> entry end)
  end
end
