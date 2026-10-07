defmodule Dataset.Training.Printer do
  def print_and_get_labels(labels, variation_size) do
    labels_array = get_labels_array(labels, variation_size)
    IO.puts("Expected output:")
    IO.inspect(labels_array)
  end

  def print_and_get_success_percentage(expected, actual) do
    if length(expected) != length(actual) do
      raise ArgumentError, "expected and actual labels must have the same length"
    end

    successes =
      expected
      |> Enum.zip(actual)
      |> Enum.count(fn {expected_label, actual_label} -> expected_label == actual_label end)

    percentage =
      case length(expected) do
        0 -> 0.0
        count -> successes * 100.0 / count
      end

    IO.puts("Success rate: #{Float.round(percentage, 2)}%")
    percentage
  end

  defp get_labels_array(labels, variation_size) do
    labels
    |> Enum.flat_map(&Nx.to_flat_list/1)
    |> Enum.chunk_every(variation_size)
    |> Enum.map(fn values -> Enum.find_index(values, &(&1 == 1)) end)
  end
end
