defmodule Dataset.Training.PrinterTest do
  use ExUnit.Case
  import ExUnit.CaptureIO

  alias Dataset.Training.Printer

  test "extracts class indices from one-hot label batches" do
    labels = Nx.tensor([[0, 1, 0], [0, 0, 1]])

    output =
      capture_io(fn ->
        assert Printer.print_and_get_labels([labels], 3) == [1, 2]
      end)

    assert output =~ "Expected output:"
  end

  test "counts each prediction exactly once" do
    output =
      capture_io(fn ->
        assert Printer.print_and_get_success_percentage([1, 2], [1, 3]) == 50.0
      end)

    assert output =~ "Success rate: 50.0%"
  end

  test "handles empty predictions without dividing by zero" do
    capture_io(fn ->
      assert Printer.print_and_get_success_percentage([], []) == 0.0
    end)
  end

  test "rejects mismatched prediction lengths" do
    assert_raise ArgumentError, fn ->
      Printer.print_and_get_success_percentage([1], [])
    end
  end
end
