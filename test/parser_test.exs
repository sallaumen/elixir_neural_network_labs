defmodule Dataset.Training.ParserTest do
  use ExUnit.Case

  alias Dataset.Training.Parser

  test "legacy batch selection follows the loader batch size" do
    assert Parser.get_working_batch_size() == 0..(Dataset.Training.Loader.batch_size() - 1)
  end
end
