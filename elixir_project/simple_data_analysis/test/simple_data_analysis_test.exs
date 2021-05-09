defmodule SimpleDataAnalysisTest do
  use ExUnit.Case
  doctest SimpleDataAnalysis

  test "validates analyse return" do
    assert SimpleDataAnalysis.analyse() == :ok
  end
end
