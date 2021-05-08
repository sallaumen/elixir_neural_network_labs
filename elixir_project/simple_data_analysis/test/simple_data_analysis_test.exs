defmodule SimpleDataAnalysisTest do
  use ExUnit.Case
  doctest SimpleDataAnalysis

  test "greets the world" do
    assert SimpleDataAnalysis.hello() == :world
  end
end
