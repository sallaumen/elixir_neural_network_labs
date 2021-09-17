defmodule SimpleDataAnalysis.MixProject do
  use Mix.Project

  def project do
    [
      app: :image_neural_network_labs,
      version: "0.1.0",
      elixir: "~> 1.11",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
      {:scidata, "~> 0.1.1"}
    ]
  end
end
