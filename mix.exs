defmodule SimpleDataAnalysis.MixProject do
  use Mix.Project

  def project do
    [
      app: :image_neural_network_labs,
      version: "0.1.0",
      elixir: "~> 1.12.3",
      start_permanent: Mix.env() == :dev,
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
      #      {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
      #      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
      #      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
      #      {:scidata, "~> 0.1.1"}
#      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
#      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
#      {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon"},
#      {:scidata, "~> 0.1.5"}
      {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon"},
      {:exla, github: "elixir-nx/exla", sparse: "exla"},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
      {:scidata, "~> 0.1.3"}
    ]
  end
end
