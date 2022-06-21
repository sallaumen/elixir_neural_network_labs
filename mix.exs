defmodule SimpleDataAnalysis.MixProject do
  use Mix.Project

  def project do
    [
      app: :image_neural_network_labs,
      version: "1.0.0",
      elixir: "~> 1.13.4",
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
      {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon"},
      {:exla, "~> 0.2"},
      {:nx, "~> 0.2.1"},
      {:scidata, "~> 0.1.6"}
    ]
  end
end
