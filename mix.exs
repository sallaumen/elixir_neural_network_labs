defmodule ImageNeuralNetworkLabs.MixProject do
  use Mix.Project

  def project do
    [
      app: :image_neural_network_labs,
      version: "1.0.0",
      elixir: "~> 1.20.0",
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
      {:axon, "~> 0.9.0"},
      {:exla, "~> 1.0.0"},
      {:nx, "~> 1.0.0"},
      {:scidata, "~> 0.1.11"}
    ]
  end
end
