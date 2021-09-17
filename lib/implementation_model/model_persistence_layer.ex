defmodule ImplementationModel.ModelPersistenceLayer do

  @base_path "lib/implementation_model/models/"

  def save_model_params(_params, _path, :skip), do: :not_saved
  def save_model_params(params, path, :run) do
    path = @base_path <> "#{path}"
    {l1, l2, l3, l4} = params

    File.write(path <> "l1", Nx.to_binary(l1))
    File.write(path <> "l2", Nx.to_binary(l2))
    File.write(path <> "l3", Nx.to_binary(l3))
    File.write(path <> "l4", Nx.to_binary(l4))
  end

  def load_model_params(path) do
    path = @base_path <> "#{path}"

    l1 =
      File.read!(path <> "l1")
      |> Nx.from_binary({:f, 32})

    l2 =
      File.read!(path <> "l2")
      |> Nx.from_binary({:f, 32})

    l3 =
      File.read!(path <> "l3")
      |> Nx.from_binary({:f, 32})

    l4 =
      File.read!(path <> "l4")
      |> Nx.from_binary({:f, 32})

    {l1, l2, l3, l4}
  end

end
