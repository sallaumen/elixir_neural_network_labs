import Config

exla_client =
  case System.get_env("EXLA_CLIENT", "host") do
    "cuda" -> :cuda
    "host" -> :host
    value -> raise "unsupported EXLA_CLIENT=#{inspect(value)} (expected host or cuda)"
  end

config :nx, :default_backend, {EXLA.Backend, client: exla_client}
config :nx, :default_defn_options, compiler: EXLA, client: exla_client

config :exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, preallocate: false],
  rocm: [platform: :rocm],
  tpu: [platform: :tpu]

import_config "#{Mix.env()}.exs"
