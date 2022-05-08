import Config

# config :exla, :clients, default: [platform: :cuda, preallocate: false]
config :nx, :default_defn_options, compiler: EXLA, client: :cuda

config :exla, :clients,
  host: [platform: :host],
  cuda: [platform: :cuda, preallocate: false],
  rocm: [platform: :rocm],
  tpu: [platform: :tpu]

import_config "#{Mix.env()}.exs"
