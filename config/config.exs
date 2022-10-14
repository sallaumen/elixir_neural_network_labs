import Config

#config :nx, :default_backend, {EXLA.Backend, client: :cuda}
#config :nx, :default_defn_options, compiler: EXLA, client: :cuda
#
#config :exla, :clients,
#  host: [platform: :host],
#  cuda: [platform: :cuda, preallocate: false],
#  rocm: [platform: :rocm],
#  tpu: [platform: :tpu]

import_config "#{Mix.env()}.exs"
