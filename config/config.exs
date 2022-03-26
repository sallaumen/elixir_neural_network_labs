import Config

#config :exla, :clients, default: [platform: :cuda]
config :nx, :default_defn_options, [compiler: EXLA, client: :cuda]

import_config "#{Mix.env()}.exs"
