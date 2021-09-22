import Config

config :exla, :clients, default: [platform: :cuda]

import_config "#{Mix.env}.exs"
