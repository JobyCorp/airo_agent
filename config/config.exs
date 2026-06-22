import Config

# Compile-time defaults. Host-specific paths, token, and bind address are set
# at runtime in runtime.exs (so the same release works on any serving host).
config :airo_agent,
  api_port: 4400,
  engines: [:llama_cpp]

config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id]
