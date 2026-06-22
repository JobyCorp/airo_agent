import Config

# Evaluated at boot on the serving host. Everything host-specific lives here.

hf_cache = System.get_env("AIRO_AGENT_MODEL_ROOT") || Path.expand("~/.cache/huggingface/hub")
llama_lib = System.get_env("LLAMA_CPP_LIB") || Path.expand("~/.unsloth/llama.cpp/build/bin")

llama_bin =
  System.get_env("LLAMA_SERVER_BIN") ||
    Path.expand("~/.unsloth/llama.cpp/build/bin/llama-server")

token = System.get_env("AIRO_AGENT_TOKEN")

# Exposure is EXPLICIT and decoupled from the token: set AIRO_AGENT_ADVERTISE_HOST
# to the host's LAN IP to serve the fleet; leave it unset for loopback-only dev.
# The token, when set, is OPTIONAL bearer auth layered on top — no longer the gate.
advertise_host = System.get_env("AIRO_AGENT_ADVERTISE_HOST") || "127.0.0.1"
exposed? = advertise_host not in ["127.0.0.1", "localhost", "::1"]

config :airo_agent,
  api_port: String.to_integer(System.get_env("AIRO_AGENT_PORT") || "4400"),
  api_token: token,
  # Control-plane bind: LAN when exposed, else loopback. Independent of the token.
  bind_ip: if(exposed?, do: {0, 0, 0, 0}, else: {127, 0, 0, 1}),
  # Host stamped into InstanceInfo.base_url so off-box Airo can reach the engine
  # (engine is always co-located with the agent; only the port is dynamic).
  advertise_host: advertise_host,
  # The engine's own --host: 0.0.0.0 when exposed (matches vllm/ollama), else loopback.
  engine_bind_host: if(exposed?, do: "0.0.0.0", else: "127.0.0.1"),
  model_roots: [hf_cache],
  llama_cpp_lib_path: llama_lib,
  engine_bin: %{llama_cpp: llama_bin}
