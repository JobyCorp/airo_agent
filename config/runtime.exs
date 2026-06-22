import Config

# Evaluated at boot on the serving host. Everything host-specific lives here.

hf_cache = System.get_env("AIRO_AGENT_MODEL_ROOT") || Path.expand("~/.cache/huggingface/hub")
llama_lib = System.get_env("LLAMA_CPP_LIB") || Path.expand("~/.unsloth/llama.cpp/build/bin")

llama_bin =
  System.get_env("LLAMA_SERVER_BIN") ||
    Path.expand("~/.unsloth/llama.cpp/build/bin/llama-server")

token = System.get_env("AIRO_AGENT_TOKEN")

config :airo_agent,
  api_port: String.to_integer(System.get_env("AIRO_AGENT_PORT") || "4400"),
  api_token: token,
  # Security default: loopback-only UNLESS a token is set. The agent can spawn
  # processes on the GPU host — never expose it to the LAN unauthenticated.
  bind_ip: if(token, do: {0, 0, 0, 0}, else: {127, 0, 0, 1}),
  model_roots: [hf_cache],
  llama_cpp_lib_path: llama_lib,
  engine_bin: %{llama_cpp: llama_bin}
