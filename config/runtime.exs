import Config

# Evaluated at boot on the serving host. Everything host-specific lives here.

hf_cache = System.get_env("AIRO_AGENT_MODEL_ROOT") || Path.expand("~/.cache/huggingface/hub")

# Engine location is host-specific — set via env in the systemd unit. Defaults
# stay neutral (binary resolved on PATH, no LD_LIBRARY_PATH) rather than baking
# in a build path.
llama_lib = System.get_env("LLAMA_CPP_LIB")
llama_bin = System.get_env("LLAMA_SERVER_BIN") || "llama-server"

token = System.get_env("AIRO_AGENT_TOKEN")

# Where Airo's /agent socket lives. Set it to push state over the channel
# (decision #3); unset ⇒ the agent logs events locally (loopback dev).
airo_socket_url = System.get_env("AIRO_SOCKET_URL")

# Stable identity for this serving host; should match the Airo Provider name.
host_id =
  System.get_env("AIRO_AGENT_HOST_ID") ||
    case :inet.gethostname() do
      {:ok, name} -> List.to_string(name)
      _ -> "unknown"
    end

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
  engine_bin: %{llama_cpp: llama_bin},
  # Channel push (decision #3): use the slipstream client when a socket URL is
  # set, else just log lifecycle events locally.
  host_id: host_id,
  airo_socket_url: airo_socket_url,
  notifier: if(airo_socket_url, do: AiroAgent.Notifier.Channel, else: AiroAgent.Notifier.Log)
