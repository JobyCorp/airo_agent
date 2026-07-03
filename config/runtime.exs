import Config

# Evaluated at boot on the serving host. Everything host-specific lives here.

hf_cache = System.get_env("AIRO_AGENT_MODEL_ROOT") || Path.expand("~/.cache/huggingface/hub")

# Engine location is host-specific — set via env in the systemd unit. Defaults
# stay neutral (binary resolved on PATH, no LD_LIBRARY_PATH) rather than baking
# in a build path.
llama_lib = System.get_env("LLAMA_CPP_LIB")
llama_bin = System.get_env("LLAMA_SERVER_BIN") || "llama-server"

# vLLM engine: spawned via the podman wrapper shipped in priv/engine/vllm-slot
# (override with VLLM_SLOT_BIN). VLLM_IMAGE is the container image the wrapper
# runs — host-specific, set on vLLM hosts only.
vllm_slot_bin =
  System.get_env("VLLM_SLOT_BIN") ||
    case :code.priv_dir(:airo_agent) do
      {:error, _} -> "vllm-slot"
      dir -> Path.join([List.to_string(dir), "engine", "vllm-slot"])
    end

vllm_image = System.get_env("VLLM_IMAGE")

# Host-level vLLM tuning argv, applied when a load profile brings no extra_argv
# of its own (a profile's extra_argv replaces it wholesale). For per-card facts
# a Deployment shouldn't have to repeat, e.g. a 16 GB card that only fits with
# "--enforce-eager --max-num-batched-tokens 2048".
vllm_extra_argv =
  case System.get_env("AIRO_VLLM_EXTRA_ARGS") do
    nil -> nil
    args -> String.split(args, " ", trim: true)
  end

# Container runtime for container-based engines (vLLM). podman (default) | docker.
# The vllm-slot wrapper reads the same env var; this exposes it to the agent for
# the boot-time orphan sweep (see AiroAgent.Engine.Vllm.reap_orphans/0).
container_runtime = System.get_env("AIRO_CONTAINER_RUNTIME") || "podman"

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
# Routable address Airo reaches this host on. Agents are distributed across the
# network, so we must NOT advertise loopback (Airo can't reach 127.0.0.1 on a
# different box). Detect the primary LAN IP — the source IP of the default route
# (opens a UDP socket but sends no packets). Override with AIRO_AGENT_ADVERTISE_HOST
# (e.g. a stable FQDN); falls back to loopback only when there's no routable IP.
detect_primary_ip = fn ->
  with {:ok, sock} <- :gen_udp.open(0),
       :ok <- :gen_udp.connect(sock, {1, 1, 1, 1}, 9),
       {:ok, {addr, _port}} <- :inet.sockname(sock) do
    :gen_udp.close(sock)

    case addr do
      {127, _, _, _} -> nil
      _ -> addr |> :inet.ntoa() |> to_string()
    end
  else
    _ -> nil
  end
end

advertise_host =
  System.get_env("AIRO_AGENT_ADVERTISE_HOST") || detect_primary_ip.() || "127.0.0.1"

exposed? = advertise_host not in ["127.0.0.1", "localhost", "::1"]

# Serving slots (Model 2): the static ports this host exposes, one resident
# model each. Concurrency = more slots. Declared up front so each is a durable,
# real serving endpoint. Default: a single slot.
slots =
  (System.get_env("AIRO_AGENT_SLOTS") || "8081")
  |> String.split(",", trim: true)
  |> Enum.map(&(&1 |> String.trim() |> String.to_integer()))

# Which engine adapters this host runs (Model 2 is per-host: a llama.cpp box and
# a vLLM box are different hosts). CSV in AIRO_AGENT_ENGINES; unknown names are
# dropped here and again in Engine.inventory_all/1 (belt and suspenders). Default
# is llama.cpp, so existing hosts need no new env.
known_engines = %{"llama_cpp" => :llama_cpp, "vllm" => :vllm}

engines =
  case System.get_env("AIRO_AGENT_ENGINES") do
    blank when blank in [nil, ""] ->
      [:llama_cpp]

    csv ->
      case csv
           |> String.split(",", trim: true)
           |> Enum.map(&known_engines[String.trim(&1)])
           |> Enum.reject(&is_nil/1) do
        [] -> [:llama_cpp]
        list -> list
      end
  end

# In test, bind an OS-assigned port (0) so the test app's control API never
# clashes with a running agent service holding 4400.
default_port = if config_env() == :test, do: "0", else: "4400"

config :airo_agent,
  api_port: String.to_integer(System.get_env("AIRO_AGENT_PORT") || default_port),
  api_token: token,
  # Control-plane bind: LAN when exposed, else loopback. Independent of the token.
  bind_ip: if(exposed?, do: {0, 0, 0, 0}, else: {127, 0, 0, 1}),
  # Routable host stamped into each slot's base_url + the agent control_url, so a
  # remote Airo can reach this host's engines and control API.
  advertise_host: advertise_host,
  # The engine's own --host: 0.0.0.0 when exposed (matches vllm/ollama), else loopback.
  engine_bind_host: if(exposed?, do: "0.0.0.0", else: "127.0.0.1"),
  # Static serving slots (ports), one resident model each.
  slots: slots,
  model_roots: [hf_cache],
  llama_cpp_lib_path: llama_lib,
  engine_bin: %{llama_cpp: llama_bin, vllm: vllm_slot_bin},
  # Container image the vllm-slot wrapper runs (vLLM hosts only; nil otherwise).
  vllm_image: vllm_image,
  vllm_extra_argv: vllm_extra_argv,
  container_runtime: container_runtime,
  # Engine adapters active on this host (per-host; see AIRO_AGENT_ENGINES above).
  engines: engines,
  # Channel push (decision #3): use the slipstream client when a socket URL is
  # set, else just log lifecycle events locally.
  host_id: host_id,
  airo_socket_url: airo_socket_url,
  notifier: if(airo_socket_url, do: AiroAgent.Notifier.Channel, else: AiroAgent.Notifier.Log)
