defmodule AiroAgent.Notifier.Channel do
  @moduledoc """
  `slipstream` client that connects to Airo's `/agent` socket and pushes Fleet
  lifecycle state (decision #3). Airo is the server; this is the client.

  Implements `AiroAgent.Notifier`: `publish/1` hands the slot event to this
  process, which pushes it as a `"slot"` message. On join and every rejoin it
  pushes a full `"register"` (agent identity + all slots), so events dropped
  while disconnected self-heal — Airo reconciles the host from the register.

  Started only when `AIRO_SOCKET_URL` is configured (see `runtime.exs`); otherwise
  the agent uses `AiroAgent.Notifier.Log` and this process isn't in the tree.
  """

  use Slipstream

  @behaviour AiroAgent.Notifier

  require Logger
  alias AiroAgent.Fleet
  alias AiroAgent.Fleet.Event

  def start_link(opts \\ []) do
    Slipstream.start_link(__MODULE__, config(opts), name: Keyword.get(opts, :name, __MODULE__))
  end

  @impl AiroAgent.Notifier
  def publish(%Event{} = event) do
    case Process.whereis(__MODULE__) do
      nil -> :ok
      pid -> send(pid, {:publish, event})
    end

    :ok
  end

  @impl Slipstream
  def init(config) do
    # Self-rescheduling heartbeat (set up once; pushes only while joined).
    Process.send_after(self(), :heartbeat, heartbeat_ms())
    {:ok, connect!(config)}
  end

  @impl Slipstream
  def handle_connect(socket) do
    Logger.info("agent channel: connected to airo; joining #{topic()}")
    {:ok, join(socket, topic())}
  end

  @impl Slipstream
  def handle_join(_topic, _reply, socket) do
    push_register(socket)
    {:ok, socket}
  end

  # Airo asks for a fresh registration (e.g. after Airo restarts).
  @impl Slipstream
  def handle_message(_topic, "resync", _payload, socket) do
    push_register(socket)
    {:ok, socket}
  end

  def handle_message(_topic, _event, _payload, socket), do: {:ok, socket}

  @impl Slipstream
  def handle_disconnect(_reason, socket) do
    # reconnect/1 returns {:ok, socket} | {:error, _} (NOT a bare socket), so
    # {:ok, reconnect(...)} double-wraps and crashes the callback dispatch — which
    # is exactly what took the agent down when Airo went away. Return it directly;
    # on error just hold the socket. Airo being down must never crash the agent.
    case reconnect(socket) do
      {:ok, socket} -> {:ok, socket}
      {:error, _reason} -> {:ok, socket}
    end
  end

  @impl Slipstream
  def handle_topic_close(topic, _reason, socket) do
    # rejoin/3 returns {:ok, socket} | {:error, :never_joined} (NOT a bare socket),
    # so wrapping it in another {:ok, _} crashes the callback dispatch. A *rejected*
    # join is "never joined" → fall back to a full reconnect (re-joins on backoff)
    # instead of tight-looping. Either branch returns a valid {:ok, %Socket{}}.
    case rejoin(socket, topic) do
      {:ok, socket} -> {:ok, socket}
      {:error, _reason} -> {:ok, reconnect(socket)}
    end
  end

  # Fleet → publish/1 → here. Drop if not joined: the next register reconciles.
  @impl Slipstream
  def handle_info({:publish, %Event{} = event}, socket) do
    if joined?(socket, topic()) do
      push(socket, topic(), "slot", slot_event(event))
    end

    {:noreply, socket}
  end

  # Periodic re-register: Airo never polls, so push fresh GPU telemetry + current
  # slot state on a timer. Re-pushing the full register also keeps deployment
  # health from decaying to :unknown between transitions.
  def handle_info(:heartbeat, socket) do
    if connected?(socket) and joined?(socket, topic()), do: push_register(socket)
    Process.send_after(self(), :heartbeat, heartbeat_ms())
    {:noreply, socket}
  end

  # --- payloads ---

  defp push_register(socket) do
    payload = %{agent: agent_meta(), slots: Enum.map(Fleet.slots(), &Map.from_struct/1)}
    push(socket, topic(), "register", payload)
  end

  defp slot_event(%Event{} = e) do
    %{
      port: e.port,
      resident_model: e.resident_model,
      revision: e.revision,
      ctx: e.ctx,
      parallel: e.parallel,
      ctx_total: e.ctx_total,
      engine_build: e.engine_build,
      status: e.type,
      reason: serialize_reason(e.reason),
      at: e.at
    }
  end

  defp heartbeat_ms, do: Application.get_env(:airo_agent, :heartbeat_ms, 10_000)

  defp serialize_reason(nil), do: nil
  defp serialize_reason(reason) when is_binary(reason), do: reason
  defp serialize_reason(reason), do: inspect(reason)

  defp agent_meta do
    %{
      control_url: control_url(),
      version: to_string(Application.spec(:airo_agent, :vsn) || ""),
      gpu: AiroAgent.GPU.snapshot()
    }
  end

  defp control_url do
    host = Application.get_env(:airo_agent, :advertise_host, "127.0.0.1")
    port = Application.get_env(:airo_agent, :api_port, 4400)
    "http://#{host}:#{port}"
  end

  # --- config ---

  defp topic, do: "agent:" <> host_id()
  defp host_id, do: Application.get_env(:airo_agent, :host_id, "unknown")

  defp config(opts) do
    [
      # opts[:uri] short-circuits socket_uri/0 so tests need no AIRO_SOCKET_URL.
      uri: opts[:uri] || socket_uri(),
      reconnect_after_msec: [1_000, 2_000, 5_000, 10_000],
      rejoin_after_msec: [1_000, 2_000, 5_000],
      test_mode?: Keyword.get(opts, :test_mode?, false)
    ]
  end

  # Build ws(s)://host[:port]/agent/websocket?host_id=..&token=.. — slipstream
  # appends vsn=2.0.0. Accepts the socket-mount URL with or without /websocket.
  defp socket_uri do
    uri = URI.parse(Application.fetch_env!(:airo_agent, :airo_socket_url))
    path = uri.path || "/agent"
    path = if String.ends_with?(path, "/websocket"), do: path, else: path <> "/websocket"

    query =
      (uri.query || "")
      |> URI.decode_query()
      |> Map.merge(Map.new([{"host_id", host_id()}] ++ token_param()))
      |> URI.encode_query()

    URI.to_string(%{uri | path: path, query: query})
  end

  defp token_param do
    case Application.get_env(:airo_agent, :api_token) do
      token when is_binary(token) and token != "" -> [{"token", token}]
      _ -> []
    end
  end
end
