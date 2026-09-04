defmodule AiroAgent.Notifier.Channel do
  @moduledoc """
  `slipstream` client that connects to one Airo's `/agent` socket and pushes
  Fleet lifecycle state (decision #3). Airo is the server; this is the client.

  One process per configured airo endpoint (S26): the **controller**
  (`AIRO_SOCKET_URL`) and each **observer** (`AIRO_OBSERVER_SOCKET_URLS`). The
  role rides the socket URL (`role=`) and the `register` payload (`agent.role`),
  so each airo knows what it was granted. Every process is registered in
  `AiroAgent.Notifier.Registry` under its URI; `publish/1` fans a slot event out
  to all of them. Observers reconnect on a longer backoff — a laptop that is
  closed should not have every GPU host retrying it every ten seconds.

  Implements `AiroAgent.Notifier`: `publish/1` hands the slot event to every
  channel, which pushes it as a `"slot"` message. On join and every rejoin each
  pushes a full `"register"` (agent identity + all slots), so events dropped
  while disconnected self-heal — Airo reconciles the host from the register.

  Started only when at least one endpoint is configured (see `runtime.exs`);
  otherwise the agent uses `AiroAgent.Notifier.Log` and no channel is in the tree.
  """

  use Slipstream

  @behaviour AiroAgent.Notifier

  require Logger
  alias AiroAgent.{Fleet, SlotInfo}
  alias AiroAgent.Fleet.Event

  @registry AiroAgent.Notifier.Registry
  @roles [:controller, :observer]

  # Observers back off harder: the controller is prod and should be back within
  # seconds; an observer is often a workstation that is simply closed.
  @controller_backoff [1_000, 2_000, 5_000, 10_000]
  @observer_backoff [1_000, 5_000, 15_000, 60_000]

  def start_link(opts \\ []) do
    uri = opts[:uri] || Application.fetch_env!(:airo_agent, :airo_socket_url)
    role = Keyword.get(opts, :role, :controller)
    unless role in @roles, do: raise(ArgumentError, "unknown airo role #{inspect(role)}")

    name = Keyword.get(opts, :name, {:via, Registry, {@registry, uri}})
    Slipstream.start_link(__MODULE__, {config(uri, role, opts), role, uri}, name: name)
  end

  @doc "Every live channel client's pid — one per configured airo endpoint."
  def clients, do: Registry.select(@registry, [{{:_, :"$1", :_}, [], [:"$1"]}])

  @impl AiroAgent.Notifier
  def publish(%Event{} = event) do
    Enum.each(clients(), &send(&1, {:publish, event}))
    :ok
  end

  @impl Slipstream
  def init({config, role, uri}) do
    # Self-rescheduling heartbeat (set up once; pushes only while joined).
    Process.send_after(self(), :heartbeat, heartbeat_ms())
    socket = config |> connect!() |> assign(:role, role) |> assign(:airo, label(uri))
    {:ok, socket}
  end

  @impl Slipstream
  def handle_connect(socket) do
    Logger.info(
      "agent channel: connected to airo #{socket.assigns.airo} as #{socket.assigns.role}; joining #{topic()}"
    )

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
    payload = %{
      agent: agent_meta(socket.assigns.role),
      slots: Enum.map(Fleet.slots(), &SlotInfo.to_payload/1)
    }

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
      # Named `deployment_id` on the wire — see `AiroAgent.SlotInfo.to_payload/1`.
      deployment_id: e.cluster_id,
      tp_rank: e.tp_rank,
      tp_size: e.tp_size,
      status: e.type,
      reason: serialize_reason(e.reason),
      at: e.at
    }
  end

  defp heartbeat_ms, do: Application.get_env(:airo_agent, :heartbeat_ms, 10_000)

  defp serialize_reason(nil), do: nil
  defp serialize_reason(reason) when is_binary(reason), do: reason
  defp serialize_reason(reason), do: inspect(reason)

  defp agent_meta(role) do
    %{
      control_url: control_url(),
      version: to_string(Application.spec(:airo_agent, :vsn) || ""),
      gpu: AiroAgent.GPU.snapshot(),
      # What this airo is to us (S26). Airo persists it and gates load/unload.
      role: role
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

  defp config(uri, role, opts) do
    [
      # In test mode slipstream never dials, so the URI is taken as given.
      uri: if(opts[:test_mode?], do: uri, else: socket_uri(uri, role)),
      reconnect_after_msec:
        if(role == :observer, do: @observer_backoff, else: @controller_backoff),
      rejoin_after_msec: [1_000, 2_000, 5_000],
      test_mode?: Keyword.get(opts, :test_mode?, false)
    ]
  end

  @doc """
  Build `ws(s)://host[:port]/agent/websocket?host_id=..&role=..[&token=..]` from
  a configured socket URL — slipstream appends `vsn=2.0.0`. Accepts the
  socket-mount URL with or without `/websocket`. Public for the contract test.
  """
  def socket_uri(url, role) when is_binary(url) and role in @roles do
    uri = URI.parse(url)
    path = uri.path || "/agent"
    path = if String.ends_with?(path, "/websocket"), do: path, else: path <> "/websocket"

    query =
      (uri.query || "")
      |> URI.decode_query()
      |> Map.merge(Map.new([{"host_id", host_id()}, {"role", to_string(role)}] ++ token_param()))
      |> URI.encode_query()

    URI.to_string(%{uri | path: path, query: query})
  end

  # `host[:port]` of an endpoint, for log lines that must tell two airos apart.
  defp label(url) do
    case URI.parse(url) do
      %URI{host: host, port: port} when is_binary(host) and is_integer(port) -> "#{host}:#{port}"
      %URI{host: host} when is_binary(host) -> host
      _ -> url
    end
  end

  defp token_param do
    case Application.get_env(:airo_agent, :api_token) do
      token when is_binary(token) and token != "" -> [{"token", token}]
      _ -> []
    end
  end
end
