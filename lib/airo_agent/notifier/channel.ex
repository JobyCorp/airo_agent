defmodule AiroAgent.Notifier.Channel do
  @moduledoc """
  `slipstream` client that connects to Airo's `/agent` socket and pushes Fleet
  lifecycle state (decision #3). Airo is the server; this is the client.

  Implements `AiroAgent.Notifier`: `publish/1` hands the event to this process,
  which pushes it on the channel. On join and every rejoin it pushes a full
  snapshot, so events dropped while disconnected self-heal — Airo reconciles its
  view of the host from the snapshot (absent ⇒ down).

  Started only when `AIRO_SOCKET_URL` is configured (see `runtime.exs`); otherwise
  the agent uses `AiroAgent.Notifier.Log` and this process isn't in the tree.
  """

  use Slipstream

  @behaviour AiroAgent.Notifier

  require Logger
  alias AiroAgent.Fleet
  alias AiroAgent.Fleet.Event

  def start_link(_opts) do
    Slipstream.start_link(__MODULE__, config(), name: __MODULE__)
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
    {:ok, connect!(config)}
  end

  @impl Slipstream
  def handle_connect(socket) do
    Logger.info("agent channel: connected to airo; joining #{topic()}")
    {:ok, join(socket, topic())}
  end

  @impl Slipstream
  def handle_join(_topic, _reply, socket) do
    push_snapshot(socket)
    {:ok, socket}
  end

  # Airo asks for a fresh snapshot (e.g. after Airo restarts).
  @impl Slipstream
  def handle_message(_topic, "resync", _payload, socket) do
    push_snapshot(socket)
    {:ok, socket}
  end

  def handle_message(_topic, _event, _payload, socket), do: {:ok, socket}

  @impl Slipstream
  def handle_disconnect(_reason, socket) do
    {:ok, reconnect(socket)}
  end

  @impl Slipstream
  def handle_topic_close(_topic, _reason, socket) do
    {:ok, rejoin(socket, topic())}
  end

  # Fleet → publish/1 → here. Drop if not joined: the next snapshot reconciles.
  @impl Slipstream
  def handle_info({:publish, %Event{} = event}, socket) do
    if joined?(socket, topic()) do
      push(socket, topic(), "event", serialize_event(event))
    end

    {:noreply, socket}
  end

  # --- payloads ---

  defp push_snapshot(socket) do
    instances = Enum.map(Fleet.snapshot(), &Map.from_struct/1)
    push(socket, topic(), "snapshot", %{instances: instances, agent: agent_meta()})
  end

  defp serialize_event(%Event{} = e) do
    %{
      type: e.type,
      model_id: e.model_id,
      info: e.info && Map.from_struct(e.info),
      reason: serialize_reason(e.reason),
      at: e.at
    }
  end

  defp serialize_reason(nil), do: nil
  defp serialize_reason(reason) when is_binary(reason), do: reason
  defp serialize_reason(reason), do: inspect(reason)

  defp agent_meta do
    %{
      advertise_host: Application.get_env(:airo_agent, :advertise_host, "127.0.0.1"),
      version: to_string(Application.spec(:airo_agent, :vsn) || "")
    }
  end

  # --- config ---

  defp topic, do: "agent:" <> host_id()
  defp host_id, do: Application.get_env(:airo_agent, :host_id, "unknown")

  defp config do
    [
      uri: socket_uri(),
      reconnect_after_msec: [1_000, 2_000, 5_000, 10_000],
      rejoin_after_msec: [1_000, 2_000, 5_000]
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
