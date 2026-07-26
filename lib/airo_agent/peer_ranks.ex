defmodule AiroAgent.PeerRanks do
  @moduledoc """
  Discovers **peer ranks of a multi-node load running on this host** — engine
  containers this agent did not start and does not supervise.

  A two-host tensor-parallel load is launched entirely from the head: the
  `vllm-slot` wrapper runs rank 0 locally and rank 1 on the worker over SSH, so
  both ranks share one supervised lifetime. The worker's own agent therefore has
  no `AiroAgent.Fleet` slot for the rank sitting on its GPU — its
  `AIRO_AGENT_SLOTS` is deliberately empty. Left alone, that host reports a
  nearly-full GPU and zero slots, and Airo cannot tell what is holding the VRAM.

  This module closes that gap by *observing* rather than owning. It polls the
  container runtime for containers the wrapper labelled (`airo.cluster.*`) and
  reports any whose rank is greater than 0 as a read-only slot. Rank 0 is
  excluded: on the head that container **is** a real Fleet slot and must not be
  reported twice.

  Discovered slots are never loadable — see `AiroAgent.Fleet.load/3`, which
  refuses a port held by a peer rank. Loading onto a GPU already carrying a rank
  would fail late and confusingly, after OOMing.

  Labels are the source of truth rather than argv or agent memory: they are
  written once by the component that spans both hosts, and they survive an agent
  restart on either side because they live on the container.
  """

  use GenServer

  require Logger

  alias AiroAgent.{Notifier, SlotInfo}
  alias AiroAgent.Fleet.Event

  @poll_ms 15_000
  @label_prefix "airo.cluster"

  def start_link(opts \\ []), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)

  @doc """
  Peer-rank slots currently on this host. `[]` when the process isn't running
  (a host with no container engine never starts it).
  """
  @spec slots() :: [SlotInfo.t()]
  def slots do
    case Process.whereis(__MODULE__) do
      nil -> []
      pid -> GenServer.call(pid, :slots)
    end
  end

  @doc "True when `port` is held by a discovered peer rank (so it can't be loaded)."
  @spec held?(pos_integer()) :: boolean()
  def held?(port), do: Enum.any?(slots(), &(&1.port == port))

  @doc false
  # Poll now instead of waiting for the timer (tests).
  def refresh, do: GenServer.call(__MODULE__, :refresh)

  @impl true
  def init(_opts) do
    {:ok, %{slots: %{}}, {:continue, :poll}}
  end

  @impl true
  def handle_continue(:poll, state), do: {:noreply, schedule(poll(state))}

  @impl true
  def handle_call(:slots, _from, state), do: {:reply, Map.values(state.slots), state}

  def handle_call(:refresh, _from, state), do: {:reply, :ok, poll(state)}

  @impl true
  def handle_info(:poll, state), do: {:noreply, schedule(poll(state))}

  def handle_info(_msg, state), do: {:noreply, state}

  # --- polling ---

  defp schedule(state) do
    Process.send_after(self(), :poll, poll_ms())
    state
  end

  # A rank appearing or disappearing is pushed to Airo as an event, so a peer
  # failure reaches the head's health immediately rather than at the next
  # register heartbeat.
  defp poll(state) do
    discovered = Map.new(discovery_module().discover(), &{&1.port, &1})

    Enum.each(Map.keys(discovered) -- Map.keys(state.slots), fn port ->
      Logger.info("peer rank up on slot :#{port} (#{discovered[port].cluster_id})")
      emit(discovered[port], :up)
    end)

    Enum.each(Map.keys(state.slots) -- Map.keys(discovered), fn port ->
      Logger.warning("peer rank gone from slot :#{port} (#{state.slots[port].cluster_id})")
      emit(state.slots[port], :down)
    end)

    %{state | slots: discovered}
  end

  @doc false
  # One `ps` call with the labels projected into the output — no per-container
  # inspect. Any runtime hiccup yields no peers rather than crashing the agent.
  def discover do
    case System.cmd(runtime(), ps_args(), stderr_to_stdout: true) do
      {out, 0} -> out |> String.split("\n", trim: true) |> Enum.flat_map(&parse_line/1)
      _ -> []
    end
  rescue
    _ -> []
  end

  defp ps_args do
    format =
      ~w(id rank size model port)
      |> Enum.map(&"{{.Label \"#{label_for(&1)}\"}}")
      |> Enum.join("\t")

    ["ps", "--filter", "label=#{@label_prefix}.id", "--format", format]
  end

  defp label_for("port"), do: "airo.slot.port"
  defp label_for(field), do: "#{@label_prefix}.#{field}"

  @doc false
  def parse_line(line) do
    case String.split(line, "\t") do
      [id, rank, size, model, port] -> build(id, rank, size, model, port)
      _ -> []
    end
  end

  # Rank 0 is the head's own container: on that host it is already a real Fleet
  # slot, so reporting it here would duplicate it.
  defp build(cluster_id, rank, size, model, port) do
    with {:ok, rank} when rank > 0 <- to_int(rank),
         {:ok, port} <- to_int(port),
         id when id != "" <- String.trim(cluster_id) do
      [
        %SlotInfo{
          port: port,
          base_url: base_url(port),
          status: :up,
          resident_model: presence(model),
          cluster_id: id,
          tp_rank: rank,
          tp_size: with({:ok, n} <- to_int(size), do: n, else: (_ -> nil))
        }
      ]
    else
      _ -> []
    end
  end

  # A peer rank serves nothing here — the head holds the API. Airo requires an
  # absolute http(s) URL on every provider, so one is reported and Airo marks the
  # slot `serves_api: false` rather than routing to it.
  defp base_url(port) do
    host = Application.get_env(:airo_agent, :advertise_host, "127.0.0.1")
    "http://#{host}:#{port}/v1"
  end

  defp emit(%SlotInfo{} = slot, type) do
    Notifier.publish(%Event{
      type: type,
      port: slot.port,
      resident_model: slot.resident_model,
      cluster_id: slot.cluster_id,
      tp_rank: slot.tp_rank,
      tp_size: slot.tp_size,
      reason: if(type == :down, do: "peer_rank_gone"),
      at: DateTime.utc_now()
    })
  end

  defp to_int(value) when is_binary(value) do
    case Integer.parse(String.trim(value)) do
      {int, ""} -> {:ok, int}
      _ -> :error
    end
  end

  defp to_int(_value), do: :error

  defp presence(value) when is_binary(value) do
    case String.trim(value) do
      "" -> nil
      trimmed -> trimmed
    end
  end

  defp presence(_value), do: nil

  defp runtime, do: Application.get_env(:airo_agent, :container_runtime, "podman")
  defp poll_ms, do: Application.get_env(:airo_agent, :peer_poll_ms, @poll_ms)

  # Swappable like :instance_module — lets tests drive the state machine without
  # a container runtime, since discovery is the only side-effecting part.
  defp discovery_module,
    do: Application.get_env(:airo_agent, :peer_discovery_module, __MODULE__)
end
