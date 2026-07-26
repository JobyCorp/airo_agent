defmodule AiroAgent.Fleet do
  @moduledoc """
  Owner of this host's serving **slots** (Model 2). A slot is a stable port that
  holds at most one resident model. Airo decides *placement* (which model in
  which slot, what to evict); the agent *executes* load/unload and pushes slot
  state — Airo never polls.

  `load/3` makes a model resident in a slot, **swapping out** whatever is there;
  `unload/1` frees a slot; `slots/0` is the level state (for channel registration
  and `GET /slots`). Transitions are emitted as `AiroAgent.Fleet.Event`s.

  Crash vs. clean teardown is distinguished by an intent flag (`unload`) and by
  demonitoring on a swap, so a deliberate eviction is never reported as a crash.
  Each slot's engine is a supervised, crash-isolated OS process.
  """

  use GenServer
  require Logger

  alias AiroAgent.{Engine, Instances, ModelRef, Notifier, PeerRanks, SlotInfo}
  alias AiroAgent.Fleet.Event

  def start_link(_), do: GenServer.start_link(__MODULE__, :ok, name: __MODULE__)

  @spec load(ModelRef.t(), pos_integer(), map()) :: {:ok, SlotInfo.t()} | {:error, term()}
  def load(%ModelRef{} = model, port, profile \\ %{}) when is_integer(port),
    do: GenServer.call(__MODULE__, {:load, model, port, profile}, 60_000)

  @spec unload(pos_integer()) :: :ok | {:error, term()}
  def unload(port) when is_integer(port), do: GenServer.call(__MODULE__, {:unload, port})

  @spec slots() :: [SlotInfo.t()]
  def slots, do: GenServer.call(__MODULE__, :slots)

  @doc false
  # Called by an Instance when readiness passes, with engine runtime props.
  def mark_up(pid, props \\ %{}) when is_pid(pid),
    do: GenServer.cast(__MODULE__, {:mark_up, pid, props})

  @impl true
  def init(:ok) do
    slots = Map.new(configured_slots(), &{&1, empty_slot(&1)})
    {:ok, %{slots: slots, intent: MapSet.new()}}
  end

  @impl true
  def handle_call({:load, %ModelRef{} = model, port, profile}, _from, state) do
    cond do
      # A rank of someone else's multi-node load is on this port. It isn't ours to
      # evict, and its GPU is already committed — loading here would OOM late.
      PeerRanks.held?(port) ->
        {:reply, {:error, :peer_rank_resident}, state}

      true ->
        do_load(model, port, profile, state)
    end
  end

  def handle_call({:unload, port}, _from, state) do
    if PeerRanks.held?(port) do
      {:reply, {:error, :peer_rank_resident}, state}
    else
      do_unload(port, state)
    end
  end

  def handle_call(:slots, _from, state) do
    # Discovered peer ranks are reported alongside this host's own slots so Airo
    # can see what is holding the GPU, but they are not part of `state.slots` —
    # nothing here owns or supervises them.
    own = state.slots |> Map.values() |> Enum.map(&slot_info/1)
    {:reply, own ++ PeerRanks.slots(), state}
  end

  defp do_load(%ModelRef{} = model, port, profile, state) do
    case Map.fetch(state.slots, port) do
      :error ->
        {:reply, {:error, :unknown_slot}, state}

      {:ok, %{status: :up, model: %ModelRef{id: id}, profile: current} = slot}
      when id == model.id and current == profile ->
        # Already resident and serving with the same profile — idempotent. A
        # changed profile (e.g. a new context size) is NOT idempotent: it falls
        # through to the reload branch below so the engine relaunches.
        {:reply, {:ok, slot_info(slot)}, state}

      {:ok, slot} ->
        # Evict whatever's resident (swap or reconfigure), then launch on the
        # now-free port with the requested profile.
        if running?(slot) do
          free_running(slot)
          await_port_free(port)
        end

        case start_engine(model, port, profile) do
          {:ok, started} ->
            state = put_in(state.slots[port], started)
            emit(started, :loading, nil)
            {:reply, {:ok, slot_info(started)}, state}

          {:error, _} = err ->
            {:reply, err, put_in(state.slots[port], empty_slot(port))}
        end
    end
  end

  defp do_unload(port, state) do
    case Map.fetch(state.slots, port) do
      {:ok, slot} when slot.pid != nil ->
        # Intent BEFORE terminate so the monitored :DOWN reads as :unloaded.
        state = update_in(state.intent, &MapSet.put(&1, port))
        DynamicSupervisor.terminate_child(Instances, slot.pid)
        {:reply, :ok, state}

      {:ok, _empty} ->
        {:reply, :ok, state}

      :error ->
        {:reply, {:error, :unknown_slot}, state}
    end
  end

  @impl true
  def handle_cast({:mark_up, pid, props}, state) do
    case find_by_pid(state, pid) do
      {port, %{status: :loading} = slot} ->
        slot = %{slot | status: :up, props: props}
        Logger.info("slot :#{port} up: #{model_id(slot)}")
        emit(slot, :up, nil)
        {:noreply, put_in(state.slots[port], slot)}

      _ ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    case find_by_ref(state, ref) do
      {port, slot} ->
        expected? = MapSet.member?(state.intent, port)

        state = %{
          state
          | slots: Map.put(state.slots, port, empty_slot(port)),
            intent: MapSet.delete(state.intent, port)
        }

        cond do
          expected? ->
            emit(slot, :unloaded, nil)

          slot.status == :loading ->
            Logger.warning("slot :#{port} failed to start #{model_id(slot)}: #{inspect(reason)}")
            emit(slot, :failed, reason)

          true ->
            Logger.warning("slot :#{port} down #{model_id(slot)}: #{inspect(reason)}")
            emit(slot, :down, reason)
        end

        {:noreply, state}

      nil ->
        {:noreply, state}
    end
  end

  # --- internals ---

  defp running?(slot), do: is_pid(slot.pid)

  # Swap teardown: stop watching the resident engine (so its :DOWN doesn't reset
  # the slot we're about to repopulate) and terminate it. terminate_child is
  # synchronous; a listening port frees on the process's exit.
  defp free_running(slot) do
    Process.demonitor(slot.ref, [:flush])
    DynamicSupervisor.terminate_child(Instances, slot.pid)
    :ok
  end

  defp start_engine(%ModelRef{} = model, port, profile) do
    spec = %{model: model, profile: profile, port: port}

    case DynamicSupervisor.start_child(Instances, {instance_module(), spec}) do
      {:ok, pid} ->
        ref = Process.monitor(pid)

        {:ok,
         %{
           port: port,
           status: :loading,
           model: model,
           profile: profile,
           pid: pid,
           ref: ref,
           props: %{},
           started_at: DateTime.utc_now()
         }}

      {:error, _} = err ->
        err
    end
  end

  defp empty_slot(port),
    do: %{
      port: port,
      status: :empty,
      model: nil,
      profile: %{},
      pid: nil,
      ref: nil,
      props: %{},
      started_at: nil
    }

  defp slot_info(slot) do
    host = Application.get_env(:airo_agent, :advertise_host, "127.0.0.1")
    props = slot.props || %{}
    cluster = cluster_info(slot)

    %SlotInfo{
      port: slot.port,
      base_url: "http://#{host}:#{slot.port}/v1",
      status: slot.status,
      resident_model: slot.model && slot.model.id,
      revision: slot.model && slot.model.revision,
      ctx: props[:ctx],
      parallel: props[:parallel],
      ctx_total: props[:ctx_total],
      engine_build: props[:engine_build],
      cluster_id: cluster[:cluster_id],
      tp_rank: cluster[:tp_rank],
      tp_size: cluster[:tp_size],
      # Effective profile once :up; fall back to the requested profile while
      # :loading (before the engine reports its resolved config).
      profile: props[:resolved_profile] || slot.profile,
      started_at: slot.started_at
    }
  end

  # This host's rank in a multi-node load, from the engine adapter (only engines
  # that can span hosts implement it). Read from the *effective* profile so a
  # default-supplied `nnodes` isn't missed. `%{}` for an ordinary slot.
  defp cluster_info(%{model: %ModelRef{engine: engine}} = slot) do
    adapter = Engine.adapter(engine)
    props = slot.props || %{}
    profile = props[:resolved_profile] || slot.profile || %{}

    if function_exported?(adapter, :cluster_info, 2) do
      adapter.cluster_info(profile, slot.port) || %{}
    else
      %{}
    end
  rescue
    # An unknown engine must never sink the slot listing.
    _ -> %{}
  end

  defp cluster_info(_empty_slot), do: %{}

  defp emit(slot, type, reason) do
    props = slot.props || %{}

    Notifier.publish(%Event{
      type: type,
      port: slot.port,
      resident_model: slot.model && slot.model.id,
      revision: slot.model && slot.model.revision,
      ctx: props[:ctx],
      parallel: props[:parallel],
      ctx_total: props[:ctx_total],
      engine_build: props[:engine_build],
      reason: reason,
      at: DateTime.utc_now()
    })
  end

  defp model_id(%{model: %ModelRef{id: id}}), do: id
  defp model_id(_), do: "(none)"

  defp find_by_pid(state, pid), do: Enum.find(state.slots, fn {_p, s} -> s.pid == pid end)
  defp find_by_ref(state, ref), do: Enum.find(state.slots, fn {_p, s} -> s.ref == ref end)

  defp instance_module, do: Application.get_env(:airo_agent, :instance_module, AiroAgent.Instance)

  defp configured_slots, do: Application.get_env(:airo_agent, :slots, [])

  # Poll until the (just-freed) port stops accepting connections, so a swap can
  # rebind it. No-op when nothing is listening (e.g. test fakes). The default
  # ceiling tracks Instance's SIGTERM→SIGKILL grace (30 s): a gracefully-exiting
  # engine may hold the port for several seconds before the hard kill lands.
  # Env-tunable so tests (fakes never free a really-occupied port) stay fast.
  defp await_port_free(port),
    do: await_port_free(port, Application.get_env(:airo_agent, :port_free_tries, 150))

  defp await_port_free(_port, 0), do: :ok

  defp await_port_free(port, tries) do
    case :gen_tcp.connect(~c"127.0.0.1", port, [], 100) do
      {:error, _} ->
        :ok

      {:ok, sock} ->
        :gen_tcp.close(sock)
        Process.sleep(100)
        await_port_free(port, tries - 1)
    end
  end
end
