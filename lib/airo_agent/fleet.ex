defmodule AiroAgent.Fleet do
  @moduledoc """
  Canonical, in-memory owner of running engine instances and the single emitter
  of lifecycle events.

  Airo never polls; the agent PUSHES state on real change via `AiroAgent.Notifier`.
  `load/2` and `unload/1` are synchronous control calls that return the current
  `InstanceInfo` immediately (status `:loading`); the `:up`/`:down`/`:failed`/
  `:unloaded` transitions that follow are emitted as `AiroAgent.Fleet.Event`s.

  Durable truth here is *the set of running instances* (`running/0`). A crash is
  an EVENT (edge), not retained state: it removes the instance from the set and
  emits `:down`. Lost pushes self-heal because Airo reconciles from the next
  snapshot — absent ⇒ down.

  Crash vs. clean unload is distinguished by an explicit intent flag, NOT by the
  exit reason: `unload/1` records intent before terminating, so the monitored
  `:DOWN` is classified `:unloaded` (expected) rather than `:down`/`:failed`.
  """

  use GenServer
  require Logger

  alias AiroAgent.{Instances, InstanceInfo, ModelRef, Notifier}
  alias AiroAgent.Fleet.Event

  def start_link(_), do: GenServer.start_link(__MODULE__, :ok, name: __MODULE__)

  @spec load(ModelRef.t(), map()) :: {:ok, InstanceInfo.t()} | {:error, term()}
  def load(%ModelRef{} = model, profile \\ %{}),
    do: GenServer.call(__MODULE__, {:load, model, profile})

  @spec unload(String.t()) :: :ok | {:error, :not_running}
  def unload(model_id) when is_binary(model_id),
    do: GenServer.call(__MODULE__, {:unload, model_id})

  @spec running() :: [InstanceInfo.t()]
  def running, do: GenServer.call(__MODULE__, :running)

  @doc "Full level state for snapshot-on-(re)connect (decision #3)."
  @spec snapshot() :: [InstanceInfo.t()]
  def snapshot, do: running()

  @doc false
  # Called by an Instance the first time its readiness predicate passes.
  def mark_up(pid) when is_pid(pid), do: GenServer.cast(__MODULE__, {:mark_up, pid})

  @impl true
  def init(:ok), do: {:ok, %{instances: %{}, intent: MapSet.new()}}

  @impl true
  def handle_call({:load, %ModelRef{id: id} = model, profile}, _from, state) do
    case state.instances do
      %{^id => entry} ->
        # Already running — idempotent, return the live instance.
        {:reply, {:ok, info(entry)}, state}

      _ ->
        case start_instance(model, profile) do
          {:ok, entry} ->
            state = put_in(state.instances[id], entry)
            emit(:loading, entry, nil)
            {:reply, {:ok, info(entry)}, state}

          {:error, _} = err ->
            {:reply, err, state}
        end
    end
  end

  def handle_call({:unload, id}, _from, state) do
    case state.instances do
      %{^id => entry} ->
        # Record intent BEFORE terminating so the monitored :DOWN is read as
        # an expected :unloaded, not a crash.
        state = update_in(state.intent, &MapSet.put(&1, id))
        DynamicSupervisor.terminate_child(Instances, entry.pid)
        {:reply, :ok, state}

      _ ->
        {:reply, {:error, :not_running}, state}
    end
  end

  def handle_call(:running, _from, state),
    do: {:reply, Enum.map(state.instances, fn {_id, e} -> info(e) end), state}

  @impl true
  def handle_cast({:mark_up, pid}, state) do
    case find_by_pid(state, pid) do
      {id, %{status: :loading} = entry} ->
        entry = %{entry | status: :up}
        Logger.info("instance up: #{id} on :#{entry.port}")
        emit(:up, entry, nil)
        {:noreply, put_in(state.instances[id], entry)}

      _ ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    case find_by_ref(state, ref) do
      {id, entry} ->
        expected? = MapSet.member?(state.intent, id)

        state = %{
          state
          | instances: Map.delete(state.instances, id),
            intent: MapSet.delete(state.intent, id)
        }

        cond do
          expected? ->
            emit(:unloaded, entry, nil)

          entry.status == :loading ->
            Logger.warning("instance #{id} failed to start: #{inspect(reason)}")
            emit(:failed, entry, reason)

          true ->
            Logger.warning("instance #{id} went down: #{inspect(reason)}")
            emit(:down, entry, reason)
        end

        {:noreply, state}

      nil ->
        {:noreply, state}
    end
  end

  # --- internals ---

  defp start_instance(%ModelRef{} = model, profile) do
    port = free_port()
    spec = %{model: model, profile: profile, port: port}

    case DynamicSupervisor.start_child(Instances, {instance_module(), spec}) do
      {:ok, pid} ->
        ref = Process.monitor(pid)

        {:ok,
         %{
           pid: pid,
           ref: ref,
           model: model,
           port: port,
           profile: profile,
           status: :loading,
           started_at: DateTime.utc_now()
         }}

      {:error, _} = err ->
        err
    end
  end

  defp info(entry) do
    # Engine is co-located with the agent; advertise the routable host (decision #1).
    host = Application.get_env(:airo_agent, :advertise_host, "127.0.0.1")

    %InstanceInfo{
      model_id: entry.model.id,
      revision: entry.model.revision,
      engine: entry.model.engine,
      host: host,
      port: entry.port,
      status: entry.status,
      base_url: "http://#{host}:#{entry.port}/v1",
      started_at: entry.started_at
    }
  end

  defp emit(type, entry, reason) do
    Notifier.publish(%Event{
      type: type,
      model_id: entry.model.id,
      info: info(entry),
      reason: reason,
      at: DateTime.utc_now()
    })
  end

  defp find_by_pid(state, pid), do: Enum.find(state.instances, fn {_id, e} -> e.pid == pid end)
  defp find_by_ref(state, ref), do: Enum.find(state.instances, fn {_id, e} -> e.ref == ref end)

  defp instance_module,
    do: Application.get_env(:airo_agent, :instance_module, AiroAgent.Instance)

  # Let the OS hand us a free port, then close and reuse it for the engine.
  defp free_port do
    {:ok, sock} = :gen_tcp.listen(0, [])
    {:ok, port} = :inet.port(sock)
    :ok = :gen_tcp.close(sock)
    port
  end
end
