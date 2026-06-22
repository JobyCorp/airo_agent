defmodule AiroAgent.Instance do
  @moduledoc """
  Owns ONE external engine OS process via `MuonTrap.Daemon`, so a native crash
  (CUDA OOM, MTP edge case, segfault) is isolated to this supervised child and
  reaped cleanly — it never touches the agent VM, let alone Airo.

  `restart: :temporary` — a model that dies is reported `:down`; Airo decides
  whether to reload. We do not silently respawn into a crash loop.
  """

  use GenServer, restart: :temporary
  require Logger
  alias AiroAgent.{Engine, InstanceInfo, ModelRef}

  def start_link(spec), do: GenServer.start_link(__MODULE__, spec)

  @spec info(pid()) :: InstanceInfo.t()
  def info(pid), do: GenServer.call(pid, :info)

  @impl true
  def init(%{model: %ModelRef{} = model, profile: profile, port: port}) do
    Process.flag(:trap_exit, true)
    adapter = Engine.adapter(model.engine)

    with {:ok, launch} <- adapter.launch_spec(model, profile, port),
         bin <- bin_for(model.engine),
         {:ok, daemon} <- MuonTrap.Daemon.start_link(bin, launch.argv, env: launch.env) do
      schedule_readiness()

      {:ok,
       %{
         model: model,
         port: port,
         daemon: daemon,
         readiness: launch.readiness,
         status: :loading,
         started_at: DateTime.utc_now()
       }}
    else
      {:error, reason} -> {:stop, reason}
    end
  end

  @impl true
  def handle_call(:info, _from, state), do: {:reply, to_info(state), state}

  @impl true
  def handle_info(:check_ready, state) do
    if ready?(state) do
      Logger.info("instance up: #{state.model.id} on :#{state.port}")
      {:noreply, %{state | status: :up}}
    else
      schedule_readiness()
      {:noreply, state}
    end
  end

  def handle_info({:EXIT, _pid, reason}, state) do
    Logger.warning("engine exited for #{state.model.id}: #{inspect(reason)}")
    {:stop, reason, state}
  end

  defp schedule_readiness, do: Process.send_after(self(), :check_ready, 1_000)

  defp ready?(%{readiness: {:http_get, path}, port: port}) do
    case Req.get("http://127.0.0.1:#{port}#{path}", retry: false, receive_timeout: 800) do
      {:ok, %{status: 200}} -> true
      _ -> false
    end
  end

  defp ready?(_), do: false

  defp to_info(s) do
    # Advertise the routable host (engine is co-located with the agent; only the
    # port is dynamic). Readiness checks still use loopback — see ready?/1.
    host = Application.get_env(:airo_agent, :advertise_host, "127.0.0.1")

    %InstanceInfo{
      model_id: s.model.id,
      revision: s.model.revision,
      engine: s.model.engine,
      host: host,
      port: s.port,
      status: s.status,
      base_url: "http://#{host}:#{s.port}/v1",
      started_at: s.started_at
    }
  end

  defp bin_for(engine) do
    Application.get_env(:airo_agent, :engine_bin, %{})
    |> Map.fetch!(engine)
  end
end
