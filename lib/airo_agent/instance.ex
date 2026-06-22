defmodule AiroAgent.Instance do
  @moduledoc """
  Owns ONE external engine OS process via `MuonTrap.Daemon`, so a native crash
  (CUDA OOM, MTP edge case, segfault) is isolated to this supervised child and
  reaped cleanly — it never touches the agent VM, let alone Airo.

  `restart: :temporary` — a dead model is NOT silently respawned. The Instance
  only reports readiness up to `AiroAgent.Fleet` (which owns canonical state and
  emits the lifecycle events); on engine exit it stops, and Fleet's monitor turns
  that into a `:down`/`:failed`/`:unloaded` event.
  """

  use GenServer, restart: :temporary

  alias AiroAgent.{Engine, Fleet, ModelRef}

  def start_link(spec), do: GenServer.start_link(__MODULE__, spec)

  @impl true
  def init(%{model: %ModelRef{} = model, profile: profile, port: port}) do
    Process.flag(:trap_exit, true)
    adapter = Engine.adapter(model.engine)

    with {:ok, launch} <- adapter.launch_spec(model, profile, port),
         bin <- bin_for(model.engine),
         {:ok, daemon} <- MuonTrap.Daemon.start_link(bin, launch.argv, env: launch.env) do
      schedule_readiness()

      {:ok,
       %{model: model, port: port, daemon: daemon, readiness: launch.readiness, ready?: false}}
    else
      {:error, reason} -> {:stop, reason}
    end
  end

  @impl true
  def handle_info(:check_ready, state) do
    cond do
      state.ready? ->
        {:noreply, state}

      ready?(state) ->
        Fleet.mark_up(self())
        {:noreply, %{state | ready?: true}}

      true ->
        schedule_readiness()
        {:noreply, state}
    end
  end

  def handle_info({:EXIT, _pid, reason}, state) do
    # The engine died; stop so Fleet's monitor classifies and emits the event.
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

  defp bin_for(engine) do
    Application.get_env(:airo_agent, :engine_bin, %{})
    |> Map.fetch!(engine)
  end
end
