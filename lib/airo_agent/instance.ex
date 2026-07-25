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

  # SIGTERM→SIGKILL grace for the engine process. muontrap's 500 ms default is
  # far too short for the teardown work SIGTERM triggers: a cluster unload's
  # vllm-slot TERM trap must docker-rm rank 0 AND reach the worker over SSH to
  # rm rank 1 (else the second host keeps serving), and even a single-node
  # docker stop needs the client to outlive the container's graceful exit.
  # This is a ceiling, not a sleep — a fast-exiting engine reaps immediately.
  @delay_to_sigkill 30_000

  def start_link(spec), do: GenServer.start_link(__MODULE__, spec)

  @impl true
  def init(%{model: %ModelRef{} = model, profile: profile, port: port}) do
    Process.flag(:trap_exit, true)
    adapter = Engine.adapter(model.engine)

    with {:ok, launch} <- adapter.launch_spec(model, profile, port),
         bin <- bin_for(model.engine),
         {:ok, daemon} <-
           MuonTrap.Daemon.start_link(bin, launch.argv,
             env: launch.env,
             delay_to_sigkill: @delay_to_sigkill
           ) do
      schedule_readiness()

      {:ok,
       %{
         model: model,
         port: port,
         adapter: adapter,
         resolved_profile: resolved_profile(adapter, model, profile),
         daemon: daemon,
         readiness: launch.readiness,
         ready?: false
       }}
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
        props = Map.put(runtime_props(state), :resolved_profile, state.resolved_profile)
        Fleet.mark_up(self(), props)
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

  # Engine-reported runtime facts (ctx/parallel/build), if the adapter exposes them.
  defp runtime_props(%{adapter: adapter, port: port}) do
    if function_exported?(adapter, :runtime_props, 1), do: adapter.runtime_props(port), else: %{}
  end

  # Effective launch profile (defaults applied), if the adapter exposes it.
  defp resolved_profile(adapter, model, profile) do
    if function_exported?(adapter, :resolve_profile, 2),
      do: adapter.resolve_profile(model, profile),
      else: profile
  end

  defp bin_for(engine) do
    Application.get_env(:airo_agent, :engine_bin, %{})
    |> Map.fetch!(engine)
  end
end
