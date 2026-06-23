defmodule AiroAgent.Test.FakeInstance do
  @moduledoc false
  # Stand-in for AiroAgent.Instance that never spawns a real engine, so the Fleet
  # state machine can be driven deterministically. Injected via
  # `config :airo_agent, :instance_module, AiroAgent.Test.FakeInstance`.
  #
  # Does NOT trap exits, so `Process.exit(pid, reason)` simulates an engine crash
  # with that exact reason (→ Fleet monitor :DOWN). `become_ready/1` simulates the
  # readiness predicate passing.

  use GenServer, restart: :temporary

  def start_link(spec), do: GenServer.start_link(__MODULE__, spec)

  def become_ready(pid, props \\ %{}), do: send(pid, {:become_ready, props})

  @impl true
  def init(%{model: _model, profile: _profile, port: _port}), do: {:ok, %{}}

  @impl true
  def handle_info({:become_ready, props}, state) do
    AiroAgent.Fleet.mark_up(self(), props)
    {:noreply, state}
  end
end
