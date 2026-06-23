defmodule AiroAgent.Notifier.Supervisor do
  @moduledoc """
  Firewall between the Airo channel client and serving.

  The channel (`AiroAgent.Notifier.Channel`) talks to Airo over the network.
  Airo going away — or an unforeseen channel bug — must NEVER take the engines
  down with it. Serving's availability cannot depend on Airo's.

  The danger is restart-intensity exhaustion, not a single crash: under the flat
  application supervisor the channel was a sibling of `Fleet`/`Instances`, so a
  channel crash-loop would blow the app supervisor's budget (default 3 restarts /
  5s) and terminate *every* child, unloading the live model.

  Isolating the channel here, with a wide restart budget, contains that: a crash
  loop is absorbed locally and never reaches the application supervisor. Network
  flaps already back off inside slipstream; this budget guards against a tight
  loop from an unforeseen callback crash. Worst case the channel subtree gives up
  and telemetry pauses until the next agent restart — serving stays up regardless.
  """

  use Supervisor

  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl true
  def init(_init_arg) do
    children = [AiroAgent.Notifier.Channel]

    # Wide window relative to the app supervisor (3/5s): tolerate frequent
    # restarts so a channel fault is contained here instead of cascading.
    Supervisor.init(children, strategy: :one_for_one, max_restarts: 100, max_seconds: 10)
  end
end
