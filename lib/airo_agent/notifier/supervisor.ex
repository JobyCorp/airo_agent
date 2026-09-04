defmodule AiroAgent.Notifier.Supervisor do
  @moduledoc """
  Firewall between the Airo channel clients and serving.

  The channels (`AiroAgent.Notifier.Channel`, one per configured airo endpoint —
  the controller plus any observers, S26) talk to Airo over the network. Airo
  going away — or an unforeseen channel bug — must NEVER take the engines down
  with it. Serving's availability cannot depend on Airo's.

  The danger is restart-intensity exhaustion, not a single crash: under the flat
  application supervisor the channel was a sibling of `Fleet`/`Instances`, so a
  channel crash-loop would blow the app supervisor's budget (default 3 restarts /
  5s) and terminate *every* child, unloading the live model.

  Isolating the channels here, with a wide restart budget, contains that: a
  crash loop is absorbed locally and never reaches the application supervisor.
  The budget is shared across all channels on purpose — the firewall protects
  serving from *all* of them, not each channel from the others. Network flaps
  already back off inside slipstream; this budget guards against a tight loop
  from an unforeseen callback crash. Worst case the channel subtree gives up and
  telemetry pauses until the next agent restart — serving stays up regardless.
  """

  use Supervisor

  alias AiroAgent.Notifier.Channel

  def start_link(init_arg) do
    Supervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl true
  def init(_init_arg) do
    children =
      for endpoint <- Application.get_env(:airo_agent, :airo_endpoints, []) do
        Supervisor.child_spec({Channel, [uri: endpoint.uri, role: endpoint.role]},
          id: {Channel, endpoint.uri}
        )
      end

    # Wide window relative to the app supervisor (3/5s): tolerate frequent
    # restarts so a channel fault is contained here instead of cascading.
    Supervisor.init(children, strategy: :one_for_one, max_restarts: 100, max_seconds: 10)
  end
end
