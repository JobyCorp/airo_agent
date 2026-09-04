defmodule AiroAgent.Application do
  @moduledoc false

  use Application

  alias AiroAgent.Engine

  @impl true
  def start(_type, _args) do
    # Before anything can load: let each configured engine reclaim children a
    # prior crashed agent left behind (a container outlives the BEAM — it's in
    # the runtime's cgroup, not ours). Engines with nothing to reclaim don't
    # implement the callback, so a llama-only host never shells out.
    reap_orphans()

    children =
      [
        # GPU telemetry (VRAM/util) — polled, read-only.
        AiroAgent.GPU,
        # Local model catalog with provenance (HF snapshot sha = revision).
        AiroAgent.Inventory,
        # One supervised OS process per *running* engine instance.
        AiroAgent.Instances,
        # Canonical in-memory state + lifecycle-event emitter; load/unload route here.
        AiroAgent.Fleet,
        # Names the channel clients (one per airo endpoint, S26) so publish/1
        # can fan out to all of them. Always started: it costs nothing and the
        # tests start channels without the notifier supervisor.
        {Registry, keys: :unique, name: AiroAgent.Notifier.Registry}
      ] ++
        peer_rank_children() ++
        notifier_children() ++
        [
          # Control-plane HTTP API. NOT on the inference data path.
          {Bandit, plug: AiroAgent.Api.Router, scheme: :http, ip: bind_ip(), port: api_port()}
        ]

    opts = [strategy: :one_for_one, name: AiroAgent.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Start the slipstream channel clients only when the channel is the configured
  # notifier (i.e. at least one airo endpoint is set). Otherwise the Log
  # notifier needs no process.
  #
  # Run it under its own supervisor (not as a bare sibling of Fleet/Instances) so
  # a channel crash-loop is contained there and can never exhaust this
  # supervisor's restart budget and take serving down with it.
  defp notifier_children do
    case Application.get_env(:airo_agent, :notifier) do
      AiroAgent.Notifier.Channel -> [AiroAgent.Notifier.Supervisor]
      _ -> []
    end
  end

  # Watch for ranks of a multi-node load that another host launched onto this
  # GPU. Only container engines can produce them, and only they can be observed
  # (the labels live on the container), so this follows the same :vllm gate as
  # the orphan sweep. Must start AFTER Fleet — Fleet merges its slots.
  defp peer_rank_children do
    if :vllm in Application.get_env(:airo_agent, :engines, []),
      do: [AiroAgent.PeerRanks],
      else: []
  end

  # Dispatched over this host's configured engines rather than naming vLLM: the
  # ability to orphan a child is a property of the engine, so the adapter that
  # has it declares `reap_orphans/0` and the next container-based backend is
  # swept without touching boot. Today only vLLM implements it.
  defp reap_orphans do
    Application.get_env(:airo_agent, :engines, [])
    |> Enum.filter(&Engine.known?/1)
    |> Enum.map(&Engine.adapter/1)
    |> Enum.each(fn adapter ->
      if Engine.exports?(adapter, :reap_orphans, 0), do: adapter.reap_orphans()
    end)

    :ok
  end

  defp api_port, do: Application.get_env(:airo_agent, :api_port, 4400)

  # Default to loopback; runtime.exs widens to LAN only with a token set.
  defp bind_ip, do: Application.get_env(:airo_agent, :bind_ip, {127, 0, 0, 1})
end
