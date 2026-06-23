defmodule AiroAgent.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    # Before anything can load: sweep container engines' orphan slots left by a
    # prior crashed agent (they outlive the BEAM — they're in the runtime's
    # cgroup, not ours). Gated so llama-only hosts never shell out to a runtime.
    reap_orphan_containers()

    children =
      [
        # GPU telemetry (VRAM/util) — polled, read-only.
        AiroAgent.GPU,
        # Local model catalog with provenance (HF snapshot sha = revision).
        AiroAgent.Inventory,
        # One supervised OS process per *running* engine instance.
        AiroAgent.Instances,
        # Canonical in-memory state + lifecycle-event emitter; load/unload route here.
        AiroAgent.Fleet
      ] ++
        notifier_children() ++
        [
          # Control-plane HTTP API. NOT on the inference data path.
          {Bandit, plug: AiroAgent.Api.Router, scheme: :http, ip: bind_ip(), port: api_port()}
        ]

    opts = [strategy: :one_for_one, name: AiroAgent.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Start the slipstream channel client only when it's the configured notifier
  # (i.e. AIRO_SOCKET_URL is set). Otherwise the Log notifier needs no process.
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

  # Only container-based engines leave orphan containers; today that's vLLM.
  defp reap_orphan_containers do
    if :vllm in Application.get_env(:airo_agent, :engines, []) do
      AiroAgent.Engine.Vllm.reap_orphans()
    end

    :ok
  end

  defp api_port, do: Application.get_env(:airo_agent, :api_port, 4400)

  # Default to loopback; runtime.exs widens to LAN only with a token set.
  defp bind_ip, do: Application.get_env(:airo_agent, :bind_ip, {127, 0, 0, 1})
end
