defmodule AiroAgent.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # GPU telemetry (VRAM/util) — polled, read-only.
      AiroAgent.GPU,
      # Local model catalog with provenance (HF snapshot sha = revision).
      AiroAgent.Inventory,
      # One supervised OS process per *running* engine instance.
      AiroAgent.Instances,
      # Control-plane HTTP API. NOT on the inference data path.
      {Bandit, plug: AiroAgent.Api.Router, scheme: :http, ip: bind_ip(), port: api_port()}
    ]

    opts = [strategy: :one_for_one, name: AiroAgent.Supervisor]
    Supervisor.start_link(children, opts)
  end

  defp api_port, do: Application.get_env(:airo_agent, :api_port, 4400)

  # Default to loopback; runtime.exs widens to LAN only with a token set.
  defp bind_ip, do: Application.get_env(:airo_agent, :bind_ip, {127, 0, 0, 1})
end
