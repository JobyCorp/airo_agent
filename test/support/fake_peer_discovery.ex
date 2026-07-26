defmodule AiroAgent.Test.FakePeerDiscovery do
  @moduledoc """
  Stands in for `AiroAgent.PeerRanks.discover/0` so tests can drive peer-rank
  appearance and disappearance without a container runtime.

  Discovery is the only side-effecting part of `PeerRanks`; everything else —
  event emission, Fleet's merge, the load guard — is state-machine logic worth
  testing directly. Selected with
  `config :airo_agent, :peer_discovery_module, AiroAgent.Test.FakePeerDiscovery`.
  """

  alias AiroAgent.SlotInfo

  @doc "Set the slots the next poll will see."
  def put(slots), do: Application.put_env(:airo_agent, :fake_peer_slots, slots)

  @doc "Convenience: one peer rank on `port`."
  def peer(port, opts \\ []) do
    %SlotInfo{
      port: port,
      base_url: "http://worker:#{port}/v1",
      status: :up,
      resident_model: Keyword.get(opts, :model, "org/big:fp8"),
      cluster_id: Keyword.get(opts, :cluster_id, "dep-7f3a"),
      tp_rank: Keyword.get(opts, :tp_rank, 1),
      tp_size: Keyword.get(opts, :tp_size, 2)
    }
  end

  def discover, do: Application.get_env(:airo_agent, :fake_peer_slots, [])
end
