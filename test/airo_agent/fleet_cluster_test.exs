defmodule AiroAgent.FleetClusterTest do
  @moduledoc """
  Fleet's handling of ranks belonging to another host's multi-node load.

  The worker in a two-host cluster has a rank sitting on its GPU that its own
  agent did not start and does not supervise. It must be *reported* so Airo can
  account for the VRAM, and it must never be treated as a slot this agent can
  load into.
  """
  use ExUnit.Case, async: false

  alias AiroAgent.{Fleet, Instances, ModelRef, PeerRanks}
  alias AiroAgent.Fleet.Event
  alias AiroAgent.Test.{FakeInstance, FakePeerDiscovery}

  @slot 8081
  @peer_slot 8082

  setup do
    Application.put_env(:airo_agent, :instance_module, FakeInstance)
    Application.put_env(:airo_agent, :notifier, AiroAgent.Test.Notifier)
    Application.put_env(:airo_agent, :test_notifier_pid, self())
    Application.put_env(:airo_agent, :peer_discovery_module, FakePeerDiscovery)
    Application.put_env(:airo_agent, :port_free_tries, 1)
    FakePeerDiscovery.put([])

    pid = start_supervised!(PeerRanks)

    on_exit(fn ->
      Application.put_env(:airo_agent, :test_notifier_pid, nil)

      for {_, child, _, _} <- DynamicSupervisor.which_children(Instances), is_pid(child) do
        DynamicSupervisor.terminate_child(Instances, child)
      end

      # terminate_child returns before Fleet has handled the resulting :DOWN, and
      # Fleet is global — an unprocessed teardown event would otherwise surface in
      # whichever test runs next. A call flushes its mailbox while the notifier
      # pid is still silenced.
      Fleet.slots()

      Application.delete_env(:airo_agent, :instance_module)
      Application.delete_env(:airo_agent, :peer_discovery_module)
      Application.delete_env(:airo_agent, :fake_peer_slots)
      Application.delete_env(:airo_agent, :port_free_tries)
      Application.put_env(:airo_agent, :notifier, AiroAgent.Notifier.Log)
    end)

    %{peer_ranks: pid}
  end

  defp model(id),
    do: %ModelRef{id: id, path: "/tmp/#{id}.gguf", engine: :llama_cpp, revision: "rev-#{id}"}

  defp see(slots) do
    FakePeerDiscovery.put(slots)
    PeerRanks.refresh()
  end

  describe "reporting" do
    test "a discovered peer rank appears alongside this host's own slots" do
      see([FakePeerDiscovery.peer(@peer_slot)])

      ports = Fleet.slots() |> Enum.map(& &1.port) |> Enum.sort()
      assert @slot in ports
      assert @peer_slot in ports
    end

    test "the peer's slot carries the shared id and its own rank" do
      see([FakePeerDiscovery.peer(@peer_slot, cluster_id: "dep-7f3a", tp_rank: 1, tp_size: 2)])

      peer = Fleet.slots() |> Enum.find(&(&1.port == @peer_slot))

      assert peer.cluster_id == "dep-7f3a"
      assert peer.tp_rank == 1
      assert peer.tp_size == 2
      assert peer.status == :up
    end

    test "no peers means the slot listing is unchanged" do
      assert Enum.map(Fleet.slots(), & &1.port) == [@slot]
    end
  end

  describe "load guard" do
    test "refuses to load onto a port held by another host's rank" do
      see([FakePeerDiscovery.peer(@peer_slot)])

      # That GPU is already committed; loading here would OOM late and confusingly.
      assert {:error, :peer_rank_resident} = Fleet.load(model("m1"), @peer_slot, %{})
    end

    test "refuses to unload another host's rank — it isn't ours to tear down" do
      see([FakePeerDiscovery.peer(@peer_slot)])

      assert {:error, :peer_rank_resident} = Fleet.unload(@peer_slot)
    end

    test "an ordinary slot still loads while a peer rank is present elsewhere" do
      see([FakePeerDiscovery.peer(@peer_slot)])

      assert {:ok, info} = Fleet.load(model("m1"), @slot, %{})
      assert info.port == @slot
      assert info.status == :loading
    end

    test "the port is loadable again once the peer rank goes away" do
      see([FakePeerDiscovery.peer(@slot)])
      assert {:error, :peer_rank_resident} = Fleet.load(model("m1"), @slot, %{})

      see([])
      assert {:ok, _} = Fleet.load(model("m1"), @slot, %{})
    end
  end

  describe "events" do
    test "a rank appearing is pushed as an up event, not left for the heartbeat" do
      see([FakePeerDiscovery.peer(@peer_slot)])

      assert_receive {:event, %Event{type: :up, port: @peer_slot} = event}
      assert event.cluster_id == "dep-7f3a"
      assert event.tp_rank == 1
      assert event.tp_size == 2
    end

    test "a rank disappearing is pushed as a down event" do
      see([FakePeerDiscovery.peer(@peer_slot)])
      assert_receive {:event, %Event{type: :up}}

      # A vanished rank breaks the whole load; Airo has to hear about it promptly.
      see([])

      assert_receive {:event, %Event{type: :down, port: @peer_slot, reason: "peer_rank_gone"}}
    end

    test "a steady rank emits nothing on repeated polls" do
      see([FakePeerDiscovery.peer(@peer_slot)])
      assert_receive {:event, %Event{type: :up}}

      PeerRanks.refresh()
      PeerRanks.refresh()

      refute_receive {:event, _}, 100
    end
  end
end
