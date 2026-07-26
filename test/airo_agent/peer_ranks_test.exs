defmodule AiroAgent.PeerRanksTest do
  @moduledoc """
  Discovery of multi-node ranks this agent did not launch.

  `discover/0` shells out to the container runtime, so these tests drive the
  pure parsing seam (`parse_line/1`) instead — same rule as the engine adapters,
  which are tested as pure functions with no processes spawned.
  """
  use ExUnit.Case, async: false

  alias AiroAgent.{PeerRanks, SlotInfo}

  setup do
    # Restore rather than delete: `advertise_host` is set by config, so deleting
    # it leaks an empty host into every later test that builds a base_url.
    prior = Application.get_env(:airo_agent, :advertise_host)
    Application.put_env(:airo_agent, :advertise_host, "192.168.68.91")
    on_exit(fn -> Application.put_env(:airo_agent, :advertise_host, prior) end)
    :ok
  end

  # id \t rank \t size \t model \t port — the ps --format projection.
  defp line(fields), do: Enum.join(fields, "\t")

  describe "parse_line/1" do
    test "reads a peer rank into a slot" do
      assert [%SlotInfo{} = slot] =
               PeerRanks.parse_line(
                 line(["dep-7f3a1c2b", "1", "2", "fraserprice/DeepSeek-V4:fp8", "8081"])
               )

      assert slot.port == 8081
      assert slot.cluster_id == "dep-7f3a1c2b"
      assert slot.tp_rank == 1
      assert slot.tp_size == 2
      assert slot.resident_model == "fraserprice/DeepSeek-V4:fp8"
      assert slot.status == :up
    end

    test "reports a base_url even though a peer serves nothing there" do
      # Airo's Provider requires an absolute http(s) URL, so one is reported and
      # Airo marks the slot serves_api: false rather than routing to it.
      assert [slot] = PeerRanks.parse_line(line(["dep-1", "1", "2", "m", "8081"]))

      assert slot.base_url == "http://192.168.68.91:8081/v1"
    end

    test "ignores rank 0 — on its own host that container is a real Fleet slot" do
      assert PeerRanks.parse_line(line(["dep-7f3a", "0", "2", "m", "8081"])) == []
    end

    test "ignores a container with no cluster id" do
      assert PeerRanks.parse_line(line(["", "1", "2", "m", "8081"])) == []
    end

    test "ignores unparseable rank or port rather than guessing" do
      assert PeerRanks.parse_line(line(["dep-1", "", "2", "m", "8081"])) == []
      assert PeerRanks.parse_line(line(["dep-1", "1", "2", "m", ""])) == []
      assert PeerRanks.parse_line(line(["dep-1", "notanint", "2", "m", "8081"])) == []
    end

    test "survives a short or empty line from the runtime" do
      assert PeerRanks.parse_line("") == []
      assert PeerRanks.parse_line("dep-1\t1") == []
    end

    test "a missing size leaves tp_size nil rather than failing the whole rank" do
      assert [slot] = PeerRanks.parse_line(line(["dep-1", "1", "", "m", "8081"]))

      assert slot.tp_size == nil
      assert slot.tp_rank == 1
    end

    test "an unlabelled model leaves resident_model nil" do
      assert [slot] = PeerRanks.parse_line(line(["dep-1", "1", "2", "", "8081"]))

      assert slot.resident_model == nil
    end
  end

  describe "discover/0" do
    test "yields no peers when the runtime is missing, rather than crashing" do
      Application.put_env(:airo_agent, :container_runtime, "definitely-not-a-runtime")
      on_exit(fn -> Application.delete_env(:airo_agent, :container_runtime) end)

      assert PeerRanks.discover() == []
    end
  end

  describe "slots/0 and held?/1" do
    test "report nothing when the process isn't running" do
      # A host with no container engine never starts PeerRanks; Fleet still calls
      # both on every load and slot listing.
      refute Process.whereis(PeerRanks)

      assert PeerRanks.slots() == []
      refute PeerRanks.held?(8081)
    end
  end
end
