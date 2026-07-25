defmodule AiroAgent.FleetTest do
  use ExUnit.Case, async: false

  alias AiroAgent.{Fleet, Instances, ModelRef}
  alias AiroAgent.Fleet.Event
  alias AiroAgent.Test.FakeInstance

  # The single slot runtime.exs declares by default in test.
  @slot 8081

  setup do
    Application.put_env(:airo_agent, :instance_module, FakeInstance)
    Application.put_env(:airo_agent, :notifier, AiroAgent.Test.Notifier)
    Application.put_env(:airo_agent, :test_notifier_pid, self())
    # Fakes never bind the slot port, but a real engine on the dev box might be
    # listening there — don't let a swap poll it for the full 30 s ceiling.
    Application.put_env(:airo_agent, :port_free_tries, 1)

    on_exit(fn ->
      # Silence first so teardown engine deaths don't reach the next test's pid.
      Application.put_env(:airo_agent, :test_notifier_pid, nil)

      for {_, pid, _, _} <- DynamicSupervisor.which_children(Instances), is_pid(pid) do
        DynamicSupervisor.terminate_child(Instances, pid)
      end

      Application.delete_env(:airo_agent, :instance_module)
      Application.delete_env(:airo_agent, :port_free_tries)
      Application.put_env(:airo_agent, :notifier, AiroAgent.Notifier.Log)
    end)

    :ok
  end

  defp model(id),
    do: %ModelRef{id: id, path: "/tmp/#{id}.gguf", engine: :llama_cpp, revision: "rev-#{id}"}

  defp child_pid do
    [{_, pid, _, _}] = DynamicSupervisor.which_children(Instances)
    pid
  end

  test "load places a model into a slot (:loading) and returns slot info" do
    assert {:ok, info} = Fleet.load(model("m1"), @slot)
    assert info.port == @slot
    assert info.status == :loading
    assert info.resident_model == "m1"
    host = Application.get_env(:airo_agent, :advertise_host)
    assert info.base_url == "http://#{host}:#{@slot}/v1"
    assert_receive {:event, %Event{type: :loading, port: @slot, resident_model: "m1"}}
  end

  test "readiness transitions the slot to :up" do
    {:ok, _} = Fleet.load(model("m2"), @slot)
    assert_receive {:event, %Event{type: :loading}}

    FakeInstance.become_ready(child_pid())

    assert_receive {:event, %Event{type: :up, port: @slot, resident_model: "m2"}}

    assert Enum.any?(
             Fleet.slots(),
             &(&1.port == @slot and &1.status == :up and &1.resident_model == "m2")
           )
  end

  test "engine props (ctx_total) and the resolved profile surface on the :up slot" do
    {:ok, _} = Fleet.load(model("m2b"), @slot, %{ctx: 36_608})
    assert_receive {:event, %Event{type: :loading}}

    # Simulate what the Instance reports up on readiness: engine /props facts plus
    # the effective launch profile (defaults applied).
    FakeInstance.become_ready(child_pid(), %{
      ctx: 36_608,
      parallel: 4,
      ctx_total: 146_432,
      engine_build: "b1-test",
      resolved_profile: %{ctx: 36_608, parallel: 4, flash_attn: "on"}
    })

    assert_receive {:event, %Event{type: :up, ctx: 36_608, parallel: 4, ctx_total: 146_432}}

    slot = Enum.find(Fleet.slots(), &(&1.port == @slot))
    assert slot.ctx == 36_608
    assert slot.ctx_total == 146_432
    # The agent default (parallel: 4) is visible, not a blank.
    assert slot.profile == %{ctx: 36_608, parallel: 4, flash_attn: "on"}
  end

  test "an unexpected exit marks the slot :down with the reason" do
    {:ok, _} = Fleet.load(model("m3"), @slot)
    FakeInstance.become_ready(child_pid())
    assert_receive {:event, %Event{type: :up}}

    Process.exit(child_pid(), :boom)

    assert_receive {:event, %Event{type: :down, port: @slot, reason: :boom}}
  end

  test "exit before readiness is :failed" do
    {:ok, _} = Fleet.load(model("m4"), @slot)
    assert_receive {:event, %Event{type: :loading}}

    Process.exit(child_pid(), :badmodel)

    assert_receive {:event, %Event{type: :failed, port: @slot, reason: :badmodel}}
  end

  test "unload frees the slot (:unloaded, not :down)" do
    {:ok, _} = Fleet.load(model("m5"), @slot)
    FakeInstance.become_ready(child_pid())
    assert_receive {:event, %Event{type: :up}}

    assert :ok = Fleet.unload(@slot)

    assert_receive {:event, %Event{type: :unloaded, port: @slot}}
    refute_receive {:event, %Event{type: :down}}
  end

  test "loading a different model swaps the slot (one engine, evicting the old)" do
    {:ok, _} = Fleet.load(model("A"), @slot)
    FakeInstance.become_ready(child_pid())
    assert_receive {:event, %Event{type: :up, resident_model: "A"}}

    assert {:ok, info} = Fleet.load(model("B"), @slot)
    assert info.resident_model == "B"
    assert info.status == :loading
    assert_receive {:event, %Event{type: :loading, port: @slot, resident_model: "B"}}
    assert length(DynamicSupervisor.which_children(Instances)) == 1
  end

  test "loading the model already resident and up is idempotent" do
    {:ok, _} = Fleet.load(model("m6"), @slot)
    FakeInstance.become_ready(child_pid())
    assert_receive {:event, %Event{type: :up}}

    assert {:ok, info} = Fleet.load(model("m6"), @slot)
    assert info.status == :up
    assert length(DynamicSupervisor.which_children(Instances)) == 1
  end

  test "loading the resident model with a changed profile reloads it" do
    {:ok, _} = Fleet.load(model("m6b"), @slot, %{ctx: 4096})
    FakeInstance.become_ready(child_pid())
    assert_receive {:event, %Event{type: :up, resident_model: "m6b"}}

    # Same model, new context size — not idempotent: the engine relaunches.
    assert {:ok, info} = Fleet.load(model("m6b"), @slot, %{ctx: 8192})
    assert info.status == :loading
    assert_receive {:event, %Event{type: :loading, port: @slot, resident_model: "m6b"}}
    assert length(DynamicSupervisor.which_children(Instances)) == 1
  end

  test "loading into an unknown slot errors" do
    assert {:error, :unknown_slot} = Fleet.load(model("x"), 9_999)
  end

  test "unloading an unknown slot errors" do
    assert {:error, :unknown_slot} = Fleet.unload(9_999)
  end

  test "slots/0 reports the configured slot" do
    assert Enum.any?(Fleet.slots(), &(&1.port == @slot))
  end
end
