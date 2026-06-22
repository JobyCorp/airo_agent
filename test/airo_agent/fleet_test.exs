defmodule AiroAgent.FleetTest do
  use ExUnit.Case, async: false

  alias AiroAgent.{Fleet, Instances, ModelRef}
  alias AiroAgent.Fleet.Event
  alias AiroAgent.Test.FakeInstance

  setup do
    Application.put_env(:airo_agent, :instance_module, FakeInstance)
    Application.put_env(:airo_agent, :notifier, AiroAgent.Test.Notifier)
    Application.put_env(:airo_agent, :test_notifier_pid, self())

    on_exit(fn ->
      Application.put_env(:airo_agent, :test_notifier_pid, nil)

      for {_, pid, _, _} <- DynamicSupervisor.which_children(Instances), is_pid(pid) do
        DynamicSupervisor.terminate_child(Instances, pid)
      end

      Application.delete_env(:airo_agent, :instance_module)
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

  test "load returns :loading immediately and emits a :loading event" do
    assert {:ok, info} = Fleet.load(model("m1"))
    assert info.status == :loading
    assert info.model_id == "m1"
    assert info.base_url == "http://127.0.0.1:#{info.port}/v1"
    assert_receive {:event, %Event{type: :loading, model_id: "m1"}}
  end

  test "readiness transitions to :up" do
    {:ok, _} = Fleet.load(model("m2"))
    assert_receive {:event, %Event{type: :loading}}

    FakeInstance.become_ready(child_pid())

    assert_receive {:event, %Event{type: :up, model_id: "m2"}}
    assert [%{status: :up, model_id: "m2"}] = Fleet.running()
  end

  test "unexpected exit while up emits :down with the reason" do
    {:ok, _} = Fleet.load(model("m3"))
    FakeInstance.become_ready(child_pid())
    assert_receive {:event, %Event{type: :up}}

    Process.exit(child_pid(), :boom)

    assert_receive {:event, %Event{type: :down, model_id: "m3", reason: :boom}}
    assert Fleet.running() == []
  end

  test "exit before readiness emits :failed" do
    {:ok, _} = Fleet.load(model("m4"))
    assert_receive {:event, %Event{type: :loading}}

    Process.exit(child_pid(), :badmodel)

    assert_receive {:event, %Event{type: :failed, model_id: "m4", reason: :badmodel}}
    assert Fleet.running() == []
  end

  test "unload emits :unloaded, not :down" do
    {:ok, _} = Fleet.load(model("m5"))
    FakeInstance.become_ready(child_pid())
    assert_receive {:event, %Event{type: :up}}

    assert :ok = Fleet.unload("m5")

    assert_receive {:event, %Event{type: :unloaded, model_id: "m5"}}
    refute_receive {:event, %Event{type: :down}}
    assert Fleet.running() == []
  end

  test "loading an already-running model is idempotent" do
    {:ok, _} = Fleet.load(model("m6"))
    FakeInstance.become_ready(child_pid())
    assert_receive {:event, %Event{type: :up}}

    assert {:ok, info} = Fleet.load(model("m6"))
    assert info.status == :up
    assert length(DynamicSupervisor.which_children(Instances)) == 1
  end

  test "unloading an unknown model is an error" do
    assert {:error, :not_running} = Fleet.unload("nope")
  end
end
