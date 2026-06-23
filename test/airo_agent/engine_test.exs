defmodule AiroAgent.EngineTest do
  use ExUnit.Case, async: false

  import ExUnit.CaptureLog
  alias AiroAgent.Engine

  describe "known?/1" do
    test "true for a registered adapter, false otherwise" do
      assert Engine.known?(:llama_cpp)
      assert Engine.known?(:vllm)
      refute Engine.known?(:tgi)
      refute Engine.known?(:nonsense)
    end
  end

  describe "inventory_all/1 tolerates a host-misconfigured engine" do
    setup do
      prev = Application.get_env(:airo_agent, :engines)
      on_exit(fn -> Application.put_env(:airo_agent, :engines, prev) end)
      :ok
    end

    test "skips an engine with no adapter (warns) instead of crashing" do
      root = Path.join(System.tmp_dir!(), "airo_eng_#{System.unique_integer([:positive])}")
      File.mkdir_p!(root)
      on_exit(fn -> File.rm_rf!(root) end)

      Application.put_env(:airo_agent, :engines, [:llama_cpp, :tgi])

      log =
        capture_log(fn ->
          # Empty root → llama.cpp returns no models; :tgi is dropped, not fetched.
          assert {:ok, []} = Engine.inventory_all(model_roots: [root])
        end)

      assert log =~ "unknown engine :tgi"
    end
  end
end
