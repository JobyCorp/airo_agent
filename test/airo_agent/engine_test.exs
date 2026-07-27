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

  describe "exports?/3" do
    test "detects an optional callback even when the module isn't loaded yet" do
      # The orphan sweep runs at boot, before anything has touched the adapter.
      # Bare function_exported?/3 answers false for an unloaded module, which
      # would silently skip the sweep under :interactive code loading.
      #
      # Purges AiroAgent.Test.LazyAdapter rather than a real adapter: unloading
      # one the rest of the suite is using would break whatever runs alongside.
      adapter = AiroAgent.Test.LazyAdapter
      :code.purge(adapter)
      :code.delete(adapter)
      refute :erlang.function_exported(adapter, :reap_orphans, 0)

      assert Engine.exports?(adapter, :reap_orphans, 0)
      # ...and it really did load it back, not just report optimistically.
      assert :erlang.function_exported(adapter, :reap_orphans, 0)
    end

    test "false for a callback an adapter doesn't implement" do
      refute Engine.exports?(AiroAgent.Engine.LlamaCpp, :cluster_info, 2)
      refute Engine.exports?(AiroAgent.Engine.LlamaCpp, :reap_orphans, 0)
      refute Engine.exports?(NotAModule, :inventory, 1)
    end
  end

  # POST /load atomizes against one shared whitelist, so a key an adapter reads
  # but the router omits can never reach it — however well documented it is. That
  # is exactly how llama.cpp's `mmproj` override was dead on arrival.
  describe "profile-key parity between the adapters and the API" do
    @adapters [AiroAgent.Engine.LlamaCpp, AiroAgent.Engine.Vllm]

    defp accepted_keys, do: MapSet.new(AiroAgent.Api.Router.profile_keys(), &String.to_atom/1)

    test "every key an adapter honours is accepted by POST /load" do
      for adapter <- @adapters do
        unreachable =
          MapSet.difference(MapSet.new(adapter.honored_profile_keys()), accepted_keys())

        assert MapSet.equal?(unreachable, MapSet.new()),
               "#{inspect(adapter)} reads #{inspect(MapSet.to_list(unreachable))}, " <>
                 "but POST /load drops those keys before the adapter sees them — " <>
                 "add them to AiroAgent.Api.Router's @profile_keys"
      end
    end

    test "every key POST /load accepts is honoured by at least one adapter" do
      honoured = @adapters |> Enum.flat_map(& &1.honored_profile_keys()) |> MapSet.new()
      dead = MapSet.difference(accepted_keys(), honoured)

      assert MapSet.equal?(dead, MapSet.new()),
             "POST /load accepts #{inspect(MapSet.to_list(dead))}, which no adapter reads"
    end

    test "the sampling knobs are the known asymmetry: llama.cpp honours them, vLLM does not" do
      sampling = [:temperature, :top_p, :repeat_penalty, :presence_penalty, :frequency_penalty]
      llama = AiroAgent.Engine.LlamaCpp.honored_profile_keys()
      vllm = AiroAgent.Engine.Vllm.honored_profile_keys()

      for key <- sampling do
        assert key in llama, "llama.cpp maps #{key} to argv"
        refute key in vllm, "vLLM maps no sampling keys — see Engine.Vllm.honored_profile_keys/0"
      end
    end

    test "the engine-neutral knobs are honoured by both" do
      for key <- [:ctx, :parallel, :extra_argv, :disable_thinking],
          adapter <- @adapters do
        assert key in adapter.honored_profile_keys(),
               "#{inspect(adapter)} must honour the engine-neutral key #{key}"
      end
    end
  end
end
