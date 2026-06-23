defmodule AiroAgent.Engine.VllmTest do
  use ExUnit.Case, async: true

  alias AiroAgent.Engine.Vllm
  alias AiroAgent.ModelRef

  defp model(opts \\ []) do
    %ModelRef{
      id: Keyword.get(opts, :id, "org/repo"),
      repo: "org/repo",
      path: Keyword.get(opts, :path, "/cache/models--org--repo/snapshots/abc"),
      engine: :vllm
    }
  end

  # Build a fake HF snapshot dir under `root`: config.json (+ a safetensors file
  # unless :weights? is false), in the canonical models--/snapshots/<sha> layout.
  defp snapshot(root, repo_dir, sha, config, opts \\ []) do
    dir = Path.join([root, repo_dir, "snapshots", sha])
    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "config.json"), Jason.encode!(config))
    if Keyword.get(opts, :weights?, true), do: File.touch!(Path.join(dir, "model.safetensors"))
    dir
  end

  describe "launch_spec/3 — vLLM ctx contract (NO ×parallel)" do
    defp arg_after(argv, flag), do: argv |> Enum.drop_while(&(&1 != flag)) |> Enum.at(1)

    test "--max-model-len is the per-request ctx, NOT ctx × parallel" do
      {:ok, spec} = Vllm.launch_spec(model(), %{ctx: 32_768, parallel: 8}, 8081)
      assert arg_after(spec.argv, "--max-model-len") == "32768"
      assert arg_after(spec.argv, "--max-num-seqs") == "8"
    end

    test "serves the snapshot dir under the routed id, on the given port" do
      {:ok, spec} = Vllm.launch_spec(model(id: "org/repo:fp8"), %{}, 8082)
      assert ["serve", "/cache/models--org--repo/snapshots/abc" | _] = spec.argv
      assert arg_after(spec.argv, "--served-model-name") == "org/repo:fp8"
      assert arg_after(spec.argv, "--port") == "8082"
      assert spec.readiness == {:http_get, "/health"}
    end

    test "omits ctx/parallel flags when unset (vLLM uses its own defaults)" do
      {:ok, spec} = Vllm.launch_spec(model(), %{}, 8081)
      refute "--max-model-len" in spec.argv
      refute "--max-num-seqs" in spec.argv
      # …but the safe defaults are still applied:
      assert arg_after(spec.argv, "--tensor-parallel-size") == "1"
      assert arg_after(spec.argv, "--dtype") == "auto"
    end
  end

  describe "resolve_profile/2" do
    test "merges the request over the model defaults so defaults are visible" do
      resolved = Vllm.resolve_profile(model(), %{ctx: 8192})
      assert resolved.ctx == 8192
      assert resolved.tensor_parallel_size == 1
      assert resolved.dtype == "auto"
    end

    test "the request wins over a default" do
      assert Vllm.resolve_profile(model(), %{tensor_parallel_size: 2}).tensor_parallel_size == 2
    end
  end

  describe "capabilities_from_config/1" do
    test "chat by default; vision when the config carries a vision_config" do
      assert Vllm.capabilities_from_config(%{"model_type" => "qwen2"}) == [:chat]
      assert :vision in Vllm.capabilities_from_config(%{"vision_config" => %{"depth" => 32}})
    end
  end

  describe "runtime_props parsing (/v1/models + /version)" do
    test "parse_models/1 extracts the resolved max_model_len as ctx" do
      # Shape of a real vLLM /v1/models response (trimmed).
      body = %{
        "object" => "list",
        "data" => [%{"id" => "org/repo", "object" => "model", "max_model_len" => 32_768}]
      }

      assert Vllm.parse_models(body) == 32_768
    end

    test "parse_models/1 tolerates a missing/empty body" do
      assert Vllm.parse_models(%{}) == nil
      assert Vllm.parse_models(%{"data" => []}) == nil
      assert Vllm.parse_models(%{"data" => [%{"id" => "x"}]}) == nil
    end

    test "parse_version/1 extracts the engine build" do
      assert Vllm.parse_version(%{"version" => "0.6.3"}) == "0.6.3"
    end

    test "parse_version/1 tolerates a missing/empty body" do
      assert Vllm.parse_version(%{}) == nil
    end
  end

  describe "reap_orphans/0 (boot-time orphan sweep)" do
    test "orphan_ids/1 splits a runtime's `ps -aq` output into ids" do
      assert Vllm.orphan_ids("abc123\ndef456\n") == ["abc123", "def456"]
      assert Vllm.orphan_ids("   \n") == []
      assert Vllm.orphan_ids("") == []
    end

    test "is a no-op (:ok) when the container runtime isn't installed" do
      prev = Application.get_env(:airo_agent, :container_runtime)
      Application.put_env(:airo_agent, :container_runtime, "airo-no-such-runtime-xyz")
      on_exit(fn -> Application.put_env(:airo_agent, :container_runtime, prev) end)

      assert Vllm.reap_orphans() == :ok
    end
  end

  describe "inventory/1 — safetensors snapshots with provenance" do
    setup do
      root = Path.join(System.tmp_dir!(), "airo_vllm_#{System.unique_integer([:positive])}")
      File.mkdir_p!(root)
      on_exit(fn -> File.rm_rf!(root) end)
      %{root: root}
    end

    test "resolves repo/revision/family/ctx and tags engine :vllm", %{root: root} do
      snapshot(root, "models--meta-llama--Llama-3.1-8B", "deadbeef", %{
        "model_type" => "llama",
        "architectures" => ["LlamaForCausalLM"],
        "max_position_embeddings" => 131_072
      })

      {:ok, [ref]} = Vllm.inventory(model_roots: [root])
      assert ref.repo == "meta-llama/Llama-3.1-8B"
      assert ref.revision == "deadbeef"
      assert ref.id == "meta-llama/Llama-3.1-8B"
      assert ref.family == "llama"
      assert ref.ctx_max == 131_072
      assert ref.engine == :vllm
      assert ref.capabilities == [:chat]
    end

    test "reads the quant method and folds it into the id", %{root: root} do
      snapshot(root, "models--org--big-fp8", "cafef00d", %{
        "model_type" => "qwen2",
        "quantization_config" => %{"quant_method" => "fp8"}
      })

      {:ok, [ref]} = Vllm.inventory(model_roots: [root])
      assert ref.quant == "fp8"
      assert ref.id == "org/big-fp8:fp8"
    end

    test "skips a snapshot that has a config but no safetensors (e.g. GGUF-only)", %{root: root} do
      snapshot(root, "models--org--gguf-only", "0001", %{"model_type" => "qwen2"},
        weights?: false
      )

      assert {:ok, []} = Vllm.inventory(model_roots: [root])
    end
  end
end
