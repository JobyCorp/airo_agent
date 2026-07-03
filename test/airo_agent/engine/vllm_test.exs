defmodule AiroAgent.Engine.VllmTest do
  use ExUnit.Case, async: true

  alias AiroAgent.Engine.Vllm
  alias AiroAgent.ModelRef

  defp model(opts \\ []) do
    %ModelRef{
      id: Keyword.get(opts, :id, "org/repo"),
      repo: "org/repo",
      path: Keyword.get(opts, :path, "/cache/models--org--repo/snapshots/abc"),
      family: Keyword.get(opts, :family),
      ctx_max: Keyword.get(opts, :ctx_max),
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

    test "omits ctx/parallel flags when unset AND ctx_max is unknown" do
      {:ok, spec} = Vllm.launch_spec(model(), %{}, 8081)
      refute "--max-model-len" in spec.argv
      refute "--max-num-seqs" in spec.argv
      # …but the safe defaults are still applied:
      assert arg_after(spec.argv, "--tensor-parallel-size") == "1"
      assert arg_after(spec.argv, "--dtype") == "auto"
    end
  end

  describe "launch_spec/3 — bare-profile ctx cap" do
    test "a bare profile is capped, not vLLM's full-window default" do
      # Without this, vLLM defaults --max-model-len to max_position_embeddings
      # (262k on Qwen3.5) and OOM-crashes a 16 GB card after weights load.
      {:ok, spec} = Vllm.launch_spec(model(ctx_max: 262_144), %{}, 8081)
      assert arg_after(spec.argv, "--max-model-len") == "32768"
    end

    test "a small model's full window is under the cap and used as-is" do
      {:ok, spec} = Vllm.launch_spec(model(ctx_max: 8192), %{}, 8081)
      assert arg_after(spec.argv, "--max-model-len") == "8192"
    end

    test "an explicit profile ctx wins over the cap (big windows are opt-in)" do
      {:ok, spec} = Vllm.launch_spec(model(ctx_max: 262_144), %{ctx: 262_144}, 8081)
      assert arg_after(spec.argv, "--max-model-len") == "262144"
    end

    test "a nil ctx in the request means unset, not uncapped" do
      {:ok, spec} = Vllm.launch_spec(model(ctx_max: 262_144), %{ctx: nil}, 8081)
      assert arg_after(spec.argv, "--max-model-len") == "32768"
    end
  end

  describe "launch_spec/3 — tool calling + thinking" do
    setup do
      root = Path.join(System.tmp_dir!(), "airo_vllm_tpl_#{System.unique_integer([:positive])}")
      File.mkdir_p!(root)
      on_exit(fn -> File.rm_rf!(root) end)
      %{dir: root}
    end

    test "Qwen3.5/Coder XML template → qwen3_xml parser + auto tool choice", %{dir: dir} do
      File.write!(Path.join(dir, "chat_template.jinja"), """
      <tool_call>\n<function=example>\n<parameter=x>\n</parameter>\n</function>\n</tool_call>
      """)

      {:ok, spec} = Vllm.launch_spec(model(path: dir, family: "qwen3_5"), %{}, 8081)
      assert "--enable-auto-tool-choice" in spec.argv
      assert arg_after(spec.argv, "--tool-call-parser") == "qwen3_xml"
      assert arg_after(spec.argv, "--reasoning-parser") == "qwen3"
    end

    test "hermes-style JSON template → hermes parser", %{dir: dir} do
      File.write!(
        Path.join(dir, "chat_template.jinja"),
        ~s(<tool_call>\n{"name": "{{ tool_call.name }}", "arguments": …}\n</tool_call>)
      )

      {:ok, spec} = Vllm.launch_spec(model(path: dir, family: "qwen3"), %{}, 8081)
      assert arg_after(spec.argv, "--tool-call-parser") == "hermes"
    end

    test "template embedded in tokenizer_config.json is found too", %{dir: dir} do
      File.write!(
        Path.join(dir, "tokenizer_config.json"),
        Jason.encode!(%{"chat_template" => "<tool_call>\n<function=f>…</function>"})
      )

      {:ok, spec} = Vllm.launch_spec(model(path: dir), %{}, 8081)
      assert arg_after(spec.argv, "--tool-call-parser") == "qwen3_xml"
    end

    test "no template / unrecognized format → no parser flags (vLLM default)", %{dir: dir} do
      {:ok, spec} = Vllm.launch_spec(model(path: dir), %{}, 8081)
      refute "--enable-auto-tool-choice" in spec.argv
      refute "--tool-call-parser" in spec.argv
      refute "--reasoning-parser" in spec.argv
    end

    test "extra_argv carrying --tool-call-parser owns the choice — nothing auto-added", %{
      dir: dir
    } do
      File.write!(Path.join(dir, "chat_template.jinja"), "<function=f>")
      extra = ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]

      {:ok, spec} = Vllm.launch_spec(model(path: dir), %{extra_argv: extra}, 8081)
      assert Enum.count(spec.argv, &(&1 == "--tool-call-parser")) == 1
      assert arg_after(spec.argv, "--tool-call-parser") == "hermes"
    end

    test "non-qwen3 family gets the tool parser but no reasoning parser", %{dir: dir} do
      File.write!(Path.join(dir, "chat_template.jinja"), "<function=f>")

      {:ok, spec} = Vllm.launch_spec(model(path: dir, family: "llama"), %{}, 8081)
      assert arg_after(spec.argv, "--tool-call-parser") == "qwen3_xml"
      refute "--reasoning-parser" in spec.argv
    end

    test "disable_thinking maps to a server-side template kwarg" do
      {:ok, spec} = Vllm.launch_spec(model(), %{disable_thinking: true}, 8081)

      assert arg_after(spec.argv, "--default-chat-template-kwargs") ==
               ~s({"enable_thinking": false})

      {:ok, spec} = Vllm.launch_spec(model(), %{}, 8081)
      refute "--default-chat-template-kwargs" in spec.argv
    end
  end

  describe "tool_parser_for_template/1" do
    test "GLM-style <tool_call> with <arg_key> pairs is NOT hermes — no parser" do
      template = "<tool_call>{{ name }}\n<arg_key>k</arg_key><arg_value>v</arg_value>"
      assert Vllm.tool_parser_for_template(template) == nil
    end

    test "nil-safe" do
      assert Vllm.tool_parser_for_template(nil) == nil
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

    test "reads ctx_max from text_config on VL models (Qwen3.5 nests it)", %{root: root} do
      snapshot(root, "models--QuantTrio--Qwen3.5-9B-AWQ", "938f8e3", %{
        "model_type" => "qwen3_5",
        "vision_config" => %{"depth" => 27},
        "text_config" => %{
          "model_type" => "qwen3_5_text",
          "max_position_embeddings" => 262_144
        }
      })

      {:ok, [ref]} = Vllm.inventory(model_roots: [root])
      assert ref.ctx_max == 262_144
      assert ref.family == "qwen3_5"
      assert :vision in ref.capabilities
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
