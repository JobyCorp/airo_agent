defmodule AiroAgent.Engine.LlamaCppTest do
  use ExUnit.Case, async: true

  alias AiroAgent.Engine.LlamaCpp
  alias AiroAgent.ModelRef

  defp model(opts \\ []) do
    %ModelRef{
      id: "org/repo:Q4",
      repo: "org/repo",
      path: Keyword.get(opts, :path, "/cache/repo/model.gguf"),
      engine: :llama_cpp
    }
  end

  describe "launch_spec/3 — contract A (ctx is the per-request window)" do
    # `-c` is llama-server's TOTAL KV budget, split across --parallel sequences.
    # Contract A: profile ctx = per-request, so the agent passes -c = ctx × parallel
    # to give each request the full window.
    defp arg_after(argv, flag) do
      argv |> Enum.drop_while(&(&1 != flag)) |> Enum.at(1)
    end

    test "scales -c to ctx × parallel" do
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{ctx: 142_000, parallel: 4}, 8081)
      assert arg_after(spec.argv, "-c") == "568000"
      assert arg_after(spec.argv, "--parallel") == "4"
    end

    test "a bare ctx allocates exactly that -c (default parallel: 1, no surprise ×N)" do
      # Default parallel is 1, so -c == ctx — no silent VRAM multiplication.
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{ctx: 36_608}, 8081)
      assert arg_after(spec.argv, "-c") == "36608"
      assert arg_after(spec.argv, "--parallel") == "1"
    end

    test "explicit parallel multiplies -c (the opt-in VRAM cost)" do
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{ctx: 36_608, parallel: 4}, 8081)
      assert arg_after(spec.argv, "-c") == "146432"
    end

    test "parallel: 1 gives the whole window to one sequence" do
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{ctx: 142_000, parallel: 1}, 8081)
      assert arg_after(spec.argv, "-c") == "142000"
    end

    test "no ctx ⇒ no -c (llama-server picks its own default)" do
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{parallel: 2}, 8081)
      refute "-c" in spec.argv
    end
  end

  describe "launch_spec/3 — disable_thinking (engine-neutral knob)" do
    test "maps to --reasoning off (NOT --reasoning-budget 0, which only caps)" do
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{disable_thinking: true}, 8081)
      assert spec.argv |> Enum.drop_while(&(&1 != "--reasoning")) |> Enum.at(1) == "off"
    end

    test "absent/false ⇒ no flag (thinking stays on, the engine default)" do
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{}, 8081)
      refute "--reasoning" in spec.argv

      {:ok, spec} = LlamaCpp.launch_spec(model(), %{disable_thinking: false}, 8081)
      refute "--reasoning" in spec.argv
    end
  end

  describe "launch_spec/3 — reasoning-budget guard (runaway thinking)" do
    # llama-server's own default is -1 (UNLIMITED): a reasoning trace that never
    # closes </think> generates until the whole ctx fills. The agent caps it by
    # default so a bare profile can't run away.
    test "a bare profile gets the default budget cap" do
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{}, 8081)
      assert arg_after(spec.argv, "--reasoning-budget") == "8192"
    end

    test "a profile's own budget wins over the guard" do
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{reasoning_budget: 24_576}, 8081)
      assert arg_after(spec.argv, "--reasoning-budget") == "24576"
    end

    test "reasoning_budget: -1 opts back into unlimited (explicit, not silent)" do
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{reasoning_budget: -1}, 8081)
      assert arg_after(spec.argv, "--reasoning-budget") == "-1"
    end

    test "disable_thinking suppresses the budget (no trace ⇒ nothing to cap)" do
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{disable_thinking: true}, 8081)
      refute "--reasoning-budget" in spec.argv
    end
  end

  describe "launch_spec/3 — repetition guard (DRY) and penalty passthrough" do
    # All repetition knobs are OPT-IN: a bare profile changes no sampling
    # defaults. DRY is the recommended loop guard (near-inert on normal text,
    # unlike --repeat-penalty, which taxes every token and hurts code).
    test "a bare profile sets no repetition flags" do
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{}, 8081)
      refute "--dry-multiplier" in spec.argv
      refute "--repeat-penalty" in spec.argv
    end

    test "a profile's dry_multiplier enables DRY for that load" do
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{dry_multiplier: 0.8}, 8081)
      assert arg_after(spec.argv, "--dry-multiplier") == "0.8"
    end

    test "classic penalties pass through when a profile sets them" do
      profile = %{repeat_penalty: 1.1, presence_penalty: 1.5, frequency_penalty: 0.2}
      {:ok, spec} = LlamaCpp.launch_spec(model(), profile, 8081)
      assert arg_after(spec.argv, "--repeat-penalty") == "1.1"
      assert arg_after(spec.argv, "--presence-penalty") == "1.5"
      assert arg_after(spec.argv, "--frequency-penalty") == "0.2"
    end
  end

  describe "inventory/1 — vision projectors are not models" do
    setup do
      root = Path.join(System.tmp_dir!(), "airo_inv_#{System.unique_integer([:positive])}")
      File.mkdir_p!(root)
      File.touch!(Path.join(root, "model-Q4_K_M.gguf"))
      File.touch!(Path.join(root, "mmproj-F16.gguf"))
      on_exit(fn -> File.rm_rf!(root) end)
      %{root: root}
    end

    test "mmproj-*.gguf are excluded; real models are kept", %{root: root} do
      {:ok, refs} = LlamaCpp.inventory(model_roots: [root])
      paths = Enum.map(refs, & &1.path)

      assert Enum.any?(paths, &String.ends_with?(&1, "model-Q4_K_M.gguf"))
      refute Enum.any?(paths, &String.contains?(&1, "mmproj"))
    end

    test "mid-name projectors (moondream2-mmproj-*.gguf) are excluded too", %{root: root} do
      # ggml-org names it with a prefix, not `mmproj-*` — the old starts_with?
      # check leaked it as a bogus servable model.
      File.touch!(Path.join(root, "moondream2-mmproj-f16-20250414.gguf"))
      File.touch!(Path.join(root, "moondream2-text-model-f16_ct-vicuna.gguf"))

      {:ok, refs} = LlamaCpp.inventory(model_roots: [root])
      paths = Enum.map(refs, & &1.path)

      refute Enum.any?(paths, &String.contains?(&1, "mmproj"))
      assert Enum.any?(paths, &String.ends_with?(&1, "moondream2-text-model-f16_ct-vicuna.gguf"))
    end

    test "a model with an mmproj sibling is tagged :vision", %{root: root} do
      # root already has model-Q4_K_M.gguf + mmproj-F16.gguf from setup.
      {:ok, refs} = LlamaCpp.inventory(model_roots: [root])
      ref = Enum.find(refs, &String.ends_with?(&1.path, "model-Q4_K_M.gguf"))

      assert :vision in ref.capabilities
      assert :chat in ref.capabilities
    end
  end

  describe "launch_spec/3 — multimodal --mmproj" do
    test "auto-attaches the sibling projector" do
      root = Path.join(System.tmp_dir!(), "airo_mm_#{System.unique_integer([:positive])}")
      File.mkdir_p!(root)
      File.touch!(Path.join(root, "moondream2-text-model.gguf"))
      proj = Path.join(root, "moondream2-mmproj-f16.gguf")
      File.touch!(proj)
      on_exit(fn -> File.rm_rf!(root) end)

      {:ok, spec} =
        LlamaCpp.launch_spec(
          model(path: Path.join(root, "moondream2-text-model.gguf")),
          %{},
          8081
        )

      assert arg_after(spec.argv, "--mmproj") == proj
    end

    test "no projector ⇒ no --mmproj flag" do
      {:ok, spec} = LlamaCpp.launch_spec(model(path: "/cache/repo/plain.gguf"), %{}, 8081)
      refute "--mmproj" in spec.argv
    end

    test "profile :mmproj overrides auto-detection" do
      {:ok, spec} =
        LlamaCpp.launch_spec(model(), %{mmproj: "/custom/proj.gguf"}, 8081)

      assert arg_after(spec.argv, "--mmproj") == "/custom/proj.gguf"
    end
  end

  describe "resolve_profile/2" do
    test "merges the request over the model defaults so defaults are visible" do
      resolved = LlamaCpp.resolve_profile(model(), %{ctx: 4096})
      assert resolved.ctx == 4096
      # The agent default that would otherwise be invisible:
      assert resolved.parallel == 1
    end

    test "the request wins over a default" do
      assert LlamaCpp.resolve_profile(model(), %{parallel: 1}).parallel == 1
    end
  end

  describe "parse_props/1" do
    test "extracts ctx, parallel, derived ctx_total, and engine build" do
      # Shape of a real llama-server /props response (trimmed).
      body = %{
        "default_generation_settings" => %{"n_ctx" => 36_608, "params" => %{}},
        "total_slots" => 4,
        "build_info" => "b1-9633186",
        "model_path" => "/cache/qwen.gguf",
        "chat_template" => "{{ huge }}"
      }

      assert LlamaCpp.parse_props(body) == %{
               ctx: 36_608,
               parallel: 4,
               # /props has no total-ctx field; derived as ctx × parallel.
               ctx_total: 146_432,
               engine_build: "b1-9633186"
             }
    end

    test "tolerates a body missing the fields" do
      assert LlamaCpp.parse_props(%{}) == %{
               ctx: nil,
               parallel: nil,
               ctx_total: nil,
               engine_build: nil
             }
    end
  end
end
