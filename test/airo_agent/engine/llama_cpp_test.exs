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
