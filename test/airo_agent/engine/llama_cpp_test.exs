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

    test "parallel defaults make a bare ctx the per-request window (÷ the default)" do
      # No explicit parallel ⇒ default_profile's parallel: 4 applies, so a request
      # for 36k per-request allocates -c = 36k × 4.
      {:ok, spec} = LlamaCpp.launch_spec(model(), %{ctx: 36_608}, 8081)
      assert arg_after(spec.argv, "-c") == "146432"
      assert arg_after(spec.argv, "--parallel") == "4"
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

  describe "resolve_profile/2" do
    test "merges the request over the model defaults so defaults are visible" do
      resolved = LlamaCpp.resolve_profile(model(), %{ctx: 4096})
      assert resolved.ctx == 4096
      # The agent default that would otherwise be invisible:
      assert resolved.parallel == 4
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
