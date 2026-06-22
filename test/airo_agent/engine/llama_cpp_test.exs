defmodule AiroAgent.Engine.LlamaCppTest do
  use ExUnit.Case, async: true

  alias AiroAgent.Engine.LlamaCpp

  describe "parse_props/1" do
    test "extracts the per-request ctx, parallel sequences, and engine build" do
      # Shape of a real llama-server /props response (trimmed).
      body = %{
        "default_generation_settings" => %{"n_ctx" => 65_536, "params" => %{}},
        "total_slots" => 4,
        "build_info" => "b1-9633186",
        "model_path" => "/cache/qwen.gguf",
        "chat_template" => "{{ huge }}"
      }

      assert LlamaCpp.parse_props(body) == %{
               ctx: 65_536,
               parallel: 4,
               engine_build: "b1-9633186"
             }
    end

    test "tolerates a body missing the fields" do
      assert LlamaCpp.parse_props(%{}) == %{ctx: nil, parallel: nil, engine_build: nil}
    end
  end
end
