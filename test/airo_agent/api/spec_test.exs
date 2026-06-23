defmodule AiroAgent.Api.SpecTest do
  use ExUnit.Case, async: true
  import Plug.Test

  alias AiroAgent.Api.{Router, Schemas, Spec}
  alias OpenApiSpex.OpenApi

  test "spec/0 builds and covers every control route" do
    assert %OpenApi{} = spec = Spec.spec()

    assert spec.paths |> Map.keys() |> Enum.sort() ==
             ~w(/gpu /health /inventory /inventory/refresh /load /slots /unload)
  end

  test "GET /openapi serves the spec as JSON" do
    conn = Router.call(conn(:get, "/openapi"), Router.init([]))

    assert conn.status == 200
    body = Jason.decode!(conn.resp_body)
    assert is_binary(body["openapi"])
    assert body["info"]["title"] == "airo_agent control API"
    assert Map.has_key?(body["paths"], "/load")
    assert body["paths"]["/load"]["post"]["operationId"] == "load"
  end

  # The schemas must match what the router actually serializes, or the published
  # contract lies. Cast the real struct → wire form through each schema.
  describe "schemas match the router's serialization" do
    test "SlotInfo (from /slots, /load)" do
      wire =
        %AiroAgent.SlotInfo{
          port: 8081,
          base_url: "http://192.168.68.85:8081/v1",
          status: :up,
          resident_model: "org/repo:Q4",
          revision: "abc123",
          ctx: 36_608,
          parallel: 4,
          ctx_total: 146_432,
          engine_build: "b1-9633186",
          profile: %{ctx: 36_608, parallel: 4},
          started_at: ~U[2026-06-22 00:00:00Z]
        }
        |> Map.from_struct()
        |> roundtrip()

      assert {:ok, _} = OpenApiSpex.cast_value(wire, Schemas.SlotInfo.schema())
    end

    test "ModelRef (from /inventory)" do
      wire =
        %AiroAgent.ModelRef{
          id: "org/repo:Q4",
          repo: "org/repo",
          revision: "abc123",
          quant: "Q4",
          family: "qwen",
          size_bytes: 123,
          ctx_max: 262_144,
          path: "/m.gguf",
          capabilities: [:chat],
          engine: :llama_cpp
        }
        |> Map.from_struct()
        # router's model_json/1 stringifies capabilities
        |> Map.update!(:capabilities, &Enum.map(&1, fn c -> to_string(c) end))
        |> roundtrip()

      assert {:ok, _} = OpenApiSpex.cast_value(wire, Schemas.ModelRef.schema())
    end

    test "GpuSnapshot (available and unavailable)" do
      for snap <- [
            %{
              available: true,
              vram_used_mb: 27_815.0,
              vram_total_mb: 32_607.0,
              util_pct: 7.0,
              power_draw_w: 44.0,
              power_limit_w: 600.0
            },
            %{available: false}
          ] do
        assert {:ok, _} = OpenApiSpex.cast_value(roundtrip(snap), Schemas.GpuSnapshot.schema())
      end
    end
  end

  # JSON round-trip = exactly what crosses the wire (atom keys/values → strings).
  defp roundtrip(map), do: map |> Jason.encode!() |> Jason.decode!()
end
