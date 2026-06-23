defmodule AiroAgent.Api.Spec do
  @moduledoc """
  The published OpenAPI spec for the control API, served at `GET /openapi`.

  This is the machine-readable contract Airo codes against (`Airo.Agents` HTTP
  client). It is the **management** surface only — never the inference data path,
  which goes directly to a slot's `base_url`.
  """

  @behaviour OpenApiSpex.OpenApi

  alias OpenApiSpex.{Info, MediaType, OpenApi, Operation, PathItem, RequestBody, Response}
  alias AiroAgent.Api.Schemas

  @impl OpenApiSpex.OpenApi
  def spec do
    %OpenApi{
      info: %Info{
        title: "airo_agent control API",
        version: to_string(Application.spec(:airo_agent, :vsn) || "0.0.0"),
        description:
          "Per-host control plane: model inventory (with provenance), slot " <>
            "load/unload/swap, and GPU telemetry. Management only — inference goes " <>
            "directly to a slot's base_url, never through here."
      },
      paths: %{
        "/health" => %PathItem{
          get: %Operation{
            operationId: "health",
            summary: "Liveness probe.",
            tags: ["health"],
            responses: %{200 => json_resp("Agent is up", Schemas.Health)}
          }
        },
        "/inventory" => %PathItem{
          get: %Operation{
            operationId: "inventory",
            summary: "Local models with provenance (revision).",
            tags: ["inventory"],
            responses: %{200 => json_resp("The local model catalog", Schemas.Models)}
          }
        },
        "/inventory/refresh" => %PathItem{
          post: %Operation{
            operationId: "refreshInventory",
            summary: "Rescan the local catalog and return it.",
            tags: ["inventory"],
            responses: %{200 => json_resp("The rescanned catalog", Schemas.Models)}
          }
        },
        "/slots" => %PathItem{
          get: %Operation{
            operationId: "slots",
            summary: "Serving slots with resident model and runtime facts.",
            tags: ["slots"],
            responses: %{200 => json_resp("The host's slots", Schemas.Slots)}
          }
        },
        "/gpu" => %PathItem{
          get: %Operation{
            operationId: "gpu",
            summary: "Host GPU telemetry snapshot.",
            tags: ["telemetry"],
            responses: %{200 => json_resp("GPU snapshot", Schemas.GpuSnapshot)}
          }
        },
        "/load" => %PathItem{
          post: %Operation{
            operationId: "load",
            summary: "Place (or swap) a model into a slot.",
            tags: ["slots"],
            requestBody: json_body("Model + slot + optional launch profile", Schemas.LoadRequest),
            responses: %{
              200 => json_resp("The slot, now :loading", Schemas.SlotInfo),
              400 => json_resp("Missing/invalid model or slot", Schemas.Error),
              404 => json_resp("Unknown model id", Schemas.Error),
              422 => json_resp("Load could not start", Schemas.Error)
            }
          }
        },
        "/unload" => %PathItem{
          post: %Operation{
            operationId: "unload",
            summary: "Free a slot.",
            tags: ["slots"],
            requestBody: json_body("The slot to free", Schemas.UnloadRequest),
            responses: %{
              200 => json_resp("Slot freed", Schemas.Ok),
              400 => json_resp("Missing/invalid slot", Schemas.Error),
              404 => json_resp("Unknown slot", Schemas.Error)
            }
          }
        }
      }
    }
    |> OpenApiSpex.resolve_schema_modules()
  end

  defp json_resp(description, schema) do
    %Response{description: description, content: %{"application/json" => %MediaType{schema: schema}}}
  end

  defp json_body(description, schema) do
    %RequestBody{
      description: description,
      required: true,
      content: %{"application/json" => %MediaType{schema: schema}}
    }
  end
end
