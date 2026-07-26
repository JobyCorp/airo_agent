defmodule AiroAgent.Api.Schemas do
  @moduledoc """
  OpenAPI schemas for the control API — the published contract Airo codes
  against. Each schema mirrors exactly what the router serializes (see
  `AiroAgent.Api.Router`): `ModelRef` from `/inventory`, `SlotInfo` from
  `/slots` and `/load`, the GPU snapshot from `/gpu`, and the load/unload bodies.
  """

  alias OpenApiSpex.Schema

  defmodule ModelRef do
    @moduledoc false
    require OpenApiSpex

    OpenApiSpex.schema(%{
      title: "ModelRef",
      description: "A local model artifact with Hugging Face provenance.",
      type: :object,
      properties: %{
        id: %Schema{
          type: :string,
          description: "Stable id Airo routes on.",
          example: "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL"
        },
        repo: %Schema{type: :string, nullable: true, description: "Hugging Face repo."},
        revision: %Schema{
          type: :string,
          nullable: true,
          description: "HF snapshot commit sha — the provenance field.",
          example: "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d"
        },
        quant: %Schema{type: :string, nullable: true, example: "UD-Q4_K_XL"},
        family: %Schema{type: :string, nullable: true, example: "qwen35moe"},
        size_bytes: %Schema{type: :integer, nullable: true},
        ctx_max: %Schema{
          type: :integer,
          nullable: true,
          description: "Model's max trained context (distinct from a slot's serving ctx)."
        },
        path: %Schema{
          type: :string,
          description: "Absolute path to the GGUF (first shard if split)."
        },
        capabilities: %Schema{type: :array, items: %Schema{type: :string}, example: ["chat"]},
        engine: %Schema{type: :string, example: "llama_cpp"}
      },
      required: [:id, :path, :engine]
    })
  end

  defmodule SlotInfo do
    @moduledoc false
    require OpenApiSpex

    OpenApiSpex.schema(%{
      title: "SlotInfo",
      description: "A serving slot (a stable port holding at most one resident model).",
      type: :object,
      properties: %{
        port: %Schema{type: :integer, example: 8081},
        base_url: %Schema{
          type: :string,
          description: "Real OpenAI-compatible serving endpoint Airo routes to.",
          example: "http://192.168.68.85:8081/v1"
        },
        status: %Schema{type: :string, enum: ["empty", "loading", "up"]},
        resident_model: %Schema{type: :string, nullable: true},
        revision: %Schema{type: :string, nullable: true},
        ctx: %Schema{
          type: :integer,
          nullable: true,
          description: "Per-request context window the engine is actually serving with."
        },
        parallel: %Schema{
          type: :integer,
          nullable: true,
          description: "Concurrent sequences sharing the slot."
        },
        ctx_total: %Schema{
          type: :integer,
          nullable: true,
          description: "Total KV budget allocated (-c) = ctx × parallel."
        },
        engine_build: %Schema{type: :string, nullable: true, example: "b1-9633186"},
        deployment_id: %Schema{
          type: :string,
          nullable: true,
          description:
            "Shared id of the multi-node load this slot is a rank of; null for a " <>
              "single-host slot. A *load* id — unrelated to Airo's Deployment records.",
          example: "dep-7f3a1c2b"
        },
        tp_rank: %Schema{
          type: :integer,
          nullable: true,
          description:
            "This host's index in the load (--node-rank). 0 is the head, and the only rank serving the API.",
          example: 1
        },
        tp_size: %Schema{
          type: :integer,
          nullable: true,
          description:
            "Number of HOSTS in the load (--nnodes), not the tensor-parallel degree. " <>
              "Each host holds 1/tp_size of the weights.",
          example: 2
        },
        profile: %Schema{
          type: :object,
          additionalProperties: true,
          description: "Effective launch profile (defaults applied)."
        },
        started_at: %Schema{type: :string, format: :"date-time", nullable: true}
      },
      required: [:port, :base_url, :status]
    })
  end

  defmodule GpuSnapshot do
    @moduledoc false
    require OpenApiSpex

    OpenApiSpex.schema(%{
      title: "GpuSnapshot",
      description: "Host GPU telemetry from nvidia-smi. Only `available` is guaranteed.",
      type: :object,
      properties: %{
        available: %Schema{type: :boolean},
        vram_used_mb: %Schema{type: :number, nullable: true},
        vram_total_mb: %Schema{type: :number, nullable: true},
        util_pct: %Schema{type: :number, nullable: true},
        power_draw_w: %Schema{type: :number, nullable: true},
        power_limit_w: %Schema{type: :number, nullable: true}
      },
      required: [:available]
    })
  end

  defmodule LoadRequest do
    @moduledoc false
    require OpenApiSpex

    OpenApiSpex.schema(%{
      title: "LoadRequest",
      description:
        "Place a model into a slot. `profile` carries launch knobs (ctx, parallel, …).",
      type: :object,
      properties: %{
        model: %Schema{type: :string, description: "ModelRef id from /inventory."},
        slot: %Schema{type: :integer, description: "Slot port to load into."},
        profile: %Schema{type: :object, additionalProperties: true}
      },
      required: [:model, :slot],
      example: %{
        "model" => "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL",
        "slot" => 8081,
        "profile" => %{"ctx" => 146_432, "parallel" => 1}
      }
    })
  end

  defmodule UnloadRequest do
    @moduledoc false
    require OpenApiSpex

    OpenApiSpex.schema(%{
      title: "UnloadRequest",
      type: :object,
      properties: %{slot: %Schema{type: :integer, description: "Slot port to free."}},
      required: [:slot]
    })
  end

  defmodule Models do
    @moduledoc false
    require OpenApiSpex

    OpenApiSpex.schema(%{
      title: "Models",
      type: :object,
      properties: %{models: %Schema{type: :array, items: ModelRef}},
      required: [:models]
    })
  end

  defmodule Slots do
    @moduledoc false
    require OpenApiSpex

    OpenApiSpex.schema(%{
      title: "Slots",
      type: :object,
      properties: %{slots: %Schema{type: :array, items: SlotInfo}},
      required: [:slots]
    })
  end

  defmodule Health do
    @moduledoc false
    require OpenApiSpex

    OpenApiSpex.schema(%{
      title: "Health",
      type: :object,
      properties: %{status: %Schema{type: :string, example: "ok"}},
      required: [:status]
    })
  end

  defmodule Ok do
    @moduledoc false
    require OpenApiSpex

    OpenApiSpex.schema(%{
      title: "Ok",
      type: :object,
      properties: %{ok: %Schema{type: :boolean}},
      required: [:ok]
    })
  end

  defmodule Error do
    @moduledoc false
    require OpenApiSpex

    OpenApiSpex.schema(%{
      title: "Error",
      type: :object,
      properties: %{
        error: %Schema{type: :string},
        model: %Schema{type: :string, nullable: true}
      },
      required: [:error]
    })
  end
end
