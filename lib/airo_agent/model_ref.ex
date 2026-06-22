defmodule AiroAgent.ModelRef do
  @moduledoc """
  Identity + provenance of a *local* model artifact.

  The whole reason `airo_agent` exists: surface the fields Ollama/LM Studio
  never tell Airo — most importantly `revision` (the Hugging Face snapshot
  commit sha), so Airo can fill `VERSION`/`REVISION`, detect upstream updates,
  and attribute performance per revision.
  """

  @type t :: %__MODULE__{
          # Stable id Airo routes on, e.g. "unsloth/Qwen3.5-122B-A10B-MTP-GGUF:UD-Q4_K_XL".
          id: String.t(),
          # HF repo, e.g. "unsloth/Qwen3.5-122B-A10B-MTP-GGUF".
          repo: String.t() | nil,
          # HF snapshot commit sha — THE provenance field. From the cache path:
          # .../snapshots/<sha>/<file>.gguf
          revision: String.t() | nil,
          quant: String.t() | nil,
          family: String.t() | nil,
          size_bytes: non_neg_integer() | nil,
          ctx_max: pos_integer() | nil,
          # Absolute path(s) to the GGUF (first shard for split models).
          path: String.t(),
          capabilities: [atom()],
          engine: atom()
        }

  @enforce_keys [:id, :path, :engine]
  defstruct id: nil,
            repo: nil,
            revision: nil,
            quant: nil,
            family: nil,
            size_bytes: nil,
            ctx_max: nil,
            path: nil,
            capabilities: [],
            engine: :llama_cpp
end
