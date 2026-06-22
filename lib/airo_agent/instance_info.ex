defmodule AiroAgent.InstanceInfo do
  @moduledoc """
  Public view of a running engine instance, reported to Airo via `GET /running`
  and `POST /load`. Airo points its OpenAI client straight at `{host, port}` —
  the agent is never on the data path.
  """

  # A *running* instance is either still loading or serving. Terminal transitions
  # (:down/:failed/:unloaded) are `AiroAgent.Fleet.Event`s, not instance states.
  @type status :: :loading | :up

  @type t :: %__MODULE__{
          model_id: String.t(),
          revision: String.t() | nil,
          engine: atom(),
          host: String.t(),
          port: pos_integer(),
          status: status(),
          # OpenAI base airo should call, e.g. "http://10.0.0.5:51817/v1".
          base_url: String.t(),
          vram_mb: non_neg_integer() | nil,
          started_at: DateTime.t() | nil
        }

  @enforce_keys [:model_id, :engine, :host, :port, :status]
  defstruct model_id: nil,
            revision: nil,
            engine: :llama_cpp,
            host: "127.0.0.1",
            port: nil,
            status: :loading,
            base_url: nil,
            vram_mb: nil,
            started_at: nil
end
