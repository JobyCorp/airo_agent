defmodule AiroAgent.SlotInfo do
  @moduledoc """
  Public view of a serving **slot** (Model 2), reported to Airo in the channel
  `register` payload and `GET /slots`.

  A slot is a stable port that holds at most one resident model. Airo serves
  inference directly at `base_url` — the agent is never on that path. `status` is
  the steady state (`:empty | :loading | :up`); terminal transitions (down,
  unloaded) are `AiroAgent.Fleet.Event`s, not slot states.
  """

  @type status :: :empty | :loading | :up

  @type t :: %__MODULE__{
          port: pos_integer(),
          base_url: String.t(),
          status: status(),
          resident_model: String.t() | nil,
          revision: String.t() | nil,
          # Runtime facts from the engine (when :up) — distinct from the model's
          # ctx_max. `ctx` is the per-request context the engine is actually
          # serving with; `parallel` is how many sequences share it.
          ctx: pos_integer() | nil,
          parallel: pos_integer() | nil,
          engine_build: String.t() | nil,
          started_at: DateTime.t() | nil
        }

  @enforce_keys [:port, :base_url, :status]
  defstruct [
    :port,
    :base_url,
    :status,
    :resident_model,
    :revision,
    :ctx,
    :parallel,
    :engine_build,
    :started_at
  ]
end
