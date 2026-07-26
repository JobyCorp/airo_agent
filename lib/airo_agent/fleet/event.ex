defmodule AiroAgent.Fleet.Event do
  @moduledoc """
  A **slot** lifecycle transition the agent pushes to Airo (a `"slot"` channel
  message). Carries the slot `port`, the model involved (`resident_model`), and a
  `reason` for the terminal cases.

  - `:loading`  — a model is being loaded into the slot.
  - `:up`       — the model is serving.
  - `:down`     — the resident engine exited unexpectedly (`reason` set).
  - `:failed`   — the engine never became ready (`reason` set).
  - `:unloaded` — the slot was freed on request.
  """

  @type type :: :loading | :up | :down | :failed | :unloaded

  @type t :: %__MODULE__{
          type: type(),
          port: pos_integer(),
          resident_model: String.t() | nil,
          revision: String.t() | nil,
          ctx: pos_integer() | nil,
          parallel: pos_integer() | nil,
          ctx_total: pos_integer() | nil,
          engine_build: String.t() | nil,
          cluster_id: String.t() | nil,
          tp_rank: non_neg_integer() | nil,
          tp_size: pos_integer() | nil,
          reason: term() | nil,
          at: DateTime.t()
        }

  @enforce_keys [:type, :port, :at]
  defstruct [
    :type,
    :port,
    :resident_model,
    :revision,
    :ctx,
    :parallel,
    :ctx_total,
    :engine_build,
    # Multi-node identity, so a peer rank's transition reaches Airo as a push
    # rather than waiting for the next register heartbeat.
    :cluster_id,
    :tp_rank,
    :tp_size,
    :reason,
    :at
  ]
end
