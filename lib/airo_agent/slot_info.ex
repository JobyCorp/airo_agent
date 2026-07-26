defmodule AiroAgent.SlotInfo do
  @moduledoc """
  Public view of a serving **slot** (Model 2), reported to Airo in the channel
  `register` payload and `GET /slots`.

  A slot is a stable port that holds at most one resident model. Airo serves
  inference directly at `base_url` — the agent is never on that path. `status` is
  the steady state (`:empty | :loading | :up`); terminal transitions (down,
  unloaded) are `AiroAgent.Fleet.Event`s, not slot states.

  ## Multi-node loads

  A model too large for one host runs as one logical load spanning a slot on each
  participating host. Every rank reports `cluster_id` (shared) plus its own
  `tp_rank` and the group's `tp_size`, so Airo can join them back together and
  tell that a host's VRAM is spoken for.

  **`tp_size` is the number of *hosts* (`--nnodes`), not the tensor-parallel
  degree (`--tensor-parallel-size`).** On a 2×GB10 pair both happen to be 2, so
  either reads correctly there — but they diverge on a 2-node × 4-GPU host, and
  Airo divides a model's weights by `tp_size` to charge each host its share.
  A host holds 1/nnodes of the weights.

  Rank 0 is the head and the only rank serving the API; `base_url` on a peer is
  reported because Airo requires an absolute URL, but nothing answers there.
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
          # serving with; `parallel` is how many sequences share it; `ctx_total`
          # is the total KV budget (`-c` = ctx × parallel) the engine allocated.
          ctx: pos_integer() | nil,
          parallel: pos_integer() | nil,
          ctx_total: pos_integer() | nil,
          engine_build: String.t() | nil,
          # The effective launch profile (defaults applied) the slot is running —
          # so the agent's defaults (e.g. parallel: 4) are never invisible.
          profile: map(),
          # Multi-node identity; all nil for an ordinary single-host slot.
          cluster_id: String.t() | nil,
          tp_rank: non_neg_integer() | nil,
          tp_size: pos_integer() | nil,
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
    :ctx_total,
    :engine_build,
    :cluster_id,
    :tp_rank,
    :tp_size,
    :started_at,
    profile: %{}
  ]

  @doc """
  Wire form for Airo (channel `register` and `GET /slots`).

  `cluster_id` goes out as **`deployment_id`** — that is the field name Airo
  reads. It is a *load* id and unrelated to Airo's own `Deployment` records,
  which is why Airo renames it back to `cluster_id` on the way in.
  """
  @spec to_payload(t()) :: map()
  def to_payload(%__MODULE__{} = slot) do
    slot
    |> Map.from_struct()
    |> Map.delete(:cluster_id)
    |> Map.put(:deployment_id, slot.cluster_id)
  end
end
