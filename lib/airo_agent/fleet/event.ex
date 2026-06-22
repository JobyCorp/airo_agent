defmodule AiroAgent.Fleet.Event do
  @moduledoc """
  A lifecycle transition the agent pushes to Airo (the payload `AiroAgent.Notifier`
  delivers). `:loading`/`:up` carry the live `InstanceInfo`; the terminal events
  carry the last snapshot plus a `reason`.

  - `:loading`  — instance started, engine up, not yet serving.
  - `:up`       — readiness predicate satisfied; serving.
  - `:down`     — was serving, then exited unexpectedly (crash/OOM). `reason` set.
  - `:failed`   — exited before ever becoming ready. `reason` set.
  - `:unloaded` — terminated on Airo's request. Not a failure; `reason` is nil.

  `:down`/`:failed` are EDGES — once emitted the instance is gone from
  `AiroAgent.Fleet.running/0`; Airo also reconciles down-ness from absence in the
  next snapshot, so a dropped event self-heals.
  """

  alias AiroAgent.InstanceInfo

  @type type :: :loading | :up | :down | :failed | :unloaded

  @type t :: %__MODULE__{
          type: type(),
          model_id: String.t(),
          info: InstanceInfo.t() | nil,
          reason: term() | nil,
          at: DateTime.t()
        }

  @enforce_keys [:type, :model_id, :at]
  defstruct [:type, :model_id, :info, :reason, :at]
end
