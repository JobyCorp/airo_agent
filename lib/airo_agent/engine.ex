defmodule AiroAgent.Engine do
  @moduledoc """
  The engine-neutral seam. Everything backend-specific lives behind these
  callbacks; process supervision, the HTTP API, and readiness polling are
  generic (see `AiroAgent.Instance`).

  Implementations: `AiroAgent.Engine.LlamaCpp` (now), `.Vllm`, `.Tgi` (later).

  Design rule: an `Engine` callback NEVER spawns or owns a process. It only
  (a) enumerates local models with provenance, and (b) describes how to launch
  one. `AiroAgent.Instance` does the spawning under a `DynamicSupervisor`, so a
  crash in the native engine is isolated to one supervised child — not the VM.
  """

  alias AiroAgent.ModelRef
  require Logger

  @typedoc "Opaque, engine-specific launch knobs. Airo stores this per model and passes it through verbatim."
  @type profile :: map()

  @typedoc "A runnable command the generic supervisor executes via MuonTrap."
  @type launch_spec :: %{
          argv: [String.t()],
          env: [{String.t(), String.t()}],
          # How the supervisor decides the instance is serving.
          readiness: {:http_get, path :: String.t()} | {:log_match, Regex.t()}
        }

  @doc "Scan the host for local models, resolving provenance (revision/quant/size/ctx)."
  @callback inventory(opts :: keyword()) :: {:ok, [ModelRef.t()]} | {:error, term()}

  @doc "Build the argv/env to serve `model` on `port` with `profile`. Pure — no side effects."
  @callback launch_spec(ModelRef.t(), profile(), port :: pos_integer()) ::
              {:ok, launch_spec()} | {:error, term()}

  @doc "Sane default launch profile for a model (e.g. flash-attn on, q8_0 KV, MTP when present)."
  @callback default_profile(ModelRef.t()) :: profile()

  @doc """
  Merge `profile` over the model's defaults → the **effective** launch profile the
  engine will run with (pure, no side effects). Optional; reported on the slot so
  Airo/UI sees the defaults actually in force (e.g. `parallel: 4`) rather than a
  blank. Defaults to the requested profile when an engine doesn't implement it.
  """
  @callback resolve_profile(ModelRef.t(), profile()) :: profile()

  @doc "Capabilities this engine can serve for `model` (e.g. [:chat, :embeddings, :vision])."
  @callback capabilities(ModelRef.t()) :: [atom()]

  @doc """
  Multi-node identity for a load, or `nil` when it runs on one host (pure).

  Optional; only engines that can span hosts implement it. `tp_size` is the
  number of **hosts**, not the tensor-parallel degree — Airo divides the model's
  weights by it to charge each host its share.
  """
  @callback cluster_info(profile(), port :: pos_integer()) ::
              %{cluster_id: String.t(), tp_rank: non_neg_integer(), tp_size: pos_integer()} | nil

  @doc """
  The profile keys this engine actually reads (pure).

  `POST /load` accepts the **union** of every engine's keys
  (`AiroAgent.Api.Router.profile_keys/0`), so a profile written for one backend
  is accepted verbatim by another and its foreign keys silently do nothing —
  `temperature` on vLLM is the live example: llama.cpp maps it to `--temp`,
  vLLM maps it nowhere at all. Declaring what an adapter honours lets
  `AiroAgent.Instance` log the difference once at launch, so the gap is visible
  instead of silent.

  This is deliberately NOT an error: dropping a foreign key is what makes one
  profile portable across hosts. Optional — an adapter that doesn't implement it
  is simply never reported on.
  """
  @callback honored_profile_keys() :: [atom()]

  @doc """
  Best-effort facts scraped from a **running** engine — what it is actually
  serving with, as opposed to what was requested. Called once at readiness by
  `AiroAgent.Instance` and merged onto the slot.

  Recognised keys: `ctx`, `parallel`, `ctx_total`, `engine_build`. An engine that
  has no runtime analogue for one reports it `nil` (vLLM owns batching
  internally, so `parallel`/`ctx_total` come from the profile instead — see
  `AiroAgent.Fleet`). Must never raise: return `%{}` when the engine is
  unreachable. Optional.
  """
  @callback runtime_props(port :: pos_integer()) :: map()

  @doc """
  Reclaim engine children left behind by a *previous* agent process, before
  anything loads.

  Only engines whose children can outlive the BEAM need this: a container runs in
  the runtime's cgroup rather than ours, whereas a `MuonTrap` child is reaped
  with the VM. Best-effort and idempotent — run once at boot, and never fatal.
  Optional.
  """
  @callback reap_orphans() :: :ok

  @optional_callbacks resolve_profile: 2,
                      cluster_info: 2,
                      honored_profile_keys: 0,
                      runtime_props: 1,
                      reap_orphans: 0

  # --- Dispatch helpers: resolve the configured/per-model engine adapter. ---

  @adapters %{llama_cpp: AiroAgent.Engine.LlamaCpp, vllm: AiroAgent.Engine.Vllm}

  @spec adapter(atom()) :: module()
  def adapter(engine), do: Map.fetch!(@adapters, engine)

  @doc """
  Whether `adapter` implements an optional callback, loading it first.

  `function_exported?/3` answers `false` for a module that merely hasn't been
  loaded yet, which is exactly the state adapters are in during boot under
  `:interactive` code loading (dev/test) — so the orphan sweep would silently
  skip every engine. `Code.ensure_loaded?/1` closes that.
  """
  @spec exports?(module(), atom(), arity()) :: boolean()
  def exports?(adapter, fun, arity),
    do: Code.ensure_loaded?(adapter) and function_exported?(adapter, fun, arity)

  @doc "Whether an engine has a registered adapter (i.e. `adapter/1` won't raise)."
  @spec known?(atom()) :: boolean()
  def known?(engine), do: Map.has_key?(@adapters, engine)

  @spec inventory_all(keyword()) :: {:ok, [ModelRef.t()]} | {:error, term()}
  def inventory_all(opts \\ []) do
    # `AIRO_AGENT_ENGINES` is host-set, so guard against an engine with no adapter
    # (e.g. `:vllm` before that adapter lands): skip + warn rather than crash the
    # whole inventory scan on a Map.fetch!/2.
    {known, unknown} =
      Application.get_env(:airo_agent, :engines, [:llama_cpp])
      |> Enum.split_with(&known?/1)

    Enum.each(unknown, fn engine ->
      Logger.warning(
        "airo_agent: ignoring unknown engine #{inspect(engine)} — no adapter registered"
      )
    end)

    known
    |> Enum.map(&adapter/1)
    |> Enum.reduce_while({:ok, []}, fn mod, {:ok, acc} ->
      case mod.inventory(opts) do
        {:ok, refs} -> {:cont, {:ok, acc ++ refs}}
        {:error, _} = err -> {:halt, err}
      end
    end)
  end
end
