defmodule AiroAgent.Engine.Vllm do
  @moduledoc """
  vLLM adapter — serves HF *safetensors* repos via an OpenAI-compatible
  `vllm serve` process, spawned in a podman container by the `vllm-slot` wrapper
  (`priv/engine/vllm-slot`). Implements `AiroAgent.Engine`: inventory (with
  HF-snapshot provenance, shared with llama.cpp via `AiroAgent.HFCache`) and pure
  argv construction. It does NOT start processes.

  Two things differ from llama.cpp and are easy to get wrong:

    * **Format.** Inventory scans HF *snapshots* (`config.json` + `*.safetensors`),
      not `*.gguf`. `path` is the snapshot DIRECTORY — vLLM takes a dir.
    * **The ctx contract.** llama.cpp's `-c = ctx × parallel` (contract A) is a
      llama.cpp quirk. vLLM's `--max-model-len` IS the per-request window
      directly; concurrency is continuous batching capped by `--max-num-seqs`.
      So here `ctx → --max-model-len` (NO ×parallel) and `parallel → --max-num-seqs`.

  Capabilities are conservative for now (`:chat`, plus `:vision` when the config
  carries a `vision_config`); finer detection (embeddings) lands later.

  Three safety/mechanism defaults this adapter owns (mechanism, not policy —
  Airo still decides; an explicit profile always wins):

    * **Bare-profile ctx cap.** Without `--max-model-len`, vLLM defaults to the
      model's FULL `max_position_embeddings` (262k on Qwen3.5) and OOM-crashes
      any card whose KV can't hold it. `default_profile/1` caps a bare load at
      `min(ctx_max, 32768)` — the vLLM twin of llama.cpp's parallel-1
      default. Big windows are an explicit opt-in via `profile.ctx`.
    * **Tool-call parser.** The model's chat template dictates the tool-call
      wire format, so `launch_spec/3` sniffs the template and adds
      `--enable-auto-tool-choice --tool-call-parser <matching>` (Qwen3.5/Coder
      XML → `qwen3_xml`; Hermes-style JSON → `hermes`). Without the parser,
      tool calls come back as raw text and Airo's tool loop breaks. Skipped
      when `extra_argv` already carries `--tool-call-parser`, or the template
      format isn't recognized.
    * **`disable_thinking` knob.** The engine-neutral profile key maps here to
      `--default-chat-template-kwargs '{"enable_thinking": false}'` (vLLM
      ≥ 0.10; llama.cpp maps the same knob to `--reasoning off`).

  `runtime_props/1` scrapes the resolved per-request context (`/v1/models`
  `max_model_len`) and engine version (`/version`). `parallel`/`ctx_total` have no
  runtime analogue — vLLM owns paged-KV batching internally (see the ctx contract
  above), so they stay `nil`; the configured `--max-num-seqs` is still visible via
  the slot's resolved `profile`.

  NB unified memory: on a DGX Spark `--gpu-memory-utilization` is a fraction of
  the *shared* ~120 GB pool, so it's left unset by default (vLLM's own default)
  rather than guessed; Airo sets it per Deployment once validated on the box.
  """
  @behaviour AiroAgent.Engine

  alias AiroAgent.{HFCache, ModelRef}
  require Logger

  # Bare-profile --max-model-len cap (see moduledoc). Explicit profile.ctx wins.
  @default_ctx_cap 32_768

  @impl true
  def inventory(opts) do
    roots = Keyword.get(opts, :model_roots, default_roots())

    refs =
      roots
      |> Enum.flat_map(&Path.wildcard(Path.join(&1, "**/snapshots/*/config.json")))
      |> Enum.map(&Path.dirname/1)
      |> Enum.uniq()
      # A repo may ship only GGUF (llama.cpp) — skip snapshots with no weights.
      |> Enum.filter(&has_safetensors?/1)
      |> Enum.map(&ref_from_dir/1)

    {:ok, refs}
  rescue
    e -> {:error, {:inventory_failed, e}}
  end

  @impl true
  def launch_spec(%ModelRef{path: dir} = model, profile, port) do
    profile = resolve_profile(model, profile)
    host = Application.get_env(:airo_agent, :engine_bind_host, "127.0.0.1")

    # --max-model-len IS the per-request window — NO ×parallel (vLLM owns the
    # paged-KV budget internally). Contrast llama.cpp's contract A.
    argv =
      [
        "serve",
        dir,
        # Advertise the id Airo routes on, not the snapshot path.
        "--served-model-name",
        model.id,
        "--host",
        host,
        "--port",
        Integer.to_string(port)
      ] ++
        flag("--max-model-len", profile[:ctx]) ++
        flag("--max-num-seqs", profile[:parallel]) ++
        flag("--tensor-parallel-size", profile[:tensor_parallel_size]) ++
        flag("--gpu-memory-utilization", profile[:gpu_memory_utilization]) ++
        flag("--dtype", profile[:dtype]) ++
        flag("--quantization", profile[:quantization]) ++
        flag("--kv-cache-dtype", profile[:kv_cache_dtype]) ++
        bool_flag("--trust-remote-code", profile[:trust_remote_code]) ++
        thinking_flags(profile[:disable_thinking]) ++
        tool_flags(model, profile) ++
        List.wrap(profile[:extra_argv])

    spec = %{
      argv: Enum.map(argv, &to_string/1),
      # The vllm-slot wrapper reads these to build the podman run (image to use,
      # cache dir to mount). Forwarded explicitly so it can't depend on whatever
      # env the agent happened to inherit.
      env: [
        {"VLLM_IMAGE", Application.get_env(:airo_agent, :vllm_image) || ""},
        {"AIRO_AGENT_MODEL_ROOT", model_root()}
      ],
      readiness: {:http_get, "/health"}
    }

    {:ok, spec}
  end

  @impl true
  def default_profile(%ModelRef{} = model) do
    # Minimal + safe: single GB10 (tp=1), auto dtype, no remote code. parallel /
    # gpu-mem / quantization are left to vLLM's own defaults (and its config
    # auto-detection) unless Airo sets them per Deployment. ctx is the one
    # exception — vLLM's own default is the model's FULL window, which OOMs
    # small cards (see moduledoc), so a bare load is capped.
    %{
      tensor_parallel_size: 1,
      dtype: "auto",
      trust_remote_code: false,
      ctx: default_ctx(model.ctx_max)
    }
  end

  # Unknown ctx_max → no cap to compute; leave --max-model-len to vLLM (it reads
  # the model config itself, which is more than we know here).
  defp default_ctx(nil), do: nil
  defp default_ctx(ctx_max), do: min(ctx_max, @default_ctx_cap)

  @impl true
  def resolve_profile(%ModelRef{} = model, profile) do
    # nil request values mean "unset", not "override the default with nothing" —
    # e.g. a load with ctx: nil must still get the bare-profile ctx cap.
    request = for {k, v} <- profile, v != nil, into: %{}, do: {k, v}
    Map.merge(default_profile(model), request)
  end

  @impl true
  def capabilities(%ModelRef{path: dir}),
    do: dir |> Path.join("config.json") |> read_config() |> capabilities_from_config()

  # --- tool calling / thinking (launch-time identity, like llama.cpp's knobs) ---

  # Engine-neutral disable_thinking knob → vLLM's server-side template kwarg.
  defp thinking_flags(true),
    do: ["--default-chat-template-kwargs", ~s({"enable_thinking": false})]

  defp thinking_flags(_), do: []

  # Auto tool-calling: sniff the chat template for its tool-call wire format and
  # add the matching parser. A profile that already carries --tool-call-parser
  # in extra_argv owns the choice; an unrecognized template gets no flags (vLLM's
  # default — tool calls return as raw text, same as before this existed).
  defp tool_flags(%ModelRef{} = model, profile) do
    extra = profile[:extra_argv] |> List.wrap() |> Enum.map(&to_string/1)

    with false <- "--tool-call-parser" in extra,
         parser when not is_nil(parser) <- tool_parser(model) do
      ["--enable-auto-tool-choice", "--tool-call-parser", parser] ++
        reasoning_parser_flags(model, extra)
    else
      _ -> []
    end
  end

  @doc false
  # The template's literal syntax → the vLLM parser that reads it back.
  #   <function=name><parameter=…>   Qwen3.5 / Qwen3-Coder XML  → qwen3_xml
  #   <tool_call>{"name": …}         Hermes-style JSON (Qwen3)  → hermes
  # Deliberately narrow: e.g. GLM also writes <tool_call> but with <arg_key>
  # pairs hermes can't parse — it matches neither arm and gets no flags.
  def tool_parser_for_template(template) when is_binary(template) do
    cond do
      String.contains?(template, "<function=") ->
        "qwen3_xml"

      String.contains?(template, "<tool_call>") and String.contains?(template, ~s({"name")) ->
        "hermes"

      true ->
        nil
    end
  end

  def tool_parser_for_template(_), do: nil

  defp tool_parser(%ModelRef{path: dir}),
    do: dir |> chat_template() |> tool_parser_for_template()

  # HF snapshots ship the template either as chat_template.jinja (new style) or
  # embedded in tokenizer_config.json.
  defp chat_template(dir) do
    jinja = Path.join(dir, "chat_template.jinja")

    with {:error, _} <- File.read(jinja),
         {:ok, body} <- File.read(Path.join(dir, "tokenizer_config.json")),
         {:ok, %{"chat_template" => t}} when is_binary(t) <- Jason.decode(body) do
      t
    else
      {:ok, template} when is_binary(template) -> template
      _ -> nil
    end
  end

  # Qwen3-family thinking traces (<think>…</think>) need the qwen3 reasoning
  # parser so they land in `reasoning`, not `content`. Harmless when thinking
  # is disabled; other families are left to vLLM's default.
  defp reasoning_parser_flags(%ModelRef{family: "qwen3" <> _}, extra) do
    if "--reasoning-parser" in extra, do: [], else: ["--reasoning-parser", "qwen3"]
  end

  defp reasoning_parser_flags(_model, _extra), do: []

  @doc false
  def capabilities_from_config(config) when is_map(config) do
    base = [:chat]
    if Map.has_key?(config, "vision_config"), do: [:vision | base], else: base
  end

  @doc """
  Best-effort runtime facts from a running vLLM engine: the resolved per-request
  context (`ctx`, from `/v1/models` `max_model_len`) and the engine build
  (`/version`). vLLM owns batching internally, so `parallel`/`ctx_total` have no
  runtime scrape (reported `nil`; the configured `--max-num-seqs` stays visible
  via the slot's resolved `profile`). Each endpoint is fetched independently and
  never raises; missing facts come back `nil`.
  """
  def runtime_props(port) when is_integer(port) do
    base = "http://127.0.0.1:#{port}"

    %{
      ctx: base |> get_json("/v1/models") |> parse_models(),
      parallel: nil,
      ctx_total: nil,
      engine_build: base |> get_json("/version") |> parse_version()
    }
  end

  @doc false
  def parse_models(%{"data" => [%{"max_model_len" => n} | _]}) when is_integer(n), do: n
  def parse_models(_), do: nil

  @doc false
  def parse_version(%{"version" => v}) when is_binary(v), do: v
  def parse_version(_), do: nil

  # Best-effort GET → decoded map body; `%{}` on any non-200, error, or raise so
  # one unreachable endpoint never sinks the other fact.
  defp get_json(base, path) do
    case Req.get(base <> path, retry: false, receive_timeout: 2_000) do
      {:ok, %{status: 200, body: body}} when is_map(body) -> body
      _ -> %{}
    end
  rescue
    _ -> %{}
  end

  @doc """
  Remove any `airo-slot-*` containers left behind by a previous agent process —
  e.g. after a hard crash (SIGKILL), where the container outlives the BEAM
  because it runs in the runtime's cgroup, not the agent's. Best-effort and
  idempotent; meant to run once at boot before anything loads, so a restart
  always starts from a clean slate. No-ops if the runtime isn't installed.
  """
  @spec reap_orphans() :: :ok
  def reap_orphans do
    rt = Application.get_env(:airo_agent, :container_runtime, "podman")

    with {out, 0} <-
           System.cmd(rt, ["ps", "-aq", "--filter", "name=airo-slot-"], stderr_to_stdout: true),
         [_ | _] = ids <- orphan_ids(out) do
      System.cmd(rt, ["rm", "-f" | ids], stderr_to_stdout: true)
      Logger.info("vllm: reaped #{length(ids)} orphan slot container(s) via #{rt}")
    end

    :ok
  rescue
    # runtime binary missing / unexpected output — nothing to reap, never fatal.
    _ -> :ok
  end

  @doc false
  def orphan_ids(ps_output), do: ps_output |> String.split() |> Enum.reject(&(&1 == ""))

  # --- provenance / config ---

  # HF cache layout: .../models--ORG--REPO/snapshots/<sha>/{config.json,*.safetensors}
  defp ref_from_dir(dir) do
    config_path = Path.join(dir, "config.json")
    {repo, revision} = HFCache.repo_and_revision(config_path)
    config = read_config(config_path)
    quant = get_in(config, ["quantization_config", "quant_method"])

    %ModelRef{
      id: build_id(repo, revision, quant),
      repo: repo,
      revision: revision,
      quant: quant,
      family: config["model_type"] || first_arch(config),
      size_bytes: safetensors_bytes(dir),
      ctx_max: ctx_max(config),
      path: dir,
      capabilities: capabilities_from_config(config),
      engine: :vllm
    }
  end

  # VL configs (e.g. Qwen3.5) nest the text model's limits under text_config —
  # a top-level-only read reports ctx_max nil, Airo can't validate capacity, and
  # a bare load OOMs small cards on the model's full window.
  defp ctx_max(config) do
    config["max_position_embeddings"] ||
      get_in(config, ["text_config", "max_position_embeddings"])
  end

  defp read_config(path) do
    with {:ok, body} <- File.read(path),
         {:ok, map} when is_map(map) <- Jason.decode(body) do
      map
    else
      _ -> %{}
    end
  end

  defp first_arch(%{"architectures" => [a | _]}), do: a
  defp first_arch(_), do: nil

  defp build_id(nil, revision, _quant), do: revision || "unknown"
  defp build_id(repo, _revision, nil), do: repo
  defp build_id(repo, _revision, quant), do: "#{repo}:#{quant}"

  defp has_safetensors?(dir), do: Path.wildcard(Path.join(dir, "*.safetensors")) != []

  defp safetensors_bytes(dir) do
    Path.wildcard(Path.join(dir, "*.safetensors"))
    |> Enum.reduce(0, fn f, acc ->
      case File.stat(f) do
        {:ok, %{size: s}} -> acc + s
        _ -> acc
      end
    end)
  end

  defp model_root, do: default_roots() |> List.first()

  defp default_roots do
    Application.get_env(:airo_agent, :model_roots, [Path.expand("~/.cache/huggingface/hub")])
  end

  defp flag(_k, nil), do: []
  defp flag(k, v), do: [k, to_string(v)]

  defp bool_flag(k, true), do: [k]
  defp bool_flag(_k, _), do: []
end
