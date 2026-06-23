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
  carries a `vision_config`); finer detection (embeddings) and the `runtime_props`
  scrape land in Phase 2.

  NB unified memory: on a DGX Spark `--gpu-memory-utilization` is a fraction of
  the *shared* ~120 GB pool, so it's left unset by default (vLLM's own default)
  rather than guessed; Airo sets it per Deployment once validated on the box.
  """
  @behaviour AiroAgent.Engine

  alias AiroAgent.{HFCache, ModelRef}

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
  def default_profile(%ModelRef{}) do
    # Minimal + safe: single GB10 (tp=1), auto dtype, no remote code. ctx /
    # parallel / gpu-mem / quantization are left to vLLM's own defaults (and its
    # config auto-detection) unless Airo sets them per Deployment.
    %{
      tensor_parallel_size: 1,
      dtype: "auto",
      trust_remote_code: false
    }
  end

  @impl true
  def resolve_profile(%ModelRef{} = model, profile),
    do: Map.merge(default_profile(model), profile)

  @impl true
  def capabilities(%ModelRef{path: dir}),
    do: dir |> Path.join("config.json") |> read_config() |> capabilities_from_config()

  @doc false
  def capabilities_from_config(config) when is_map(config) do
    base = [:chat]
    if Map.has_key?(config, "vision_config"), do: [:vision | base], else: base
  end

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
      ctx_max: config["max_position_embeddings"],
      path: dir,
      capabilities: capabilities_from_config(config),
      engine: :vllm
    }
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
