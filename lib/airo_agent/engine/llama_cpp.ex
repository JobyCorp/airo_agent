defmodule AiroAgent.Engine.LlamaCpp do
  @moduledoc """
  llama.cpp adapter — wraps a *bare* `llama-server` from your own build, so the
  bleeding-edge MTP path (`--spec-type draft-mtp`) is preserved.

  Implements `AiroAgent.Engine`: inventory (with HF-snapshot provenance) and
  pure argv/env construction. It does NOT start processes.
  """

  @behaviour AiroAgent.Engine

  alias AiroAgent.ModelRef

  @impl true
  def inventory(opts) do
    roots = Keyword.get(opts, :model_roots, default_roots())

    refs =
      roots
      |> Enum.flat_map(&Path.wildcard(Path.join(&1, "**/*.gguf")))
      # split-shard models: keep only shard 00001 / unsharded files.
      |> Enum.reject(&shard_continuation?/1)
      # *mmproj*.gguf are vision PROJECTORS — companions loaded WITH a base model
      # via --mmproj, not standalone servable models. Listed, they inventory as
      # bogus `family=clip` rows Airo would try to load. Their presence is instead
      # surfaced as the sibling model's :vision capability (see caps_for/1).
      |> Enum.reject(&projector?/1)
      |> Enum.map(&ref_from_path/1)

    {:ok, refs}
  rescue
    e -> {:error, {:inventory_failed, e}}
  end

  @impl true
  def launch_spec(%ModelRef{path: path} = model, profile, port) do
    profile = resolve_profile(model, profile)
    host = Application.get_env(:airo_agent, :engine_bind_host, "127.0.0.1")

    # Tool-use + thinking knobs (launch-time identity; restart-on-change):
    # Vision models ship a separate mmproj projector alongside the text GGUF
    # (e.g. moondream2); llama-server loads the pair via --mmproj. Auto-attach
    # the sibling projector (profile :mmproj overrides), so a multimodal model
    # actually serves vision instead of loading text-only.
    argv =
      ["-m", path, "--host", host, "--port", Integer.to_string(port)] ++
        flag("--mmproj", profile[:mmproj] || mmproj_sibling(path)) ++
        flag("-c", total_ctx(profile)) ++
        flag("-ngl", profile[:ngl]) ++
        flag("-ncmoe", profile[:n_cpu_moe]) ++
        flag("--flash-attn", profile[:flash_attn]) ++
        flag("--cache-type-k", profile[:cache_type_k]) ++
        flag("--cache-type-v", profile[:cache_type_v]) ++
        flag("--spec-type", profile[:spec_type]) ++
        flag("--parallel", profile[:parallel]) ++
        bool_flag("--jinja", profile[:jinja]) ++
        flag("--chat-template", profile[:chat_template]) ++
        flag("--reasoning-budget", profile[:reasoning_budget]) ++
        flag("--reasoning-format", profile[:reasoning_format]) ++
        List.wrap(profile[:extra_argv])

    env =
      case profile[:ld_library_path] do
        nil -> []
        p -> [{"LD_LIBRARY_PATH", p}]
      end

    spec = %{
      argv: Enum.map(argv, &to_string/1),
      env: env,
      readiness: {:http_get, "/health"}
    }

    {:ok, spec}
  end

  @impl true
  def default_profile(%ModelRef{} = model) do
    %{
      ngl: 999,
      flash_attn: "on",
      cache_type_k: "q8_0",
      cache_type_v: "q8_0",
      # Default 1, NOT 4: under contract A the engine's `-c` is `ctx × parallel`,
      # so parallel MULTIPLIES VRAM (each sequence is another full `ctx` of KV).
      # A default of 4 silently quadruples a bare `ctx` and OOM-crashes the engine
      # on load. Concurrency is an explicit, VRAM-aware opt-in (Airo sets it per
      # Deployment); the safe default gives one request exactly the `ctx` asked.
      parallel: 1,
      # On by default: use the model's embedded chat template so tool calls parse
      # (unsloth Qwen GGUFs ship one). Airo can override per-profile with false.
      jinja: true,
      # MTP only when the GGUF actually ships an MTP head.
      spec_type: if(mtp?(model), do: "draft-mtp", else: nil),
      ld_library_path: Application.get_env(:airo_agent, :llama_cpp_lib_path)
    }
  end

  @impl true
  def resolve_profile(%ModelRef{} = model, profile),
    do: Map.merge(default_profile(model), profile)

  @impl true
  def capabilities(%ModelRef{path: path}), do: caps_for(path)

  # :vision when a multimodal projector sits beside the model. Matches any
  # *mmproj*.gguf — naming differs by publisher (unsloth `mmproj-F16.gguf`,
  # ggml-org `moondream2-mmproj-f16-*.gguf`).
  defp caps_for(path) do
    if mmproj_sibling(path), do: [:vision, :chat], else: [:chat]
  end

  # The mmproj projector file beside `path`, or nil. Used both to flag :vision and
  # to pass --mmproj at launch.
  defp mmproj_sibling(path) do
    path
    |> Path.dirname()
    |> Path.join("*.gguf")
    |> Path.wildcard()
    |> Enum.find(&projector?/1)
  end

  @doc """
  Best-effort runtime facts from a running engine's `/props` — the per-request
  context (`ctx`), parallel sequences (`parallel`), and engine build. These are
  what the engine is *actually* serving with, distinct from the model's ctx_max.
  Never raises; returns `%{}` if the engine is unreachable.
  """
  def runtime_props(port) when is_integer(port) do
    case Req.get("http://127.0.0.1:#{port}/props", retry: false, receive_timeout: 2_000) do
      {:ok, %{status: 200, body: body}} when is_map(body) -> parse_props(body)
      _ -> %{}
    end
  rescue
    _ -> %{}
  end

  @doc false
  def parse_props(body) when is_map(body) do
    gen = Map.get(body, "default_generation_settings", %{})
    ctx = Map.get(gen, "n_ctx")
    parallel = Map.get(body, "total_slots")

    %{
      # Per-request window the engine is actually serving with.
      ctx: ctx,
      parallel: parallel,
      # Total KV budget allocated (`-c`) = per-request × parallel. /props has no
      # field for it, so derive it; distinct from the model's ctx_max.
      ctx_total: ctx_total(ctx, parallel),
      engine_build: Map.get(body, "build_info")
    }
  end

  defp ctx_total(ctx, parallel) when is_integer(ctx) and is_integer(parallel),
    do: ctx * parallel

  defp ctx_total(_, _), do: nil

  # --- provenance extraction ---

  # HF cache layout: .../models--ORG--REPO/snapshots/<sha>/[VARIANT/]file.gguf
  defp ref_from_path(path) do
    {repo, revision} = AiroAgent.HFCache.repo_and_revision(path)
    file = Path.basename(path)
    meta = AiroAgent.GGUF.metadata(path)

    %ModelRef{
      id: build_id(repo, file),
      repo: repo,
      revision: revision,
      quant: quant_from_filename(file),
      family: meta.family,
      size_bytes: file_size(path),
      ctx_max: meta.ctx_max,
      path: path,
      capabilities: caps_for(path),
      engine: :llama_cpp
    }
  end

  defp build_id(nil, file), do: Path.rootname(file)

  defp build_id(repo, file) do
    case quant_from_filename(file) do
      nil -> repo
      q -> "#{repo}:#{q}"
    end
  end

  defp quant_from_filename(file) do
    case Regex.run(~r/(UD-[A-Z0-9_]+|Q\d[A-Z0-9_]*|IQ\d[A-Z0-9_]*|BF16|F16)/, file) do
      [_, q] -> q
      _ -> nil
    end
  end

  defp mtp?(%ModelRef{repo: repo, path: path}),
    do: String.contains?(to_string(repo) <> path, "MTP")

  defp shard_continuation?(file),
    do: Regex.match?(~r/-0000[2-9]-of-\d+\.gguf$/, file)

  # Multimodal vision projector (CLIP/SigLIP encoder) — a companion to a base
  # model, never served on its own. Matched by `mmproj` ANYWHERE in the name, not
  # just as a prefix: unsloth ships `mmproj-F16.gguf`, ggml-org ships
  # `moondream2-mmproj-f16-20250414.gguf`. Case-insensitive for safety.
  defp projector?(file), do: String.contains?(String.downcase(Path.basename(file)), "mmproj")

  defp file_size(path) do
    case File.stat(path) do
      {:ok, %{size: s}} -> s
      _ -> nil
    end
  end

  defp default_roots do
    Application.get_env(:airo_agent, :model_roots, [
      Path.expand("~/.cache/huggingface/hub")
    ])
  end

  # Contract A: profile `ctx` is the PER-REQUEST window. llama-server's `-c` is
  # the TOTAL KV budget it splits across `--parallel` sequences, so to give each
  # request the full `ctx` we request `ctx × parallel`. nil `ctx` ⇒ omit `-c`
  # (let llama-server default). VRAM scales with the total, i.e. with parallel.
  defp total_ctx(%{ctx: ctx} = profile) when is_integer(ctx),
    do: ctx * (Map.get(profile, :parallel) || 1)

  defp total_ctx(_), do: nil

  defp flag(_k, nil), do: []
  defp flag(k, v), do: [k, to_string(v)]

  # Valueless switches (e.g. --jinja): emit the flag only when explicitly on.
  defp bool_flag(k, true), do: [k]
  defp bool_flag(_k, _), do: []
end
