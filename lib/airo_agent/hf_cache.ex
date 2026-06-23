defmodule AiroAgent.HFCache do
  @moduledoc """
  Helpers for the Hugging Face hub cache layout, shared by engine adapters.

  The cache stores each model under
  `<root>/models--ORG--REPO/snapshots/<sha>/<files>`, where `<sha>` is the HF
  snapshot commit — the `revision` provenance that is the whole reason this app
  exists. Both the llama.cpp (GGUF) and vLLM (safetensors) adapters resolve
  provenance from this same layout, so the parsing lives here once.
  """

  @doc """
  Extract `{repo, revision}` from any path inside the HF hub cache.

  `repo` is the `ORG/REPO` decoded from the `models--ORG--REPO` dir; `revision`
  is the `<sha>` snapshot segment. Either is `nil` when the path isn't in the
  canonical cache layout (e.g. a loose file), so callers degrade gracefully.
  """
  @spec repo_and_revision(Path.t()) :: {String.t() | nil, String.t() | nil}
  def repo_and_revision(path) do
    parts = Path.split(path)

    repo =
      case Enum.find(parts, &String.starts_with?(&1, "models--")) do
        nil -> nil
        dir -> dir |> String.replace_prefix("models--", "") |> String.replace("--", "/")
      end

    revision =
      case Enum.drop_while(parts, &(&1 != "snapshots")) do
        ["snapshots", sha | _] -> sha
        _ -> nil
      end

    {repo, revision}
  end
end
