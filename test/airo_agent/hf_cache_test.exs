defmodule AiroAgent.HFCacheTest do
  use ExUnit.Case, async: true

  alias AiroAgent.HFCache

  describe "repo_and_revision/1" do
    test "decodes ORG/REPO and the snapshot sha from a cache path" do
      path =
        "/home/x/.cache/huggingface/hub/models--unsloth--Qwen3-GGUF/snapshots/abc123def/model-Q4.gguf"

      assert HFCache.repo_and_revision(path) == {"unsloth/Qwen3-GGUF", "abc123def"}
    end

    test "works for a safetensors snapshot dir (vLLM layout), not just a file" do
      path =
        "/cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

      assert HFCache.repo_and_revision(path) ==
               {"meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"}
    end

    test "nils both when the path isn't in the canonical cache layout" do
      assert HFCache.repo_and_revision("/models/loose/model.gguf") == {nil, nil}
    end

    test "nils the revision when there's no snapshots segment" do
      assert HFCache.repo_and_revision("/cache/models--org--repo/blobs/deadbeef") ==
               {"org/repo", nil}
    end
  end
end
