defmodule AiroAgent.GGUFTest do
  use ExUnit.Case, async: true

  alias AiroAgent.GGUF

  # --- minimal GGUF binary builders ---

  defp gstr(s), do: <<byte_size(s)::little-64, s::binary>>

  defp kv_string(key, val), do: gstr(key) <> <<8::little-32>> <> gstr(val)
  defp kv_u32(key, v), do: gstr(key) <> <<4::little-32, v::little-32>>
  defp kv_u64(key, v), do: gstr(key) <> <<10::little-32, v::little-64>>

  # array of strings (e.g. a tokenizer vocab) — what we must skip over
  defp kv_str_array(key, items) do
    gstr(key) <>
      <<9::little-32, 8::little-32, length(items)::little-64>> <>
      Enum.map_join(items, "", &gstr/1)
  end

  defp gguf(kvs),
    do: <<"GGUF", 3::little-32, 0::little-64, length(kvs)::little-64>> <> Enum.join(kvs, "")

  test "extracts family (general.architecture) and ctx_max (<arch>.context_length)" do
    data =
      gguf([
        kv_string("general.architecture", "qwen3"),
        kv_u32("qwen3.context_length", 8192)
      ])

    assert GGUF.from_binary(data) == %{family: "qwen3", ctx_max: 8192}
  end

  test "skips a (large) array between the fields" do
    data =
      gguf([
        kv_string("general.architecture", "llama"),
        kv_str_array("tokenizer.ggml.tokens", ~w(a bb ccc dddd)),
        kv_u64("llama.context_length", 131_072)
      ])

    assert GGUF.from_binary(data) == %{family: "llama", ctx_max: 131_072}
  end

  test "stops early once both fields are found (ignores trailing pairs)" do
    # Declares 3 pairs but only 2 are present; early-stop avoids reading past them.
    tail = kv_string("general.architecture", "qwen3") <> kv_u32("qwen3.context_length", 4096)
    data = <<"GGUF", 3::little-32, 0::little-64, 3::little-64>> <> tail

    assert GGUF.from_binary(data) == %{family: "qwen3", ctx_max: 4096}
  end

  test "missing fields yield nil, not a crash" do
    data = gguf([kv_string("general.name", "Some Model"), kv_u32("general.file_type", 15)])
    assert GGUF.from_binary(data) == %{family: nil, ctx_max: nil}
  end

  test "non-GGUF / truncated input is best-effort nil" do
    assert GGUF.from_binary("not a gguf file at all") == %{family: nil, ctx_max: nil}
    assert GGUF.from_binary(<<"GGUF", 3::little-32>>) == %{family: nil, ctx_max: nil}
    assert GGUF.metadata("/no/such/file.gguf") == %{family: nil, ctx_max: nil}
  end
end
