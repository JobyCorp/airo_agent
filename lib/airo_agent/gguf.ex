defmodule AiroAgent.GGUF do
  @moduledoc """
  Minimal, best-effort reader for the metadata header of a GGUF file.

  We only need two fields for the shelf — `general.architecture` (family) and
  `<arch>.context_length` (max context) — both of which standard writers place
  near the front, before the big tokenizer arrays. So we read a bounded prefix
  of the file, parse key/value pairs in order, and **stop as soon as both are
  found** (never touching the multi-MB tokenizer vocab). Anything unparseable or
  missing yields `nil` — provenance/inventory must never crash on a model file.

  Layout (GGUF v2/v3): magic `"GGUF"`, `version:u32`, `tensor_count:u64`,
  `kv_count:u64`, then `kv_count` pairs of `key:gguf_string` + typed value. All
  little-endian. A gguf_string is `len:u64` + `len` bytes.
  """

  # gguf_metadata_value_type enum
  @uint8 0
  @int8 1
  @uint16 2
  @int16 3
  @uint32 4
  @int32 5
  @float32 6
  @bool 7
  @string 8
  @array 9
  @uint64 10
  @int64 11
  @float64 12

  @default_read 2_000_000

  @type t :: %{family: String.t() | nil, ctx_max: pos_integer() | nil}

  @doc "Read `%{family, ctx_max}` from a GGUF file. Best-effort — never raises."
  @spec metadata(Path.t(), pos_integer()) :: t()
  def metadata(path, read_bytes \\ @default_read) do
    with {:ok, fd} <- :file.open(path, [:read, :binary, :raw]),
         {:ok, data} <- :file.pread(fd, 0, read_bytes) do
      _ = :file.close(fd)
      from_binary(data)
    else
      _ -> empty()
    end
  rescue
    _ -> empty()
  end

  @doc false
  # Parse already-read GGUF bytes (exposed for testing).
  @spec from_binary(binary()) :: t()
  def from_binary(data) when is_binary(data), do: parse(data)

  defp empty, do: %{family: nil, ctx_max: nil}

  defp parse(
         <<"GGUF", _version::little-32, _tensor_count::little-64, kv_count::little-64,
           rest::binary>>
       ),
       do: read_kvs(rest, kv_count, empty())

  defp parse(_), do: empty()

  # Stop on: no pairs left, both fields found, or buffer exhausted mid-value.
  defp read_kvs(_data, 0, acc), do: acc
  defp read_kvs(_data, _n, %{family: f, ctx_max: c} = acc) when f != nil and c != nil, do: acc

  defp read_kvs(data, n, acc) do
    case read_kv(data) do
      {:ok, key, value, rest} -> read_kvs(rest, n - 1, capture(acc, key, value))
      :halt -> acc
    end
  end

  defp capture(acc, "general.architecture", value) when is_binary(value),
    do: %{acc | family: value}

  defp capture(acc, key, value) when is_integer(value) and value > 0 do
    if String.ends_with?(key, ".context_length"), do: %{acc | ctx_max: value}, else: acc
  end

  defp capture(acc, _key, _value), do: acc

  defp read_kv(data) do
    with {:ok, key, rest} <- read_string(data),
         {:ok, value, rest} <- read_value(rest) do
      {:ok, key, value, rest}
    else
      _ -> :halt
    end
  end

  defp read_string(<<len::little-64, str::binary-size(len), rest::binary>>), do: {:ok, str, rest}
  defp read_string(_), do: :error

  # The value's type tag is a uint32; read it, then the typed payload.
  defp read_value(<<type::little-32, rest::binary>>), do: read_typed(type, rest)
  defp read_value(_), do: :error

  defp read_typed(@string, <<len::little-64, str::binary-size(len), rest::binary>>),
    do: {:ok, str, rest}

  defp read_typed(@uint8, <<v, rest::binary>>), do: {:ok, v, rest}
  defp read_typed(@int8, <<v::signed, rest::binary>>), do: {:ok, v, rest}
  defp read_typed(@bool, <<v, rest::binary>>), do: {:ok, v == 1, rest}
  defp read_typed(@uint16, <<v::little-16, rest::binary>>), do: {:ok, v, rest}
  defp read_typed(@int16, <<v::little-signed-16, rest::binary>>), do: {:ok, v, rest}
  defp read_typed(@uint32, <<v::little-32, rest::binary>>), do: {:ok, v, rest}
  defp read_typed(@int32, <<v::little-signed-32, rest::binary>>), do: {:ok, v, rest}
  defp read_typed(@float32, <<v::little-float-32, rest::binary>>), do: {:ok, v, rest}
  defp read_typed(@uint64, <<v::little-64, rest::binary>>), do: {:ok, v, rest}
  defp read_typed(@int64, <<v::little-signed-64, rest::binary>>), do: {:ok, v, rest}
  defp read_typed(@float64, <<v::little-float-64, rest::binary>>), do: {:ok, v, rest}

  defp read_typed(@array, <<elem_type::little-32, count::little-64, rest::binary>>),
    do: skip_array(rest, elem_type, count)

  defp read_typed(_, _), do: :error

  # Array elements carry no per-element type tag (the array declares it once).
  defp skip_array(rest, _elem_type, 0), do: {:ok, :array, rest}

  defp skip_array(data, elem_type, count) do
    case skip_elem(data, elem_type) do
      {:ok, rest} -> skip_array(rest, elem_type, count - 1)
      :error -> :error
    end
  end

  defp skip_elem(<<len::little-64, _::binary-size(len), rest::binary>>, @string), do: {:ok, rest}

  defp skip_elem(<<_::little-32, rest::binary>>, t) when t in [@uint32, @int32, @float32],
    do: {:ok, rest}

  defp skip_elem(<<_::little-64, rest::binary>>, t) when t in [@uint64, @int64, @float64],
    do: {:ok, rest}

  defp skip_elem(<<_::little-16, rest::binary>>, t) when t in [@uint16, @int16], do: {:ok, rest}
  defp skip_elem(<<_, rest::binary>>, t) when t in [@uint8, @int8, @bool], do: {:ok, rest}
  defp skip_elem(_, _), do: :error
end
