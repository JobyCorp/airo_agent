defmodule AiroAgent.GPU do
  @moduledoc """
  Read-only GPU telemetry (VRAM/util/power) via `nvidia-smi`, polled and cached.
  Airo uses this to make load/evict decisions against a VRAM budget.

  Unified-memory GPUs (NVIDIA GB10 / Grace-Blackwell, as on a DGX Spark) report
  **no FB memory via NVML** — `nvidia-smi` returns `N/A` for memory.{total,used}.
  There the model's KV cache lives in the shared system pool, so we fall back to
  `/proc/meminfo` for the memory budget and tag the snapshot `mem_source:
  :unified` (vs `:nvml`). Util/power still come from `nvidia-smi` either way.
  """

  use GenServer

  @poll_ms 5_000

  def start_link(_), do: GenServer.start_link(__MODULE__, :ok, name: __MODULE__)

  @spec snapshot() :: map()
  def snapshot, do: GenServer.call(__MODULE__, :snapshot)

  @impl true
  def init(:ok) do
    schedule()
    {:ok, poll()}
  end

  @impl true
  def handle_call(:snapshot, _from, state), do: {:reply, state, state}

  @impl true
  def handle_info(:poll, _state) do
    schedule()
    {:noreply, poll()}
  end

  defp schedule, do: Process.send_after(self(), :poll, @poll_ms)

  defp poll do
    query =
      "memory.used,memory.total,utilization.gpu,power.draw,power.limit"

    case System.cmd("nvidia-smi", ["--query-gpu=#{query}", "--format=csv,noheader,nounits"],
           stderr_to_stdout: true
         ) do
      {out, 0} -> out |> parse() |> with_memory_fallback()
      _ -> %{available: false}
    end
  rescue
    _ -> %{available: false}
  end

  @doc false
  def parse(out) do
    case out |> String.trim() |> String.split(",") |> Enum.map(&String.trim/1) do
      [used, total, util, draw, limit] ->
        %{
          available: true,
          vram_used_mb: to_num(used),
          vram_total_mb: to_num(total),
          util_pct: to_num(util),
          power_draw_w: to_num(draw),
          power_limit_w: to_num(limit),
          mem_source: :nvml
        }

      _ ->
        %{available: false}
    end
  end

  # NVML gave no memory total (unified-memory GPU) → use the system pool instead.
  defp with_memory_fallback(%{available: true, vram_total_mb: nil} = snap) do
    case File.read("/proc/meminfo") do
      {:ok, contents} ->
        case parse_meminfo(contents) do
          {total_mb, used_mb} ->
            %{snap | vram_total_mb: total_mb, vram_used_mb: used_mb, mem_source: :unified}

          :error ->
            snap
        end

      _ ->
        snap
    end
  end

  defp with_memory_fallback(snap), do: snap

  @doc false
  # Parse `/proc/meminfo` → `{total_mb, used_mb}` where used = MemTotal - MemAvailable.
  def parse_meminfo(contents) do
    fields =
      for line <- String.split(contents, "\n", trim: true),
          [key, val | _] <- [String.split(line, ~r/:\s*/, parts: 2)],
          into: %{} do
        {key, val |> String.replace(" kB", "") |> String.trim()}
      end

    with {total_kb, _} <- Float.parse(Map.get(fields, "MemTotal", "")),
         {avail_kb, _} <- Float.parse(Map.get(fields, "MemAvailable", "")) do
      {total_kb / 1024, (total_kb - avail_kb) / 1024}
    else
      _ -> :error
    end
  end

  defp to_num(s) do
    case Float.parse(s) do
      {f, _} -> f
      :error -> nil
    end
  end
end
