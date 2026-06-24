defmodule AiroAgent.GPU do
  @moduledoc """
  Read-only GPU telemetry (VRAM/util/power) via `nvidia-smi`, polled and cached.
  Airo uses this to make load/evict decisions against a VRAM budget.

  Unified-memory GPUs (NVIDIA GB10 / Grace-Blackwell, as on a DGX Spark) report
  **no FB memory via NVML** — `nvidia-smi` returns `N/A` for memory.{total,used}.
  There the model's KV cache lives in the shared system pool, so we fall back to
  `/proc/meminfo` for the memory budget and tag the snapshot `mem_source:
  :unified` (vs `:nvml`). Util/power still come from `nvidia-smi` either way.

  macOS / Apple Silicon (Metal, as on a Mac mini) has no `nvidia-smi` at all and
  also runs on unified memory. There we read the pool from `sysctl hw.memsize`
  (total) + `vm_stat` (in-use pages) and tag `mem_source: :unified`; util/power
  aren't exposed without private frameworks, so they're `nil`.
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

  # macOS has no nvidia-smi; route it to the Metal/unified-memory path. Everything
  # else (Linux, incl. the GB10 unified host) goes through nvidia-smi.
  defp poll do
    case :os.type() do
      {:unix, :darwin} -> poll_darwin()
      _ -> poll_nvidia()
    end
  end

  defp poll_nvidia do
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

  # Apple Silicon: report the unified pool (sysctl total + vm_stat used). No
  # util/power without private frameworks, so leave them nil.
  defp poll_darwin do
    with {total_out, 0} <- System.cmd("sysctl", ["-n", "hw.memsize"]),
         {vmstat_out, 0} <- System.cmd("vm_stat", []),
         {total_mb, used_mb} <- parse_darwin_mem(total_out, vmstat_out) do
      %{
        available: true,
        vram_total_mb: total_mb,
        vram_used_mb: used_mb,
        util_pct: nil,
        power_draw_w: nil,
        power_limit_w: nil,
        mem_source: :unified
      }
    else
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

  @doc false
  # Parse `sysctl -n hw.memsize` (bytes) + `vm_stat` → `{total_mb, used_mb}`,
  # where used ≈ (active + wired + compressor) pages × page size (Activity
  # Monitor's "Memory Used"). The free/inactive/speculative pages are reclaimable
  # and counted as available.
  def parse_darwin_mem(total_out, vmstat_out) do
    pages = parse_vmstat(vmstat_out)

    with {total_bytes, _} <- Integer.parse(String.trim(total_out)),
         page_size when is_integer(page_size) <- pages[:page_size] do
      used_pages = (pages[:active] || 0) + (pages[:wired] || 0) + (pages[:compressor] || 0)
      {total_bytes / 1_048_576, used_pages * page_size / 1_048_576}
    else
      _ -> :error
    end
  end

  # vm_stat → %{page_size, active, wired, compressor}. Header carries the page
  # size ("page size of 16384 bytes"); each count line is "Label: <n>." (trailing
  # dot). Missing fields come back nil and are treated as 0 by the caller.
  defp parse_vmstat(out) do
    page_size =
      case Regex.run(~r/page size of (\d+) bytes/, out) do
        [_, n] -> String.to_integer(n)
        _ -> nil
      end

    counts =
      for line <- String.split(out, "\n", trim: true),
          [key, val] <- [String.split(line, ":", parts: 2)],
          {n, _} <- [val |> String.replace(".", "") |> String.trim() |> Integer.parse()],
          into: %{},
          do: {String.trim(key), n}

    %{
      page_size: page_size,
      active: counts["Pages active"],
      wired: counts["Pages wired down"],
      compressor: counts["Pages occupied by compressor"]
    }
  end

  defp to_num(s) do
    case Float.parse(s) do
      {f, _} -> f
      :error -> nil
    end
  end
end
