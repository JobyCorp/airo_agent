defmodule AiroAgent.GPU do
  @moduledoc """
  Read-only GPU telemetry (VRAM/util/power) via `nvidia-smi`, polled and cached.
  Airo uses this to make load/evict decisions against a VRAM budget.
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
      {out, 0} -> parse(out)
      _ -> %{available: false}
    end
  rescue
    _ -> %{available: false}
  end

  defp parse(out) do
    case out |> String.trim() |> String.split(",") |> Enum.map(&String.trim/1) do
      [used, total, util, draw, limit] ->
        %{
          available: true,
          vram_used_mb: to_num(used),
          vram_total_mb: to_num(total),
          util_pct: to_num(util),
          power_draw_w: to_num(draw),
          power_limit_w: to_num(limit)
        }

      _ ->
        %{available: false}
    end
  end

  defp to_num(s) do
    case Float.parse(s) do
      {f, _} -> f
      :error -> nil
    end
  end
end
