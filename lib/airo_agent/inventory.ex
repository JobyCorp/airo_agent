defmodule AiroAgent.Inventory do
  @moduledoc """
  Cached local model catalog. Holds the provenance Airo fills its shelf with.
  Rescan on demand (`POST`-triggered `refresh/0`) so newly pulled GGUFs appear.
  """

  use GenServer
  alias AiroAgent.{Engine, ModelRef}

  def start_link(_), do: GenServer.start_link(__MODULE__, :ok, name: __MODULE__)

  @spec list() :: [ModelRef.t()]
  def list, do: GenServer.call(__MODULE__, :list)

  @spec get(String.t()) :: {:ok, ModelRef.t()} | {:error, :not_found}
  def get(id), do: GenServer.call(__MODULE__, {:get, id})

  @spec refresh() :: [ModelRef.t()]
  def refresh, do: GenServer.call(__MODULE__, :refresh, 30_000)

  @impl true
  def init(:ok), do: {:ok, %{models: []}, {:continue, :scan}}

  @impl true
  def handle_continue(:scan, state), do: {:noreply, %{state | models: scan()}}

  @impl true
  def handle_call(:list, _from, state), do: {:reply, state.models, state}

  def handle_call({:get, id}, _from, state) do
    case Enum.find(state.models, &(&1.id == id)) do
      nil -> {:reply, {:error, :not_found}, state}
      ref -> {:reply, {:ok, ref}, state}
    end
  end

  def handle_call(:refresh, _from, state) do
    models = scan()
    {:reply, models, %{state | models: models}}
  end

  defp scan do
    case Engine.inventory_all() do
      {:ok, refs} -> refs
      {:error, _} -> []
    end
  end
end
