defmodule AiroAgent.Instances do
  @moduledoc """
  Bare `DynamicSupervisor` that runs one external engine process per running
  model. The lifecycle brain — load/unload, monitoring, change detection, and
  event emission — lives in `AiroAgent.Fleet`; this module only spawns and reaps
  children under `restart: :temporary`.
  """

  use DynamicSupervisor

  def start_link(_), do: DynamicSupervisor.start_link(__MODULE__, :ok, name: __MODULE__)

  @impl true
  def init(:ok), do: DynamicSupervisor.init(strategy: :one_for_one)
end
