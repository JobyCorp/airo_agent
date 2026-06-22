defmodule AiroAgent.Instances do
  @moduledoc """
  Supervises one external engine process per *running* model. Exposes the
  mechanism (load/unload/running); the *policy* (when to load, what to evict
  under a VRAM budget) lives in Airo, by design.
  """

  use DynamicSupervisor
  alias AiroAgent.{Instance, Inventory, InstanceInfo}

  def start_link(_), do: DynamicSupervisor.start_link(__MODULE__, :ok, name: __MODULE__)

  @impl true
  def init(:ok), do: DynamicSupervisor.init(strategy: :one_for_one)

  @spec load(String.t(), map()) :: {:ok, InstanceInfo.t()} | {:error, term()}
  def load(model_id, profile \\ %{}) do
    with {:ok, model} <- Inventory.get(model_id),
         nil <- find_pid(model_id),
         port <- free_port(),
         {:ok, pid} <-
           DynamicSupervisor.start_child(
             __MODULE__,
             {Instance, %{model: model, profile: profile, port: port}}
           ) do
      {:ok, Instance.info(pid)}
    else
      pid when is_pid(pid) -> {:ok, Instance.info(pid)}
      {:error, _} = err -> err
    end
  end

  @spec unload(String.t()) :: :ok | {:error, :not_running}
  def unload(model_id) do
    case find_pid(model_id) do
      nil -> {:error, :not_running}
      pid -> DynamicSupervisor.terminate_child(__MODULE__, pid)
    end
  end

  @spec running() :: [InstanceInfo.t()]
  def running do
    for {_, pid, _, _} <- DynamicSupervisor.which_children(__MODULE__), is_pid(pid) do
      Instance.info(pid)
    end
  end

  defp find_pid(model_id) do
    Enum.find_value(running_pids(), fn pid ->
      if Instance.info(pid).model_id == model_id, do: pid
    end)
  end

  defp running_pids,
    do: for({_, pid, _, _} <- DynamicSupervisor.which_children(__MODULE__), is_pid(pid), do: pid)

  # Let the OS hand us a free port, then close and reuse it for the engine.
  defp free_port do
    {:ok, sock} = :gen_tcp.listen(0, [])
    {:ok, port} = :inet.port(sock)
    :ok = :gen_tcp.close(sock)
    port
  end
end
