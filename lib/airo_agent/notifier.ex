defmodule AiroAgent.Notifier do
  @moduledoc """
  Seam for pushing `AiroAgent.Fleet.Event`s outward to Airo.

  The default implementation just logs (`AiroAgent.Notifier.Log`); decision #3
  swaps in a `slipstream` channel client (push + presence + snapshot-on-reconnect).
  Select the implementation with `config :airo_agent, :notifier, MyImpl`.
  """

  alias AiroAgent.Fleet.Event

  @callback publish(Event.t()) :: :ok

  @spec publish(Event.t()) :: :ok
  def publish(%Event{} = event), do: impl().publish(event)

  defp impl, do: Application.get_env(:airo_agent, :notifier, AiroAgent.Notifier.Log)
end
