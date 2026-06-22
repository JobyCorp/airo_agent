defmodule AiroAgent.Notifier.Log do
  @moduledoc """
  Default `AiroAgent.Notifier`: logs each slot transition. A placeholder until a
  channel client is configured, and a useful fallback for loopback dev.
  """

  @behaviour AiroAgent.Notifier

  require Logger
  alias AiroAgent.Fleet.Event

  @impl true
  def publish(%Event{type: type, port: port, resident_model: model, reason: reason}) do
    detail = if reason, do: " (#{inspect(reason)})", else: ""
    Logger.info("slot :#{port} #{type} #{model || "(none)"}#{detail}")
    :ok
  end
end
