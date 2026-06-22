defmodule AiroAgent.Notifier.Log do
  @moduledoc """
  Default `AiroAgent.Notifier`: logs each lifecycle event. A placeholder until the
  channel client (decision #3) lands, and a useful fallback for loopback dev.
  """

  @behaviour AiroAgent.Notifier

  require Logger
  alias AiroAgent.Fleet.Event

  @impl true
  def publish(%Event{type: type, model_id: id, reason: reason}) do
    detail = if reason, do: " (#{inspect(reason)})", else: ""
    Logger.info("fleet event: #{type} #{id}#{detail}")
    :ok
  end
end
