defmodule AiroAgent.Test.Notifier do
  @moduledoc false
  # Forwards each Fleet event to the pid in `:airo_agent, :test_notifier_pid`, so a
  # test can `assert_receive {:event, %Event{...}}`.

  @behaviour AiroAgent.Notifier

  @impl true
  def publish(event) do
    case Application.get_env(:airo_agent, :test_notifier_pid) do
      pid when is_pid(pid) -> send(pid, {:event, event})
      _ -> :ok
    end

    :ok
  end
end
