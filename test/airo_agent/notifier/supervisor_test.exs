defmodule AiroAgent.Notifier.SupervisorTest do
  # Locks the firewall contract: the channel runs under a supervisor whose
  # restart budget is far wider than the application supervisor's (3/5s), so a
  # channel crash-loop is absorbed here and can never cascade into serving.
  use ExUnit.Case, async: true

  alias AiroAgent.Notifier

  test "supervises the channel with a budget wider than the app supervisor's" do
    {:ok, {flags, children}} = Notifier.Supervisor.init(:ok)

    assert flags.strategy == :one_for_one
    # App supervisor tolerates 3 restarts / 5s; this must be much more forgiving
    # or a channel loop would still bubble up.
    assert flags.intensity >= 100
    assert flags.period >= 10

    assert [%{id: Notifier.Channel}] = children
  end
end
