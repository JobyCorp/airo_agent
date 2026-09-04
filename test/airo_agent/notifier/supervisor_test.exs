defmodule AiroAgent.Notifier.SupervisorTest do
  # Locks the firewall contract: the channels run under a supervisor whose
  # restart budget is far wider than the application supervisor's (3/5s), so a
  # channel crash-loop is absorbed here and can never cascade into serving.
  # S26: one child per configured airo endpoint, controller and observers alike.
  use ExUnit.Case, async: false

  alias AiroAgent.Notifier

  setup do
    prev = Application.get_env(:airo_agent, :airo_endpoints)

    on_exit(fn ->
      if is_nil(prev),
        do: Application.delete_env(:airo_agent, :airo_endpoints),
        else: Application.put_env(:airo_agent, :airo_endpoints, prev)
    end)

    :ok
  end

  test "supervises one channel per endpoint with a budget wider than the app supervisor's" do
    Application.put_env(:airo_agent, :airo_endpoints, [
      %{uri: "wss://llm.local.joby.gg/agent", role: :controller},
      %{uri: "ws://jobybook.local.joby.gg:4004/agent", role: :observer}
    ])

    {:ok, {flags, children}} = Notifier.Supervisor.init(:ok)

    assert flags.strategy == :one_for_one
    # App supervisor tolerates 3 restarts / 5s; this must be much more forgiving
    # or a channel loop would still bubble up.
    assert flags.intensity >= 100
    assert flags.period >= 10

    assert [
             %{id: {Notifier.Channel, "wss://llm.local.joby.gg/agent"}, start: {_, _, [ctrl]}},
             %{
               id: {Notifier.Channel, "ws://jobybook.local.joby.gg:4004/agent"},
               start: {_, _, [obs]}
             }
           ] = children

    assert ctrl[:role] == :controller
    assert obs[:role] == :observer
  end

  test "no endpoints means no children" do
    Application.put_env(:airo_agent, :airo_endpoints, [])
    {:ok, {_flags, []}} = Notifier.Supervisor.init(:ok)
  end
end
