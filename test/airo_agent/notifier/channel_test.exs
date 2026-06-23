defmodule AiroAgent.Notifier.ChannelTest do
  # Locks the agent↔airo wire contract: the shapes of the `register` and `slot`
  # messages (and the heartbeat re-register) the agent pushes, which airo parses.
  use Slipstream.SocketTest

  alias AiroAgent.Fleet.Event
  alias AiroAgent.Notifier.Channel

  @topic "agent:jobycorp"

  setup do
    prev =
      Enum.map(
        [:host_id, :advertise_host, :api_port],
        &{&1, Application.get_env(:airo_agent, &1)}
      )

    Application.put_env(:airo_agent, :host_id, "jobycorp")
    Application.put_env(:airo_agent, :advertise_host, "192.168.1.5")
    Application.put_env(:airo_agent, :api_port, 4400)

    on_exit(fn ->
      for {k, v} <- prev do
        if is_nil(v),
          do: Application.delete_env(:airo_agent, k),
          else: Application.put_env(:airo_agent, k, v)
      end
    end)

    client =
      start_supervised!({Channel, [uri: "ws://airo.test/agent/websocket", test_mode?: true]})

    %{client: client}
  end

  test "joins agent:<host_id> and pushes a register with agent identity + slots", %{
    client: client
  } do
    connect_and_assert_join(client, @topic, _params, :ok)

    assert_push(@topic, "register", payload)

    assert %{agent: agent, slots: slots} = payload
    # Agent identity airo upserts into the Agent record.
    assert agent.control_url == "http://192.168.1.5:4400"
    assert Map.has_key?(agent, :version)
    assert Map.has_key?(agent, :gpu)
    # Slots are the host's serving endpoints; each carries the fields airo needs.
    assert is_list(slots)

    for slot <- slots do
      assert Map.keys(slot) |> Enum.sort() ==
               ~w(base_url ctx engine_build parallel port resident_model revision started_at status)a
    end
  end

  test "pushes a slot transition with the contract shape", %{client: client} do
    connect_and_assert_join(client, @topic, _params, :ok)
    assert_push(@topic, "register", _join_register)

    Channel.publish(%Event{
      type: :up,
      port: 8081,
      resident_model: "org/repo:Q4",
      revision: "abc123",
      ctx: 65_536,
      parallel: 4,
      engine_build: "b1-9633186",
      reason: nil,
      at: ~U[2026-06-22 00:00:00Z]
    })

    assert_push(@topic, "slot", payload)

    assert payload == %{
             port: 8081,
             resident_model: "org/repo:Q4",
             revision: "abc123",
             ctx: 65_536,
             parallel: 4,
             engine_build: "b1-9633186",
             status: :up,
             reason: nil,
             at: ~U[2026-06-22 00:00:00Z]
           }
  end

  test "serializes a non-binary down reason to a string", %{client: client} do
    connect_and_assert_join(client, @topic, _params, :ok)
    assert_push(@topic, "register", _)

    Channel.publish(%Event{
      type: :down,
      port: 8081,
      resident_model: "org/repo:Q4",
      revision: "abc123",
      reason: {:shutdown, :killed},
      at: ~U[2026-06-22 00:00:00Z]
    })

    assert_push(@topic, "slot", %{status: :down, reason: reason})
    assert reason == inspect({:shutdown, :killed})
  end

  test "the heartbeat re-pushes register", %{client: client} do
    connect_and_assert_join(client, @topic, _params, :ok)
    assert_push(@topic, "register", _)

    send(client, :heartbeat)

    assert_push(@topic, "register", %{agent: _, slots: _})
  end

  test "a disconnect (Airo down) does not crash the client", %{client: client} do
    connect_and_assert_join(client, @topic, _params, :ok)
    assert_push(@topic, "register", _)

    ref = Process.monitor(client)
    disconnect(client, :closed)

    refute_receive {:DOWN, ^ref, :process, _, _}, 300
    assert Process.alive?(client)
  end
end
