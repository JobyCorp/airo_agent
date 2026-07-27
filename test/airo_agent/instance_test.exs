defmodule AiroAgent.InstanceTest do
  @moduledoc """
  The real `AiroAgent.Instance` (not `Test.FakeInstance`), covering the seam that
  reports a profile key an engine will not read.

  `POST /load` atomizes against the union of every engine's keys, so a profile
  written for llama.cpp loads happily on a vLLM host with its sampling knobs
  doing nothing at all. That portability is deliberate; the silence was not.
  """
  use ExUnit.Case, async: false

  import ExUnit.CaptureLog

  alias AiroAgent.{Instance, ModelRef}

  setup do
    prev_bin = Application.get_env(:airo_agent, :engine_bin)

    # A harmless stand-in for the engine binary: it rejects the argv and exits,
    # which is fine — the warning is emitted in init/1, before the spawn.
    Application.put_env(:airo_agent, :engine_bin, %{llama_cpp: "/bin/cat", vllm: "/bin/cat"})
    # Instance start_link/1 links to us; a fast engine exit must not fail the test.
    Process.flag(:trap_exit, true)

    on_exit(fn -> Application.put_env(:airo_agent, :engine_bin, prev_bin) end)
    :ok
  end

  defp model(engine, path),
    do: %ModelRef{id: "m-#{engine}", path: path, engine: engine, revision: "rev"}

  defp start(model, profile) do
    capture_log(fn ->
      case Instance.start_link(%{model: model, profile: profile, port: 8099}) do
        {:ok, pid} -> Process.exit(pid, :kill)
        _ -> :ok
      end

      # Let the log flush before capture_log returns.
      Process.sleep(20)
    end)
  end

  test "warns about a profile key the engine does not read" do
    dir = Path.join(System.tmp_dir!(), "airo_inst_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)

    # temperature is a llama.cpp knob; the vLLM adapter maps no sampling keys.
    log = start(model(:vllm, dir), %{ctx: 4096, temperature: 0.7, top_p: 0.9})

    assert log =~ "ignoring profile key(s)"
    assert log =~ "temperature"
    assert log =~ "top_p"
    # Keys the engine DOES read are not named.
    refute log =~ "ctx,"
  end

  test "says nothing when every key is honoured" do
    log = start(model(:llama_cpp, "/tmp/airo-nonexistent.gguf"), %{ctx: 4096, temperature: 0.7})

    refute log =~ "ignoring profile key"
  end

  test "says nothing for a bare profile" do
    log = start(model(:llama_cpp, "/tmp/airo-nonexistent.gguf"), %{})

    refute log =~ "ignoring profile key"
  end
end
