defmodule AiroAgent.Engine.VllmSlotWrapperTest do
  @moduledoc """
  Runs `priv/engine/vllm-slot` against stub `ssh`/runtime binaries.

  The cluster branch builds rank 1's command as an array and then shell-quotes it
  through `printf '%q '` before handing it to SSH, so a label added in the wrong
  place is either dropped or mangled — a failure mode no Elixir-level test can
  see. These stubs record the argv each rank was actually launched with.
  """
  use ExUnit.Case, async: true

  @wrapper Path.expand("../../../priv/engine/vllm-slot", __DIR__)

  setup do
    dir = System.tmp_dir!() |> Path.join("vllm-slot-test-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    log = Path.join(dir, "calls.log")

    # Stubs record their argv and exit 0. `ssh` also stands in for the worker's
    # model-dir precheck, which must succeed for the launch to proceed.
    for name <- ~w(fakert ssh) do
      path = Path.join(dir, name)
      File.write!(path, "#!/usr/bin/env bash\necho \"#{name} $*\" >> #{log}\nexit 0\n")
      File.chmod!(path, 0o755)
    end

    on_exit(fn -> File.rm_rf!(dir) end)
    %{dir: dir, log: log}
  end

  defp run(dir, env) do
    base = [
      {"PATH", dir <> ":" <> System.get_env("PATH")},
      {"AIRO_CONTAINER_RUNTIME", "fakert"},
      {"VLLM_IMAGE", "example/vllm:latest"},
      {"AIRO_AGENT_MODEL_ROOT", dir}
    ]

    System.cmd(@wrapper, ~w(serve /models/big --port 8081 --node-rank 0),
      env: base ++ env,
      stderr_to_stdout: true
    )
  end

  defp lines(log), do: log |> File.read!() |> String.split("\n", trim: true)

  # The wrapper makes several ssh calls (model-dir precheck, stale-container
  # sweep, then the launch) — match on the one that actually starts a container.
  defp rank_launch(log, "1") do
    Enum.find(lines(log), &(String.starts_with?(&1, "ssh ") and &1 =~ "run --rm"))
  end

  defp rank_launch(log, "0") do
    Enum.find(lines(log), &(String.starts_with?(&1, "fakert ") and &1 =~ "run --rm"))
  end

  describe "cluster mode" do
    @cluster_env [
      {"AIRO_VLLM_CLUSTER", "1"},
      {"AIRO_VLLM_WORKER_SSH", "jody@192.168.100.11"},
      {"AIRO_VLLM_MASTER_IP", "192.168.100.10"},
      {"AIRO_VLLM_NCCL_IF", "enP2p1s0f1np1"},
      {"AIRO_VLLM_CLUSTER_ID", "dep-7f3a1c2b"},
      {"AIRO_VLLM_NNODES", "2"},
      {"AIRO_VLLM_SERVED_MODEL", "org/big:fp8"}
    ]

    test "stamps cluster identity on both ranks", %{dir: dir, log: log} do
      {_out, 0} = run(dir, @cluster_env)

      for rank <- ~w(0 1) do
        launch = rank_launch(log, rank)

        assert launch =~ "airo.cluster.id=dep-7f3a1c2b"
        assert launch =~ "airo.cluster.size=2"
        assert launch =~ "airo.cluster.model=org/big:fp8"
        assert launch =~ "airo.slot.port=8081"
      end
    end

    test "each rank is labelled with its own rank number", %{dir: dir, log: log} do
      {_out, 0} = run(dir, @cluster_env)

      assert rank_launch(log, "0") =~ "airo.cluster.rank=0"
      refute rank_launch(log, "0") =~ "airo.cluster.rank=1"

      # Rank 1's labels have to survive `printf '%q '` quoting on the way to SSH.
      assert rank_launch(log, "1") =~ "airo.cluster.rank=1"
      refute rank_launch(log, "1") =~ "airo.cluster.rank=0"
    end

    test "rank 1 still gets --node-rank flipped and --headless appended", %{dir: dir, log: log} do
      {_out, 0} = run(dir, @cluster_env)
      remote = rank_launch(log, "1")

      assert remote =~ "--node-rank 1"
      assert remote =~ "--headless"
    end
  end

  describe "single-host mode" do
    test "adds no cluster labels", %{dir: dir, log: log} do
      {_out, 0} = run(dir, [])

      launch = rank_launch(log, "0")

      assert launch =~ "run --rm --name airo-slot-8081"
      refute launch =~ "airo.cluster"
    end
  end

  # A checkpoint can ship a newer tokenizer encoding than the runtime image carries;
  # the mount has to reach BOTH ranks or they encode prompts differently, which is
  # invisible until generations quietly diverge.
  describe "encoding override" do
    @target "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py"

    defp encoding_file(dir) do
      path = Path.join(dir, "encoding_dsv4.py")
      File.write!(path, "# stub\n")
      path
    end

    test "mounts read-only over the image copy", %{dir: dir, log: log} do
      src = encoding_file(dir)
      {_out, 0} = run(dir, [{"AIRO_VLLM_ENCODING_FILE", src}])

      assert rank_launch(log, "0") =~ "-v #{src}:#{@target}:ro"
    end

    test "reaches both ranks in cluster mode", %{dir: dir, log: log} do
      src = encoding_file(dir)

      {_out, 0} =
        run(dir, [
          {"AIRO_VLLM_CLUSTER", "1"},
          {"AIRO_VLLM_WORKER_SSH", "jody@192.168.100.11"},
          {"AIRO_VLLM_MASTER_IP", "192.168.100.10"},
          {"AIRO_VLLM_NCCL_IF", "enP2p1s0f1np1"},
          {"AIRO_VLLM_ENCODING_FILE", src}
        ])

      # Rank 1's mount has to survive `printf '%q '` on the way to SSH.
      for rank <- ~w(0 1), do: assert(rank_launch(log, rank) =~ "#{src}:#{@target}:ro")
    end

    test "honors a custom target path", %{dir: dir, log: log} do
      src = encoding_file(dir)

      {_out, 0} =
        run(dir, [{"AIRO_VLLM_ENCODING_FILE", src}, {"AIRO_VLLM_ENCODING_TARGET", "/opt/enc.py"}])

      assert rank_launch(log, "0") =~ "-v #{src}:/opt/enc.py:ro"
    end

    test "refuses to launch when the source is missing", %{dir: dir, log: log} do
      {out, 78} = run(dir, [{"AIRO_VLLM_ENCODING_FILE", Path.join(dir, "nope.py")}])

      assert out =~ "not found"
      # Bails before touching the runtime at all, so no stub ever ran — better a
      # dead slot than one silently encoding prompts the wrong way.
      refute File.exists?(log)
    end

    test "'none' sentinel and unset both mean no mount", %{dir: dir, log: log} do
      {_out, 0} = run(dir, [{"AIRO_VLLM_ENCODING_FILE", "none"}])
      {_out, 0} = run(dir, [])

      for line <- lines(log), do: refute(line =~ @target)
    end
  end
end
