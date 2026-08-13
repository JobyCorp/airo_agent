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
    #
    # `inspect` is the exception: it must FAIL, modelling a container that is
    # gone. The wrapper polls it to confirm the runtime released the slot name
    # before relaunching, so a stub that reports the name as still present would
    # spin the poll for its full 10s timeout and then refuse to launch.
    for name <- ~w(fakert ssh) do
      path = Path.join(dir, name)

      File.write!(path, """
      #!/usr/bin/env bash
      echo "#{name} $*" >> #{log}
      [ "${1:-}" = inspect ] && exit 1
      exit 0
      """)

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

  @cluster_env [
    {"AIRO_VLLM_CLUSTER", "1"},
    {"AIRO_VLLM_WORKER_SSH", "jody@192.168.100.11"},
    {"AIRO_VLLM_MASTER_IP", "192.168.100.10"},
    {"AIRO_VLLM_NCCL_IF", "enP2p1s0f1np1"},
    {"AIRO_VLLM_CLUSTER_ID", "dep-7f3a1c2b"},
    {"AIRO_VLLM_NNODES", "2"},
    {"AIRO_VLLM_SERVED_MODEL", "org/big:fp8"}
  ]

  describe "slot name release" do
    test "waits for the runtime to free the name before launching", %{dir: dir, log: log} do
      {_out, 0} = run(dir, [])

      calls = lines(log)
      inspect_at = Enum.find_index(calls, &(&1 == "fakert inspect airo-slot-8081"))
      launch_at = Enum.find_index(calls, &String.starts_with?(&1, "fakert run --rm"))

      # `rm -f` returns before the name is actually free, so the wrapper must
      # confirm via `inspect` — and must do it BEFORE `run`, or the relaunch
      # races the previous container's teardown and dies with exit 125.
      assert inspect_at, "wrapper never polled `inspect` to confirm the name was released"
      assert launch_at, "wrapper never launched the container"
      assert inspect_at < launch_at, "polled for the name only AFTER launching"
    end

    test "worker rank polls the name on the worker too", %{dir: dir, log: log} do
      {_out, 0} = run(dir, @cluster_env)

      assert Enum.any?(
               lines(log),
               &(String.starts_with?(&1, "ssh ") and &1 =~ "inspect airo-slot-8081")
             ),
             "rank 1 never waited for the worker to release the slot name"
    end
  end

  describe "cluster mode" do
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

  # `encoding_file` mounts one well-known path; an image can need several patched
  # files at once (anemll 0.1.1: the 0731 encoding AND the nvfp4_ds_mla dispatch
  # fix). Same both-ranks / fail-loud contract, because a partial overlay leaves
  # the ranks running different code — which only shows up as divergent output.
  describe "file overlays" do
    defp overlay_src(dir, name) do
      path = Path.join(dir, name)
      File.write!(path, "# stub\n")
      path
    end

    test "mounts every entry read-only", %{dir: dir, log: log} do
      a = overlay_src(dir, "flashmla_sparse.py")
      b = overlay_src(dir, "scheduler.py")

      {_out, 0} =
        run(dir, [{"AIRO_VLLM_OVERLAY_FILES", "#{a}:/opt/a.py\n#{b}:/opt/b.py"}])

      launch = rank_launch(log, "0")
      assert launch =~ "-v #{a}:/opt/a.py:ro"
      assert launch =~ "-v #{b}:/opt/b.py:ro"
    end

    test "reaches both ranks in cluster mode", %{dir: dir, log: log} do
      src = overlay_src(dir, "flashmla_sparse.py")

      {_out, 0} =
        run(dir, [
          {"AIRO_VLLM_CLUSTER", "1"},
          {"AIRO_VLLM_WORKER_SSH", "jody@192.168.100.11"},
          {"AIRO_VLLM_MASTER_IP", "192.168.100.10"},
          {"AIRO_VLLM_NCCL_IF", "enP2p1s0f1np1"},
          {"AIRO_VLLM_OVERLAY_FILES", "#{src}:/opt/a.py"}
        ])

      # Rank 1's mount has to survive `printf '%q '` on the way to SSH.
      for rank <- ~w(0 1), do: assert(rank_launch(log, rank) =~ "#{src}:/opt/a.py:ro")
    end

    test "coexists with an encoding override", %{dir: dir, log: log} do
      enc = encoding_file(dir)
      src = overlay_src(dir, "flashmla_sparse.py")

      {_out, 0} =
        run(dir, [
          {"AIRO_VLLM_ENCODING_FILE", enc},
          {"AIRO_VLLM_OVERLAY_FILES", "#{src}:/opt/a.py"}
        ])

      launch = rank_launch(log, "0")
      assert launch =~ "-v #{enc}:#{@target}:ro"
      assert launch =~ "-v #{src}:/opt/a.py:ro"
    end

    test "refuses to launch when any source is missing", %{dir: dir, log: log} do
      present = overlay_src(dir, "present.py")
      missing = Path.join(dir, "nope.py")

      {out, 78} =
        run(dir, [{"AIRO_VLLM_OVERLAY_FILES", "#{present}:/opt/a.py\n#{missing}:/opt/b.py"}])

      assert out =~ "not found"
      # One good entry must not buy a partial mount — bail before the runtime runs.
      refute File.exists?(log)
    end

    test "refuses a malformed entry", %{dir: dir, log: log} do
      {out, 78} = run(dir, [{"AIRO_VLLM_OVERLAY_FILES", "/tmp/no-target-here"}])

      assert out =~ "not host:target"
      refute File.exists?(log)
    end

    test "'none' sentinel and unset both mean no mount", %{dir: dir, log: log} do
      {_out, 0} = run(dir, [{"AIRO_VLLM_OVERLAY_FILES", "none"}])
      {_out, 0} = run(dir, [])

      for line <- lines(log), do: refute(line =~ "/opt/a.py")
    end
  end
end
