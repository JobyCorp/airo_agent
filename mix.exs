defmodule AiroAgent.MixProject do
  use Mix.Project

  def project do
    [
      app: :airo_agent,
      version: "0.1.0",
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      description:
        "Host-side control surface for local model serving. Supervises an inference engine " <>
          "(llama.cpp now, vLLM/TGI later) and exposes inventory + lifecycle to Airo. " <>
          "Control plane only — never on the inference data path.",
      deps: deps(),
      releases: releases()
    ]
  end

  # Prod release: a self-contained tarball with bundled ERTS, so a serving host
  # needs no Erlang/Elixir installed. Built inside a container matching the
  # host's glibc (see bin/deploy.sh / DEPLOY.md); shipped over SSH and run as a
  # systemd system unit alongside the engines.
  defp releases do
    [
      airo_agent: [
        include_executables_for: [:unix],
        # Standalone control plane — no clustering, so skip distribution/epmd.
        # (The systemd unit also sets RELEASE_DISTRIBUTION=none.)
        applications: [runtime_tools: :permanent]
      ]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  def application do
    [
      extra_applications: [:logger, :runtime_tools],
      mod: {AiroAgent.Application, []}
    ]
  end

  defp deps do
    [
      # HTTP control API (mirrors airo's Bandit+Plug stack).
      {:bandit, "~> 1.5"},
      {:plug, "~> 1.16"},
      # Published contract (airo already uses open_api_spex — same dialect).
      {:open_api_spex, "~> 3.21"},
      # Outbound: scrape engine /metrics, read GGUF metadata. NOT for inference.
      {:req, "~> 0.5"},
      {:jason, "~> 1.4"},
      # Robust supervision of the external engine OS process: guarantees the
      # child dies with the BEAM and is reaped on crash (better than a raw Port).
      {:muontrap, "~> 1.5"},
      # Phoenix Channels CLIENT: the agent connects to airo's /agent socket and
      # pushes host state (decision #3). Airo is the server; we are the client.
      {:slipstream, "~> 1.1"},
      {:telemetry, "~> 1.0"},
      {:telemetry_metrics, "~> 1.0"},
      {:telemetry_poller, "~> 1.0"}
    ]
  end
end
