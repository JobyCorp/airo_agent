defmodule AiroAgent do
  @moduledoc """
  Host-side model-serving **control plane**, consumed by Airo.

  `airo_agent` runs on each model-serving host. It starts and stops local
  inference engines (`llama-server` now; vLLM/TGI later), swaps which model each
  one serves, surfaces local-model provenance, and reports host GPU telemetry —
  pushing all of it to Airo over a channel so Airo never has to poll.

  It is **not** a server of models and **not** a provider. The engine is the
  server; Airo is the brain. This app is the mechanism in between.

  ## The one rule — three layers

  > Engine serves. Agent controls. Airo decides.

    * **Engine** (an Airo *Provider*) — OpenAI-compatible inference on a static
      port. Spawned by the agent, or run externally.
    * **airo_agent** — for engines it spawned: start/stop, load/unload/swap,
      provenance inventory, host telemetry, health push. Never on the data path.
    * **Airo** — routing, load/evict/VRAM policy, the model shelf.

  ## Key modules

    * `AiroAgent.Fleet` — canonical slot state; `load/3`, `unload/1`, `slots/0`.
    * `AiroAgent.Instance` / `AiroAgent.Instances` — one crash-isolated engine OS
      process per slot, supervised via `MuonTrap`.
    * `AiroAgent.Engine` + `AiroAgent.Engine.LlamaCpp` — the engine-neutral seam
      (inventory, launch spec, profile, capabilities).
    * `AiroAgent.Inventory` / `AiroAgent.ModelRef` — local catalog with HF-snapshot
      provenance (revision).
    * `AiroAgent.GPU` — `nvidia-smi` telemetry snapshot.
    * `AiroAgent.Api.Router` (+ `AiroAgent.Api.Spec`) — the control HTTP API and
      its published OpenAPI contract (`GET /openapi`).
    * `AiroAgent.Notifier.Channel` — the `slipstream` client that pushes state to
      Airo, isolated under `AiroAgent.Notifier.Supervisor` so it can never take
      serving down.

  See `DESIGN.md` for the full design (Model 2: one agent, N slot-providers).
  """
end
