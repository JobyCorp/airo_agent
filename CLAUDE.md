# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`airo_agent` is the **host-side control plane** for local model serving, consumed by Airo (a Phoenix AI-gateway). It runs on each GPU serving host and supervises inference *engines* (`llama-server` now; vLLM landing; TGI later), swaps which model each engine serves, surfaces local-model provenance (HF snapshot revision), and pushes host/GPU state to Airo over a channel so Airo never polls.

The cardinal rule, repeated throughout the code: **this app is never on the inference data path.** It is control plane only. Inference goes *directly* to a slot's `base_url`. If you find yourself proxying tokens, you're in the wrong layer.

> Engine serves. Agent controls. Airo decides.

`DESIGN.md` is the source of truth for the design ("Model 2: one agent, N slot-providers"); `DESIGN-vllm.md` covers the vLLM-specific design. `README.md` documents the control API and env vars.

## Commands

```sh
mix deps.get          # fetch deps
mix test              # run the suite (ExUnit)
mix test test/airo_agent/fleet_test.exs              # one file
mix test test/airo_agent/fleet_test.exs:42           # one test by line
mix format            # format (import_deps: [:slipstream])
iex -S mix            # boot the full supervision tree locally (loopback dev)
```

Deploy is over SSH to GPU hosts, one host at a time (see `bin/deploy.sh` header for the full usage and the multi-arch build story):

```sh
bin/deploy.sh                  # deploy to every deploy/hosts/*.env host
bin/deploy.sh gpu-01 sparky    # deploy to specific ssh aliases
bin/rollback.sh gpu-01         # instant symlink-flip rollback
```

## Architecture

The whole system is built around an **engine-neutral seam** so backends are swappable without touching supervision, the HTTP API, or readiness logic.

- **`AiroAgent.Engine`** (`engine.ex`) — the behaviour. Adapters (`Engine.LlamaCpp`, `Engine.Vllm`) implement `inventory/1` (scan host for local models + provenance) and `launch_spec/3` (build argv/env to serve a model — **pure, no side effects**). **Design rule: an Engine callback NEVER spawns or owns a process.** `@adapters` maps `:llama_cpp`/`:vllm` → module; `inventory_all/1` merges across the host's configured engines and skips unknown ones rather than crashing. Optional callbacks (`resolve_profile/2`, `cluster_info/2`, `honored_profile_keys/0`, `runtime_props/1`, `reap_orphans/0`) are dispatched through `Engine.exports?/3` — use it, not bare `function_exported?/3`, which answers `false` for a module that merely hasn't loaded yet (the orphan sweep runs at boot, before anything touches an adapter).

- **`AiroAgent.Fleet`** (`fleet.ex`) — the **lifecycle brain** and canonical in-memory state (a GenServer). Owns this host's *slots* (a slot = a static port holding at most one resident model). `load/3` swaps a model into a slot; `unload/1` frees it. It monitors each engine process and classifies its `:DOWN` as `:unloaded` (intended) vs `:failed`/`:down` (crash) using an intent flag set before teardown and demonitor-on-swap. Emits `Fleet.Event`s on every transition.

- **`AiroAgent.Instance`** (`instance.ex`) — owns ONE external engine OS process via `MuonTrap.Daemon`, so a native crash (CUDA OOM, segfault) is isolated to this supervised child and reaped with the BEAM — never the VM. `restart: :temporary` (a dead model is NOT silently respawned; Fleet decides). Polls readiness (`{:http_get, path}`), then calls `Fleet.mark_up/2` with engine runtime props.

- **`AiroAgent.Instances`** (`instances.ex`) — bare `DynamicSupervisor`, one child per running engine. No logic; Fleet drives it.

- **`AiroAgent.Notifier`** (`notifier.ex`) — seam for pushing `Fleet.Event`s to Airo. `Notifier.Log` (default, loopback dev) just logs; `Notifier.Channel` is a `slipstream` Phoenix-Channels *client* that connects to Airo's `/agent` socket and pushes state. Selected by config based on whether `AIRO_SOCKET_URL` is set. The channel runs under its own supervisor so a crash-loop can't take serving down.

- **`AiroAgent.Inventory`** (`inventory.ex`) — cached local model catalog with provenance; rescan via `POST /inventory/refresh`.

- **`AiroAgent.ModelRef`** (`model_ref.ex`) — identity + provenance of a local artifact. The `revision` field (HF snapshot commit sha) is the reason this app exists — it's what Ollama/LM Studio never tell Airo.

- **`AiroAgent.GPU`** (`gpu.ex`) — polled, read-only VRAM/util telemetry.

- **HTTP API** under `lib/airo_agent/api/` — Bandit + Plug, with an `open_api_spex` contract served at `GET /openapi`. Endpoints: `/health`, `/inventory`, `/slots`, `/gpu`, `POST /load|/unload|/inventory/refresh`. Management only.

Boot order (`application.ex`): orphan-container sweep (vLLM only) → `GPU` → `Inventory` → `Instances` → `Fleet` → notifier (conditional) → Bandit API.

### Two things that are subtle and easy to get wrong

1. **The ctx contract differs per engine.** llama.cpp's `-c` is the *total* KV budget split across `--parallel` sequences, so the agent sets `-c = ctx × parallel` (contract "A": `ctx` is the per-request window). vLLM's `--max-model-len` IS the per-request window directly (no ×parallel), and `parallel → --max-num-seqs`. A vLLM load with NO `ctx` is capped at `min(ctx_max, 32768)` — vLLM's own default is the model's full window, which OOMs small cards. See the moduledocs in `engine/llama_cpp.ex` and `engine/vllm.ex`.

2. **A changed `profile` is NOT idempotent.** `Fleet.load/3` short-circuits only when the same model is resident *with the same profile*; a new profile (e.g. different ctx) falls through to a full engine relaunch.

3. **`POST /load` accepts the union of every engine's profile keys.** A key the target engine doesn't read is dropped — deliberately, so one profile is portable across hosts. Sampling is the live asymmetry: llama.cpp maps `temperature`/`top_p`/the three penalties to argv, vLLM maps none. Adapters declare what they read via `honored_profile_keys/0`; `Instance` logs the difference at launch, and `AiroAgent.EngineTest` pins those lists against the router allowlist in **both** directions (a key an adapter reads but the router omits can never reach it — that was the `mmproj` bug).

## Configuration

All host-specific config is resolved at runtime in `config/runtime.exs` from env vars (so one release runs on any host). Key vars: `AIRO_AGENT_PORT` (4400), `AIRO_AGENT_SLOTS` (CSV of serving ports), `AIRO_AGENT_ADVERTISE_HOST` (LAN IP — a non-loopback value *exposes* the API + engines on `0.0.0.0`), `AIRO_AGENT_ENGINES` (CSV: `llama_cpp`/`vllm`), `AIRO_AGENT_MODEL_ROOT`, `AIRO_SOCKET_URL` (when set → channel push, else log-only), `AIRO_AGENT_TOKEN` (optional bearer auth, decoupled from exposure). Full table in `README.md`. Exposure is explicit and independent of the token — setting `ADVERTISE_HOST` to a LAN IP is what opens the bind to `0.0.0.0`.

## Testing conventions

The engine process is mocked out via `config :airo_agent, :instance_module, AiroAgent.Test.FakeInstance` (`test/support/fake_instance.ex`), which never spawns a real engine — it lets tests drive the Fleet state machine deterministically: `Process.exit(pid, reason)` simulates an engine crash, `FakeInstance.become_ready/1` simulates readiness passing. A `TestNotifier` (`test/support/test_notifier.ex`) captures emitted events. Engine adapters are tested as pure functions (argv/inventory) — no processes spawned.
