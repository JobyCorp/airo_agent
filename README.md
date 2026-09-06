# airo_agent

Host-side model-serving **control plane**, consumed by Airo.

> **Engine serves. Agent controls. Airo decides.**

## What

`airo_agent` is a small OTP release that runs on **each GPU serving host**
(x86 boxes and arm64 DGX Sparks alike). It starts, supervises and stops the
local inference engines — `llama-server` and vLLM today — swaps which model each
one serves, scans the local Hugging Face cache for provenance, reads GPU
telemetry, and **pushes** all of that to Airo over a Phoenix channel. Airo
commands it back over a small HTTP control API.

It is **not** a server of models and **not** a provider. Each engine it spawns
is a real OpenAI-compatible endpoint on a fixed port — a *slot* — and Airo routes
inference **straight to that port**. The agent is never on the inference path.

| Layer | Owns |
|---|---|
| **Engine** (an Airo Provider) | OpenAI-compatible inference on a static port. Spawned by the agent, or run externally. |
| **airo_agent** | for engines it spawned: load/unload/swap, provenance inventory, host telemetry, health push. Never on the inference path. |
| **Airo** | routing, load/evict/VRAM policy, the model shelf. |

## Why

Airo can talk to any OpenAI-compatible engine directly, so why an agent?

- **Someone has to own the GPU.** Which model is resident, how much VRAM is
  left, whether a swap will fit — these are *host* facts, and the engines
  themselves don't know or report them. One agent per host owns that view and
  pushes it, so Airo never polls and never guesses.
- **Provenance the engines can't give.** Ollama and LM Studio serve a model
  by a friendly name; they don't tell you *which* artifact. The agent resolves
  the Hugging Face snapshot commit (`revision`) for every local model, so Airo's
  model shelf and usage attribution are keyed to the exact weights that served.
- **Loading is control, not config.** Putting a model into a slot is an
  operational act on the engine, not a routing binding. Keeping that in a
  separate layer means Airo's routes and the host's resident state can differ
  on purpose, and a crashed engine is an event Airo reacts to rather than a
  config drift it discovers later.
- **Crash isolation.** A CUDA OOM or a segfault in an engine kills one
  supervised OS process, never the agent, and the agent going away never kills
  the engine (a hard-restarted agent reaps orphans on the next boot). Airo
  being down never takes serving down either — the channel client runs under
  its own wide-budget supervisor.
- **More than one Airo can watch.** An agent pushes to exactly **one
  controller** (`AIRO_SOCKET_URL`) and any number of **observers**
  (`AIRO_OBSERVER_SOCKET_URLS`). Observers get the same stream and may route to
  the slots, but the controller alone may load and unload. That is how a dev
  Airo on a workstation sees the production fleet live without being able to
  disturb it.

## How

**Two channels, two directions.** The agent is the WebSocket *client*: it joins
Airo's `/agent` socket as `agent:<host_id>` and pushes a full `register`
(identity + every slot + GPU) on join, on every rejoin, and as a 10 s heartbeat,
plus a `slot` message on each transition (`loading → up | down | failed`). Airo
*commands* it over HTTP on `control_url` (`POST /load`, `POST /unload`, …) and
only ever gets an "accepted" back — the result arrives as a push. If the socket
drops or the heartbeats stop, Airo marks the host disconnected or stale and its
deployments unknown.

**Slots.** `AIRO_AGENT_SLOTS` is a CSV of serving ports; each holds at most one
resident model. `Fleet` is the lifecycle brain (one GenServer, canonical state);
`Instance` owns one engine OS process via `MuonTrap.Daemon` and polls readiness;
a swap is unload-then-load with the intent recorded first, so a `:DOWN` can be
told apart from a crash.

**Engine-neutral seam.** Adapters (`Engine.LlamaCpp`, `Engine.Vllm`) are pure:
`inventory/1` scans, `launch_spec/3` builds argv and env, neither spawns. The
same launch *profile* is portable across hosts; keys an engine doesn't read are
dropped and logged. Two contracts differ and are easy to get wrong — llama.cpp's
`-c` is the **total** KV budget (`ctx × parallel`), vLLM's `--max-model-len` is
the **per-request** window — see the adapter moduledocs.

**vLLM runs in a container.** The `:vllm` "binary" is `priv/engine/vllm-slot`, a
bash wrapper around `docker`/`podman run` that adds the boilerplate (GPU flags,
`--ipc=host`, the HF cache mount, read-only file overlays for patched
tokenizer/kernel files), names the container `airo-slot-<port>`, and waits for
the runtime to release that name before relaunching. In cluster mode it also
starts rank 1 on the worker Spark over SSH so both ranks share one supervised
lifetime. It is bash 3.2 safe, so its tests run on macOS too.

**Deploy.** `bin/deploy.sh` builds a glibc-matched release per architecture in a
container and rolls it to hosts **one at a time** with a health gate, because
restarting an agent drains every engine it owns. Config is entirely env
(`/etc/airo-agent.env`, installed from `deploy/hosts/<host>.env`).

Design: [`DESIGN.md`](DESIGN.md) is the source of truth (Model 2: one agent, N
slot-providers); [`DESIGN-vllm.md`](DESIGN-vllm.md) and
[`DESIGN-cluster-slots.md`](DESIGN-cluster-slots.md) cover the vLLM and two-host
specifics; roles and liveness are specified in Airo's
`docs/design/DESIGN-agent-lifecycle-and-roles.md`.

## Control API

HTTP, management only — inference goes **directly** to a slot's `base_url`, never
through here. The published OpenAPI contract is served at `GET /openapi`
(`AiroAgent.Api.Spec`).

```
GET  /health
GET  /inventory          -> { models: [ModelRef] }   # local catalog + provenance
GET  /slots              -> { slots:  [SlotInfo]  }   # ports, resident model, runtime facts
GET  /gpu                -> GPU telemetry snapshot
POST /inventory/refresh  -> rescan the local catalog
POST /load   { model, slot, profile? }  -> SlotInfo   # place/swap a model into a slot
POST /unload { slot }                   -> { ok: true }
GET  /openapi            -> the OpenAPI spec for all of the above
```

`profile` carries launch knobs. Note the context contract (**A**): `ctx` is the
**per-request** window — the agent sets llama-server's `-c = ctx × parallel`,
since `-c` is the total KV budget split across `--parallel` sequences. So
`{ "ctx": 146432, "parallel": 1 }` gives one request the full ~142k window.
(vLLM has no ×parallel: `ctx → --max-model-len` directly, and a load *without*
`ctx` is capped at `min(ctx_max, 32768)` so vLLM's own full-window default
can't OOM a small card.)

Engine-neutral knob: `"disable_thinking": true` turns off reasoning traces at
launch — llama.cpp gets `--reasoning off`, vLLM gets
`--default-chat-template-kwargs '{"enable_thinking": false}'`. vLLM loads also
auto-detect the model's tool-call format from its chat template and add the
matching `--tool-call-parser` (an `extra_argv` carrying `--tool-call-parser`
overrides).

`POST /load` accepts the **union** of every engine's profile keys, so one
profile is portable across hosts and any key the target engine doesn't read is
simply dropped. That is deliberate, but it is not silent: the agent logs
`ignoring profile key(s) …` at launch, naming them. The asymmetry worth knowing
is **sampling** — `temperature`, `top_p`, `repeat_penalty`, `presence_penalty`
and `frequency_penalty` reach llama-server as server-side defaults, and the vLLM
adapter maps none of them (vLLM's own whitelist is narrow, and per-request
params override server defaults anyway — set sampling in Airo, or pass
`--override-generation-config` via `extra_argv`). Each adapter declares what it
reads via `Engine.honored_profile_keys/0`, and a test pins that list against the
API allowlist in both directions, so a key an engine reads can never again be
dropped before it arrives.

## Configuration (env)

| Var | Default | Purpose |
|---|---|---|
| `AIRO_AGENT_PORT` | `4400` | Control API port. |
| `AIRO_AGENT_SLOTS` | `8081` | Comma-separated serving-slot ports. |
| `AIRO_AGENT_ADVERTISE_HOST` | auto (primary LAN IP) | Host Airo reaches this agent + its engines at. A non-loopback value **exposes** the control API and engines on `0.0.0.0`. |
| `AIRO_AGENT_MODEL_ROOT` | `~/.cache/huggingface/hub` | Where to scan for GGUF models. |
| `AIRO_AGENT_HOST_ID` | hostname | Stable id Airo keys the host on. |
| `AIRO_AGENT_TOKEN` | — | Bearer token for the control API + channel join (optional; independent of exposure). |
| `AIRO_SOCKET_URL` | — | The **controller** Airo's `/agent` socket — the one Airo allowed to load/unload here. When set, the channel notifier connects and pushes state (else log-only). |
| `AIRO_OBSERVER_SOCKET_URLS` | — | CSV of **observer** Airo `/agent` sockets (S26). Each gets the same register/slot stream, joined with `role=observer`; Airo refuses load/unload from an observer. Leave unset (not empty) when none. |
| `AIRO_VLLM_EXTRA_ARGS` | — | Host-level vLLM tuning argv (e.g. `--enforce-eager --max-num-batched-tokens 2048` on a 16 GB card), used when a load profile has no `extra_argv` of its own. |
| `AIRO_VLLM_CLUSTER_WORKER_SSH` | — | Two-host TP (head host only): ssh target for the worker GPU host (key auth), e.g. `jody@192.168.100.11`. Set together with `_MASTER_IP` + `_NCCL_IF` to accept `nnodes: 2` loads. |
| `AIRO_VLLM_CLUSTER_MASTER_IP` | — | Head's fabric IP (torch.distributed rendezvous + rank-0 `VLLM_HOST_IP`). |
| `AIRO_VLLM_CLUSTER_WORKER_IP` | host part of `WORKER_SSH` | Worker's fabric IP (rank-1 `VLLM_HOST_IP`). |
| `AIRO_VLLM_CLUSTER_NCCL_IF` | — | NCCL/GLOO socket interface on both hosts (e.g. `enP2p1s0f1np1`). |
| `AIRO_VLLM_CLUSTER_NCCL_HCA` | — | RDMA device for `NCCL_IB_HCA` (e.g. `roceP2p1s0f1`; optional). |
| `AIRO_VLLM_CLUSTER_GID_INDEX` | — | `NCCL_IB_GID_INDEX` — must be the RoCE v2 + IPv4 GID (5 on DGX Spark; optional). |
| `AIRO_LLAMA_REASONING_BUDGET` | `8192` | Default `--reasoning-budget` (thinking-token cap) for llama.cpp loads that bring no `reasoning_budget` of their own — the runaway-thinking guard. `-1` disables the guard host-wide (the engine's unlimited default). |
| `AIRO_LLAMA_DRY_MULTIPLIER` | `0` (off) | Optional repetition guard: default `--dry-multiplier` for llama.cpp loads that bring no `dry_multiplier` of their own (DRY breaks degenerate loops, near-inert on normal text). Set e.g. `0.8` to enable host-wide; per-request sampler params still override. |
| `LLAMA_SERVER_BIN` | `llama-server` (PATH) | Engine binary. |
| `LLAMA_CPP_LIB` | — | `LD_LIBRARY_PATH` for the engine (shared-lib builds). |

## Run

```sh
mix deps.get
mix test
iex -S mix          # boots the supervision tree (GPU, Inventory, Fleet, API, notifier)
```

In dev on the serving host it runs as a systemd **user** service (`airo-agent`)
via `mix run --no-halt`; for prod, build an OTP release and run it as a system
unit alongside the engines.
