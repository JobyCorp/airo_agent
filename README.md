# airo_agent

Host-side model-serving **control plane**, consumed by Airo.

`airo_agent` runs on each model-serving host. It starts/stops local inference
engines (`llama-server` now; vLLM/TGI later), swaps which model each one serves,
surfaces local-model provenance (the Hugging Face snapshot revision), and reports
host GPU telemetry — pushing all of it to Airo over a channel so Airo never polls.

It is **not** a server of models and **not** a provider. The engine is the
server; Airo is the brain. This app is the mechanism in between.

> **Engine serves. Agent controls. Airo decides.**

| Layer | Owns |
|---|---|
| **Engine** (an Airo Provider) | OpenAI-compatible inference on a static port. Spawned by the agent, or run externally. |
| **airo_agent** | for engines it spawned: load/unload/swap, provenance inventory, host telemetry, health push. Never on the inference path. |
| **Airo** | routing, load/evict/VRAM policy, the model shelf. |

See [`DESIGN.md`](DESIGN.md) for the full design (Model 2: one agent, N
slot-providers) — it is the source of truth.

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

## Configuration (env)

| Var | Default | Purpose |
|---|---|---|
| `AIRO_AGENT_PORT` | `4400` | Control API port. |
| `AIRO_AGENT_SLOTS` | `8081` | Comma-separated serving-slot ports. |
| `AIRO_AGENT_ADVERTISE_HOST` | auto (primary LAN IP) | Host Airo reaches this agent + its engines at. A non-loopback value **exposes** the control API and engines on `0.0.0.0`. |
| `AIRO_AGENT_MODEL_ROOT` | `~/.cache/huggingface/hub` | Where to scan for GGUF models. |
| `AIRO_AGENT_HOST_ID` | hostname | Stable id Airo keys the host on. |
| `AIRO_AGENT_TOKEN` | — | Bearer token for the control API + channel join (optional; independent of exposure). |
| `AIRO_SOCKET_URL` | — | Airo `/agent` socket; when set, the channel notifier connects and pushes state (else log-only). |
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
