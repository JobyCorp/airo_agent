# airo_agent — host-side control surface for local model serving

## Purpose

`airo_agent` is a small OTP service that runs **on each model-serving host** and
gives Airo a clean, engine-neutral **control surface** over a local inference
engine — the kind of knobs LM Studio exposes, but headless and consumed by Airo
instead of a GUI.

It exists because traditional backends (Ollama, LM Studio, Unsloth Studio) are
**opaque about model identity**. They tell Airo `qwen3:4b`, not *which* GGUF /
HF revision. Airo's model shelf already has the schema for provenance and
per-revision performance (`VERSION`, `REVISION`, version-performance panel) — it
just lacks a backend that tells it the truth. `airo_agent` is that backend,
because **Airo (via the agent) chooses the exact artifact**, so the snapshot sha
*is* the revision.

## Boundary (the one rule)

> **Mechanism in the agent. Policy and the data path in Airo.**

| Layer | Owns |
|---|---|
| **Airo** | Router, manager, **model shelf**. Provenance, version-performance, HF-update detection, **load/evict policy**. Routes inference. Drives serving *only* by calling the agent. Never spawns an engine. |
| **airo_agent** | The control surface. Inventory (with provenance), engine lifecycle (load/unload/running), GPU telemetry. The *mechanism*. |
| **engine** (`llama-server` now; vLLM/TGI later) | Serves inference. Airo calls it **directly**. |

The agent is **never on the inference data path** — that's the explicit reason
we did *not* use `llama-swap` (a data-path proxy that duplicates Airo's gateway,
model list, and swap policy). Airo gets `base_url` from the agent and connects
straight to the engine.

## Components (supervision tree)

```
AiroAgent.Supervisor (:one_for_one)
├── AiroAgent.GPU            # nvidia-smi telemetry, polled + cached
├── AiroAgent.Inventory      # local catalog w/ provenance (HF snapshot sha = revision)
├── AiroAgent.Instances      # DynamicSupervisor: one OS process per running model
│     └── AiroAgent.Instance # owns a llama-server via MuonTrap.Daemon (restart: :temporary)
└── Bandit(AiroAgent.Api.Router)   # control API; loopback unless a token is set
```

Crash isolation is the point of doing this in OTP: a native engine crash (CUDA
OOM, MTP edge case, segfault) dies inside one supervised `Instance`, is reaped
by MuonTrap, and never takes down the agent — let alone Airo. (This is exactly
why the engine is an external process, **not** a NIF. Contrast Ortex, which is
correctly a NIF: small, CPU, stable.)

## Engine-neutral seam

`AiroAgent.Engine` is a behaviour; everything backend-specific is behind it:

- `inventory/1` — enumerate local models with provenance.
- `launch_spec/3` — **pure** argv/env builder for `{model, profile, port}`.
- `default_profile/1` — sane defaults (e.g. `--flash-attn on`, q8_0 KV, `--spec-type draft-mtp` when the GGUF ships an MTP head).
- `capabilities/1`.

Process supervision, readiness polling, and the HTTP API are generic. Adding
vLLM later = one new adapter module; the contract Airo speaks does not change.
`profile` is an **opaque per-engine blob** Airo stores on the shelf and passes
through verbatim.

Keeping `--spec-type draft-mtp` available is a first-class reason to wrap a
*bare* `llama-server` from our own build rather than a vendor runtime: the MTP
speedup survives.

## Control contract (Airo ↔ agent)

```
GET  /health
GET  /inventory          -> { models: [ModelRef] }   # provenance source for the shelf
POST /inventory/refresh  -> rescan
GET  /running            -> { instances: [InstanceInfo] }
GET  /gpu                -> telemetry snapshot (VRAM budget input for Airo's policy)
POST /load   {model, profile?}  -> InstanceInfo   # { base_url, revision, status, ... }
POST /unload {model}            -> { ok: true }
```

`ModelRef` carries `revision` (HF snapshot sha), `quant`, `size`, `path`.
`InstanceInfo` carries `base_url` + `revision` — Airo points its OpenAI client
at `base_url` and stamps `revision` onto usage rows.

To be published as an `open_api_spex` document (same dialect Airo already uses)
so the contract is generated, not prose.

## Provenance flow (the payoff)

1. `inventory` resolves `…/snapshots/<sha>/file.gguf` → `revision = <sha>` → fills Airo's `VERSION`/`REVISION`.
2. Airo polls the HF API for the repo's current sha vs the deployed one → "update available".
3. `load` returns the launched `revision`; Airo tags usage rows → `Version performance` becomes per-sha → regression detection.

None of this is possible on Ollama/LM Studio, because they never surface the
revision or correlate per-version performance. That capability is what justifies
building rather than adopting.

## Airo side (separate change, in `airo`)

A `local-gguf` provider driver — a *client* of this contract, alongside the
existing Infinity/Ollama drivers. It:

- pulls `inventory` → shelf + provenance;
- on a routing decision, checks `running`, calls `load` if cold, applying **Airo's** evict policy under a VRAM budget (`/gpu`);
- routes inference directly to `base_url`;
- stamps `revision` onto usage; polls HF for newer revisions.

## Security

The agent can spawn processes on the GPU host → treat as privileged. Default:
**loopback-only** bind; widen to `0.0.0.0` *only* when `AIRO_AGENT_TOKEN` is set,
and require it as a bearer token on every request. (Lesson from the Unsloth
Cloudflare-tunnel default: never expose a serving-control surface unauthenticated.)

## Fleet topology

One agent per serving host — jobycorp (llama.cpp), sparky, the vLLM box — all
speaking the same contract, with **Airo as the single cross-host brain/shelf**.
Same provider pattern Airo already uses, but one where Airo owns the launch
intent and therefore the provenance.

## Deploy

Self-contained OTP release installed alongside the engine on each host, run as a
systemd unit. Config via env (`AIRO_AGENT_PORT`, `AIRO_AGENT_TOKEN`,
`AIRO_AGENT_MODEL_ROOT`, `LLAMA_SERVER_BIN`, `LLAMA_CPP_LIB`); see `runtime.exs`.

## Open questions / TODO

- [ ] Publish the contract as `open_api_spex`.
- [ ] Readiness: `/health` 200 means *server* up; gate on first successful `/v1/models` for *model* ready.
- [ ] GGUF header parse for `family`/`ctx_max` (currently nil) — read metadata kv directly.
- [ ] VRAM accounting on `load` (reject/evict before OOM) vs. leaving all policy to Airo.
- [ ] Desired-state reconcile loop vs. the current imperative load/unload.
- [ ] Streaming engine logs to Airo for the loading phase.
- [ ] `MuonTrap` cgroup limits per instance?
