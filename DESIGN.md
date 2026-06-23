# airo_agent — host-side model-serving control plane

## Purpose

`airo_agent` is a small OTP service that runs **on each model-serving host**. It
is a **control plane** over local inference engines (`llama-server` now,
vLLM/TGI later): it starts and stops engines, swaps which model each one serves,
surfaces local-model provenance, and reports host telemetry — and it pushes all
of that to Airo over a channel so Airo never has to poll.

It is **not** a server of models and it is **not** a provider. The engine
(`llama-server`) is the server; Airo is the brain. `airo_agent` is the
mechanism in between.

## The one rule — three layers, one job each

> **Engine serves. Agent controls. Airo decides.**

| Layer | Is | Owns |
|---|---|---|
| **Engine** (an Airo **Provider**) | the server | OpenAI-compatible inference on a **static** port. Spawned by the agent, or run externally. |
| **airo_agent** | the mechanism | for engines **it spawned**: start/stop, load/unload/swap a model, provenance inventory, host GPU/VRAM telemetry, health push. |
| **Airo** | the intent | routing, load/evict/VRAM **policy**, the model shelf, host grouping. |

Everything below is a consequence of this table. When a question is ambiguous,
resolve it by asking which layer the concern belongs to.

## Topology (Model 2: one agent, N providers)

```
HOST = AGENT  (one control plane per host; owns the host GPU view)   ← Airo's /agents entity
 ├─ spawns → Provider "slot A"  llama-server :8081/v1   ┐ agent_id = this agent
 ├─ spawns → Provider "slot B"  llama-server :8082/v1   ┘  (managed)
 └─ (external) vLLM :8000/v1  →  Provider               agent_id = null  (unmanaged; agent ignores)

   serving ──►  provider.base_url        (real, static — a genuine OpenAI endpoint)
   control ──►  the host agent           (load/unload/swap per slot, over its own control API)
   GPU/VRAM ─►  the host agent           (host-level; reported to Airo as policy input)
   policy   ──►  Airo                     (what to load/evict, using that telemetry)
```

The decisive point: **the agent is not a provider.** The *engines* are the
providers. The agent is a separate, host-resident control plane that manages the
engines it spawns.

## Roles & responsibilities

**The agent IS responsible for** (only for engines it spawned):

1. **Engine lifecycle** — spawn and supervise an engine process per serving
   slot, crash-isolated; load / unload / **swap** the resident model.
2. **Slot registration** — register each slot with Airo as a real Provider
   (stable `base_url`) and keep its resident-model + up/down state in sync.
3. **Provenance inventory** — scan local artifacts; resolve the HF snapshot sha
   (`revision`) — the fact Ollama/LM Studio can't give.
4. **Host telemetry** — GPU/VRAM for the host; pushed to Airo as policy input.
5. **Health push** — engine readiness/liveness over the channel; no Airo polling.

**The agent IS NOT:**

- **A provider, or on the inference data path.** It never serves or proxies a
  token. Airo's OpenAI client connects to the *engine* directly.
- **The router or the load/evict/VRAM decider.** It receives a launch profile
  and a load/unload command and *executes* them. It never *decides*.
- **An adopter of foreign engines.** It manages only engines it spawned. An
  externally-run vLLM/llama-server is a plain static Provider (`agent_id = null`)
  that Airo talks to directly; the agent neither wraps nor touches it. **No
  provider needs any Airo-specific customization** — engines are stock
  OpenAI-compatible binaries.
- **A cross-host brain.** One agent per host; Airo aggregates the fleet.
- **A model fetcher.** Acquisition (HF download) is out of band for now.

## Core concepts

- **Provider (the engine).** A durable, OpenAI-compatible serving endpoint at a
  **stable port**. `base_url` is a real inference URL. For managed providers,
  `Provider.agent_id` points at the managing agent; for external ones it is
  `null`. This is unchanged from how Airo already models vLLM/Ollama — the
  engine just happens to be agent-managed.
- **Slot.** The intuition for a *managed* Provider: a stable serving port that
  holds **one model at a time**. The agent swaps models into a slot. Concurrency
  on a host = **more slots** (more ports), each its own Provider. A single-GPU
  box that runs one big model = one slot; a big-VRAM box = a few. Slots are
  **declared** (a configured port set), so they're durable — no dynamic-port
  churn.
- **Agent (Airo entity).** One per host: `host_id`, `control_url`, version, GPU
  telemetry, presence. Surfaced in Airo's `/agents` view. It *manages* providers
  but is not itself one. `Provider.agent_id` is the only link.
- **Deployment.** Unchanged: a `(provider, model)` binding Airo routes to. A
  managed slot-Provider may have several candidate Deployments; at any instant
  ≤1 is **resident** (loaded). Which one is resident is runtime state the agent
  reports; *which one should be* is Airo's placement policy.

## The slot model (static ports, swap, concurrency)

- The agent is configured with a set of serving **ports** (its slots). Each slot
  is registered as a durable Provider with `base_url = http://<host>:<port>/v1`.
- **Load** = launch the engine on a slot's port with model M (+ launch profile).
- **Swap** = load a different model into an occupied slot (implicit
  unload-then-load on the same port).
- **Concurrency** = number of slots. Two models served at once ⇒ two slots ⇒ two
  ports ⇒ two engine processes. Nothing is dynamic; every `base_url` is stable.
- **Placement** — *which model goes in which slot, and what to evict* — is
  **Airo's policy**, decided from the agent's GPU telemetry + the set of
  resident models. The agent only executes "load M into slot S" / "unload S".

This is the heart of why Model 2 is coherent: a Provider is always a real,
stable serving endpoint, because a slot is a declared port, not a per-load
allocation.

## Launch profile: context vs concurrency

There are **two** concurrency axes, easily confused:

1. **Across slots** — more slots ⇒ more models served at once (above).
2. **Within a slot** — `llama-server --parallel N` serves N concurrent requests
   from *one* engine.

The trap is how they interact with context. `llama-server`'s `-c` is the
**total** KV-cache budget, **split** across the `--parallel` sequences, so each
request sees `-c / parallel`. Under contract A (below) the agent inverts this —
it sets `-c = ctx × parallel` so each request gets the full `ctx` — which means
**`parallel` multiplies VRAM**: every added sequence is another whole `ctx` of KV
cache. The agent reports the real per-request window as `SlotInfo.ctx` (from the
engine's `/props` `n_ctx`) and the total `-c` as `SlotInfo.ctx_total`.

Because of that multiplication, the safe default is **`parallel: 1`** — a bare
`ctx` allocates exactly `-c = ctx`, no surprise. (A previous default of `4`
quadrupled a bare `ctx` into 4× the KV cache; a `ctx: 146432` that fit fine at
`parallel: 1` OOM-crashed the engine at `-c = 585728` when loaded with the
default still at 4. llama-server **segfaults** rather than erroring on a KV
cudaMalloc failure, especially with `-ngl 999` blocking its auto-fit — so an
over-large `ctx × parallel` is not a soft failure.) Concurrency is an explicit,
VRAM-aware opt-in.

Both `ctx` and `parallel` are first-class profile keys (already whitelisted and
plumbed to `-c` / `--parallel`); Airo owns them per-Deployment. The defects were
that the contract was implicit and the knob is invisible in the UI.

**Contract — DECIDED: (A) `ctx` is the per-request window.** The agent sets
`-c = ctx * (parallel || 1)`, so "set ctx, get ctx" holds regardless of
`parallel` — aligning with how every model card, and Airo's own `Model.ctx_max`,
use "context". Consequence: VRAM grows with `parallel` (each added sequence is
another full `ctx` of KV), so an over-large `ctx × parallel` must be validated
against GPU telemetry (issue **A4**); unvalidated, it surfaces as the engine
going `:failed`. (Rejected: **B** = `ctx` as total `-c` with default
`parallel: 1`; **C** = keep total `-c` and fix only the UI.)

**Agent tasks (this repo) — DONE (`LlamaCpp`, `Fleet`, `SlotInfo`, channel):**
- **A1.** The slot now reports its **resolved profile** (defaults applied) via the
  optional `Engine.resolve_profile/2` callback → `SlotInfo.profile`, so the
  agent's defaults (e.g. `parallel: 1`) are never an invisible blank.
- **A2-data.** `SlotInfo`/slot events now carry **`ctx_total`** (the `-c`
  allocated = `ctx × parallel`, derived from `/props` since llama-server exposes
  no total field) beside `ctx` (per-request) and `parallel`.

**Cross-repo issues (Airo — UI/config, owned by the Airo work, NOT this repo):**
- **A2 (UI legibility).** Render per-request context *and* total KV + parallelism
  together on the slot/deployment view (e.g. "36,608/req · ×4 parallel ·
  146,432 total"). Consumes `ctx` / `ctx_total` / `parallel` from the slot.
- **A3 (launch-profile WRITE path).** The Deployment config UI must let an
  operator set `ctx`, `parallel`, and the other profile knobs. The read path
  exists; without a write path these are settable only by hand-editing profile
  JSON — which is *why* the agent default silently won here. **This is the fix
  for the reported "set 142k, got 36k" symptom.**
- **A4 (VRAM-fit validation).** Before load, validate requested `ctx`×`parallel`
  against the Agent's GPU telemetry and warn/block on no-fit. An over-large total
  does **not** fail softly: llama-server **segfaults** on the KV cudaMalloc
  failure (observed 2026-06-22 — a `ctx: 146432, parallel: 4` ⇒ `-c 585728` OOM),
  reported by the agent as the slot going `:failed`. Default `parallel: 1`
  removes the common footgun, but a too-large `ctx` (or explicit `parallel`) can
  still overcommit — only a real VRAM check closes it. Part of placement/VRAM
  policy. (An agent-side pre-flight VRAM guard is a possible defense-in-depth,
  but estimating KV size needs more GGUF metadata and overlaps this policy.)

## Control contract (Airo ↔ agent)

**State flows by push (channel); control flows by request (HTTP).** Airo never
polls the agent.

**Registration** (on channel connect / reconnect): the agent announces itself
and its slots; Airo upserts the **Agent** record and each slot **Provider**.
```
agent:  { host_id, control_url, version, gpu }
slots:  [ { port, base_url, resident_model | null, status } , ... ]
```
On reconnect the agent re-sends the full set so Airo's view self-heals
(absent ⇒ down).

**Push** (agent → Airo, channel): per-slot lifecycle/health transitions
(`loading | up | down | swapped`, with the resident model id), coarse GPU
telemetry, and presence (the connection itself = host up; disconnect ⇒ mark the
host's managed providers down).

**Control** (Airo → agent, HTTP on `control_url`):
```
GET  /inventory          -> local models with provenance (revision)
GET  /slots              -> slots + resident model + status
GET  /gpu                -> host telemetry
POST /load   {model, slot}   -> load/swap model into a slot
POST /unload {slot}          -> free a slot
```
The control API is the *management* surface only. It is never used for
inference.

## Serving path

1. A request routes to a Deployment under a managed slot-Provider.
2. Airo checks the slot's resident model. If it isn't the requested one (cold or
   wrong model), Airo applies **placement policy** — pick/evict a slot — and
   calls the agent (`POST /load {model, slot}`), waiting until the engine reports
   ready.
3. Airo serves with a **plain OpenAI call to `provider.base_url`** — the slot's
   real static endpoint. No retarget, no indirection.

The agent is **off this path entirely.** The "ensure the right model is loaded"
step keys on `provider.agent_id` and lives in Airo's dispatch, not in a serving
adapter.

## Provenance (the payoff)

`inventory` resolves `…/snapshots/<sha>/file.gguf` → `revision = <sha>`. Airo
fills the shelf's `REVISION`/`VERSION` from it, polls HF for newer shas
("update available"), and attributes per-revision performance. This is the
capability Ollama/LM Studio/Unsloth can't provide — and the reason to build
rather than adopt.

## Supervision & crash isolation

Each slot's engine is an external OS process owned by the agent via
`MuonTrap.Daemon` (`restart: :temporary`). A native crash (CUDA OOM, segfault,
MTP edge case) dies inside one supervised child, is reaped, and is reported as
that slot going `down` — it never touches the agent VM, let alone Airo. This is
exactly why the engine is an external process, **not** a NIF.

## Engine-neutral seam

`AiroAgent.Engine` is a behaviour; everything backend-specific is behind it
(`inventory`, pure `launch_spec`, `default_profile`, `capabilities`). Adding
vLLM later = one adapter module; the contract Airo speaks does not change. The
launch `profile` (ctx, KV-quant, `--jinja`, reasoning flags, MTP `draft-mtp`,
etc.) is an opaque per-engine blob Airo stores on the Deployment and passes
through verbatim on load. Keeping a *bare* `llama-server` from our own build is a
first-class reason for the seam: the MTP speedup survives.

## Foreign / unmanaged engines

A serving process the agent did **not** spawn (a hand-run vLLM) is a normal Airo
Provider with `agent_id = null`. Airo routes to it directly via its wire adapter
(`:vllm`/`:openai`); it has no lifecycle control and no health push (the prober
covers it, as today). Managed and unmanaged providers coexist on the same host
as peers. If you want Airo to control that vLLM's lifecycle, you don't wrap it —
you let the **agent launch** a vLLM engine (via the engine seam), and it becomes
a managed slot.

## Security

The agent can spawn processes on the GPU host → privileged. Default: control API
**loopback-only**; widen to the LAN only with an explicit advertise host, and
require `AIRO_AGENT_TOKEN` as a bearer when set (also the channel join token).
Engine ports serve on the trusted LAN; per-engine auth (reuse the token as
`--api-key`) is a deferred sprint.

## Deploy

One agent per serving host (not a per-engine sidecar — GPU telemetry, the
artifact inventory, and VRAM coordination are host-level and want a single
host-resident owner). Self-contained OTP release run as a systemd unit alongside
the engines. Config via env (control port, advertise host, token, model root,
engine binary/lib, serving-slot ports, Airo socket URL, host id).

## Airo-side shape (the consumer)

- **`Agent` entity + `/agents` LiveView** — registered hosts: control_url, GPU,
  presence, the providers they manage.
- **`Provider.agent_id`** (nullable) — the managed/unmanaged switch. Managed
  providers carry the real static engine `base_url`; serving is plain
  `OpenAICompatible`.
- **`Airo.Agents` context** — talks to the agent `control_url` for
  load/unload/swap/inventory/gpu, keyed on `agent_id`.
- **Dispatch hook** — before serving a managed provider, ensure the right model
  is resident (placement policy → agent load). Keyed on `agent_id`, in dispatch,
  not in an adapter.
- **Channel ingest** — registration → upsert Agent + slot-Providers; push →
  per-slot health + resident model; disconnect → host's managed providers down.

## Supersedes

This replaces the earlier **agent-as-provider** design, where the agent itself
was the Airo Provider (`base_url` = its control URL), engines were hidden on
**dynamic** ports, and serving **retargeted** the provider's URL to a hidden
engine. That fused two incompatible models (agent-as-provider vs.
agent-as-manager) and made `provider.base_url` a control URL wearing a serving
URL's field — which forced carve-outs (serving retarget, prober skip) and left a
latent footgun (any generic consumer of `base_url` would hit the control port).
Model 2 dissolves all of it: the agent's per-host engine supervision, inventory,
telemetry, and channel push carry over largely intact; the airo side de-fuses
into Agent + `agent_id` + plain OpenAI serving.

## Open questions / out of scope

- **Placement & eviction policy** (Airo): slot selection, VRAM fit, what to evict
  — design TBD; the agent only executes the resulting load/unload.
- **Slot count / port-pool config** ergonomics; whether slots are fully
  pre-declared or drawn from a range and persisted.
- **Swap semantics on a busy slot** — drain in-flight requests before unload?
- Per-engine auth (token as `--api-key`); model pull/`POST /pull`; GGUF-header
  parse for `ctx_max`/`family`; `open_api_spex` publication of the control
  contract.
