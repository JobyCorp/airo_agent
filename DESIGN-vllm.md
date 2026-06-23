# DESIGN — vLLM as a second engine (and the first arm64 host)

Extends [`DESIGN.md`](DESIGN.md). That doc already names this: *"Implementations:
`AiroAgent.Engine.LlamaCpp` (now), `.Vllm`, `.Tgi` (later)."* This is the `.Vllm`
plan, written against a concrete first host — a **DGX Spark** (`sparky`).

> **Engine serves. Agent controls. Airo decides.** vLLM is just another engine
> the agent controls; nothing about that rule changes.

## Thesis: the seam already fits; the host is what's new

`AiroAgent.Engine` is an engine-neutral behaviour, and `AiroAgent.Instance`
spawns *any* `{bin, argv, env}` under muontrap and polls a readiness predicate.
Nothing in `Fleet`, `Instances`, the slot model, or the channel mentions
llama.cpp. So vLLM is **one new adapter** (`AiroAgent.Engine.Vllm`) registered in
`Engine.@adapters`, plus a small set of **engine-neutral gaps** that this
particular host forces open (telemetry on unified memory, per-host engine
selection, an arm64 release).

The slot model maps cleanly: a vLLM process on a static port = one Provider, one
resident model, swap = relaunch. Identical to llama-server. Concurrency *within*
a slot is vLLM's continuous batching; concurrency *across* slots is more ports.
Unchanged.

## Ground truth: what `sparky` actually is

Probed 2026-06-23 (`ssh sparky`):

| | Value | Consequence |
|---|---|---|
| Arch | **aarch64** | release must be built for arm64 (see Build) |
| OS | Ubuntu 24.04.4 LTS | distro matches the x86 fleet — only arch differs |
| CPU / Mem | 20 cores / **121 GiB unified** | "VRAM" budget = the unified pool, not discrete |
| GPU | **NVIDIA GB10** (Grace-Blackwell), driver 580.159.03, CUDA 13.0 | unified memory; `nvidia-smi` FB memory = **N/A** |
| Toolchain | Python 3.12.3, **no `nvcc`**, **no Elixir** | needs bundled-ERTS release; containerized vLLM |
| Containers | **docker 29.2.1**, no podman | we install **podman** (see Spawn) |
| vLLM | not installed | host bootstrap (out of band) |
| HF cache | `~/.cache/huggingface/hub`, 193 GiB | safetensors repos for vLLM already present |

Two of these are load-bearing and surprised us: **GB10 reports no GPU memory via
NVML** (`memory.total/used/free = N/A`; only util + power populate), and the box
is **arm64**. Both are handled below.

## The adapter: `AiroAgent.Engine.Vllm`

Implements the same five callbacks as `LlamaCpp`. Differences are about *format*
(safetensors, not GGUF) and *launch semantics* (continuous batching, not a split
KV budget) — not about the seam.

### Callback map

| Callback | llama.cpp (today) | vLLM |
|---|---|---|
| `inventory/1` | glob `**/*.gguf`, read GGUF metadata | scan `models--*/snapshots/<sha>/config.json`; read `config.json` |
| `launch_spec/3` | `llama-server -m <file> -c …` | `vllm serve <snapshot-dir> --served-model-name <id> …` (via podman) |
| `default_profile/1` | ngl / flash-attn / cache-type | `tensor-parallel-size:1`, `gpu-memory-utilization`, `dtype:auto`, `max-num-seqs` |
| `resolve_profile/2` | merge over defaults | same |
| `capabilities/1` | `:vision` if `mmproj` sibling | from `config.json` architectures |
| `runtime_props/1` (bespoke) | scrape `/props` | scrape `/v1/models` (`max_model_len`) + `/metrics` (version) |
| readiness | `{:http_get, "/health"}` | **identical** — vLLM is OpenAI-compatible |

### The one trap: the ctx contract is llama.cpp-specific

`DESIGN.md`'s **Contract A** — `llama-server`'s `-c = ctx × parallel`, one KV
budget split across `--parallel` sequences — is a **llama.cpp quirk and must not
leak into vLLM.** vLLM's `--max-model-len` *is* the per-request window directly;
concurrency is continuous batching over a paged KV cache, bounded by
`--gpu-memory-utilization` and capped by `--max-num-seqs`. So in the vLLM
adapter:

- `profile.ctx → --max-model-len` (**no `× parallel`**),
- `profile.parallel → --max-num-seqs`,
- `ctx_total` (the split-KV total) has **no vLLM analogue** — report `nil`, or
  echo `max_model_len`. PagedAttention owns the KV budget internally.

`SlotInfo` already carries `ctx` / `parallel` / `ctx_total`; the adapter fills
them with vLLM's meaning, and the contract-A multiply stays buried in `LlamaCpp`.

### Launch (profile → vLLM flags)

```
vllm serve <snapshot-dir>
  --served-model-name <ModelRef.id>      # so Airo routes on the id it knows, not the path
  --host <engine_bind_host> --port <P>
  --max-model-len <ctx>                  # contract: ctx is per-request, no ×parallel
  --max-num-seqs <parallel>
  --tensor-parallel-size <tp>            # 1 on a single GB10
  --gpu-memory-utilization <frac>        # NB: fraction of the UNIFIED pool — validate on Spark
  --dtype <dtype> [--quantization <q>] [--kv-cache-dtype <kv>]
  <extra_argv…>
```

- **`--model` is the local revision-pinned snapshot dir**, so vLLM serves exactly
  the sha inventory resolved — provenance preserved end-to-end, same guarantee
  GGUF gives. Set `HF_HUB_OFFLINE=1` so it never reaches for the network.
- **`--served-model-name <id>`** is mandatory: vLLM otherwise advertises the ugly
  snapshot path as the model id, breaking Airo's `model` routing. (llama.cpp
  didn't need this — Airo routes to the provider, and there's one model per slot.)

### Inventory & provenance (safetensors)

Same HF cache layout as GGUF (`models--ORG--REPO/snapshots/<sha>/…`), so the
**revision sha — the whole reason this app exists — resolves identically.** A
vLLM `ModelRef` is built from `config.json`:

- `repo`, `revision` — from the path (see refactor below).
- `family` — `config.json` `model_type` / `architectures`.
- `ctx_max` — `max_position_embeddings`.
- `quant` — `quantization_config.quant_method` (fp8 / awq / gptq / nvfp4) or `nil`.
- `size_bytes` — sum of `*.safetensors`.
- `path` — the **snapshot dir** (vLLM takes a directory).
- `engine: :vllm`.

Coexistence: a repo may ship **both** GGUF and safetensors. Each is inventoried
under its own `engine` tag and surfaces to Airo as a distinct candidate
Deployment on a distinct Provider. The llama.cpp glob already ignores
safetensors; the vLLM scan ignores GGUF-only repos (no `config.json` + no
`*.safetensors`).

**Refactor:** `repo_and_revision/1` lives in `LlamaCpp` but is format-agnostic.
Extract it (and the `models--`/`snapshots` parsing) to **`AiroAgent.HFCache`** so
both adapters share one provenance resolver. This is the only change to existing
engine code.

## Spawn & lifecycle: podman (locked)

This is where vLLM diverges most operationally. llama-server is a static binary
muontrap reaps cleanly. vLLM on Grace-Blackwell is realistically a **container**
(NVIDIA ships tuned arm64 vLLM images; bare-metal ARM wheels are painful, and the
host has no `nvcc`). The hazard: **`docker run` hands the container to `dockerd`,
so muontrap killing the client does *not* reap the container** — silently
breaking the architecture's load-bearing invariant ("the engine dies with the
BEAM"), which both crash-isolation and the deploy-drain model depend on.

**Decision: podman (daemonless, rootless).** The container runs inside the
agent's own process tree, not under a separate daemon, so muontrap's `SIGTERM`
(and the deploy/stop drain) tears it down like a native binary. Concretely:

- `engine_bin[:vllm]` → a small shipped wrapper `vllm-slot` (in the release).
  `Vllm.launch_spec/3` returns the **vLLM** args; the wrapper adds the podman
  boilerplate from host config (image, cache mount, GPU device, port, name).
- **Named, idempotent containers** — one per slot, `airo-slot-<port>`. Every load
  (and agent boot) runs `podman rm -f airo-slot-<port>` first, then
  `podman run --rm --name airo-slot-<port> …`. This makes swaps and crash
  recovery clean **regardless** of any orphan: the next load force-removes a
  stale container. Belt: the wrapper traps `SIGTERM → podman stop`.
- **GPU passthrough via CDI** — `--device nvidia.com/gpu=all`
  (`nvidia-ctk cdi generate`), the daemonless path. `nvidia-container-toolkit` on
  the host is bootstrap.
- **Cache mount** — `-v <model_root>:<model_root>:ro` so the snapshot path the
  adapter passes resolves identically inside the container.

`Instance`/`Fleet` are untouched: to them this is just another supervised child
that becomes ready when `GET /health` returns 200.

## Three engine-neutral gaps this host forces open

Not vLLM-specific — these are "the fleet is now heterogeneous" gaps.

### 1. GPU telemetry is blind on GB10 (required fix)

`AiroAgent.GPU.poll` queries `memory.total` → **`N/A` on GB10** → `vram_total_mb:
nil` → Airo can't size the box, so it can't place anything. Fix: a per-host
**memory-source strategy**.

- Detect unified memory (NVML FB total `N/A`, or explicit `AIRO_AGENT_GPU_MEM=unified`).
- On unified hosts, report the pool from the **OS** (`/proc/meminfo`):
  `vram_total_mb = MemTotal`, `vram_used_mb = MemTotal − MemAvailable`. Keep
  `util_pct` / `power_*` from `nvidia-smi` (those *do* populate on GB10).
- Caveat (Airo-side, noted not solved here): unified memory is shared with the
  CPU/OS, so the "VRAM budget" is softer than discrete VRAM. Airo's placement
  policy should apply a headroom factor on unified hosts.

### 2. Engine selection must go per-host

`engines: [:llama_cpp]` is hardcoded in `config.exs` (compile-time). Make it
runtime: `AIRO_AGENT_ENGINES` (CSV) in `runtime.exs` → `engines: [:vllm]` on
sparky, `[:llama_cpp]` on jobycorp. Add a `:vllm` entry to the engine-bin /
launcher config and `vllm_image` / `vllm_extra_mounts` keys.

### 3. The release for sparky must be arm64 (locked: buildx on the 9950X3D)

sparky is `aarch64` with **no Elixir** → it needs the bundled-ERTS release
([DEPLOY.md](DEPLOY.md)), built for arm64. The decisive fact:

> **airo_agent has zero NIFs and exactly one native C file**
> (`deps/muontrap/c_src/muontrap.c`). Everything else is **architecture-independent
> BEAM bytecode.**

So an emulated arm64 build only has to compile *one small .c file* plus bundle the
arm64 ERTS — qemu never faces a heavy native compile. On the build host (a Ryzen
9950X3D) that's trivial. The plan:

- **Swap the builder to a precompiled-OTP, multi-arch base.** The current
  `bin/docker-build/Dockerfile` compiles OTP from source via mise/kerl — *that*
  would be brutal under emulation. Rebase on
  `hexpm/elixir:1.19.5-erlang-28.x-ubuntu-24.04` (precompiled OTP for both
  arches) + `build-essential` for muontrap's `cc`. Bonus: the x86 build gets
  faster too (no kerl), and the Dockerfile shrinks.
- **One build host for the whole fleet** via `docker buildx`:
  - jobycorp (x86) → `--platform linux/amd64`, native.
  - sparky (arm64) → `--platform linux/arm64`, emulated but cheap.
- `deploy.sh` gains a per-host `PLATFORM` (from `deploy/hosts/<host>.env`, or
  `ssh <host> uname -m`). One-time host setup: register binfmt
  (`docker run --privileged tonistiigi/binfmt --install arm64`).
- **Prod boxes stay Elixir-free** (the bundle decision holds) and build load
  never lands on the GPU box. Fallback if qemu ever disappoints: build on sparky
  itself in an arm64 container (`git archive HEAD | ssh sparky`) — native, but
  puts build load + source on a serving box; only if needed.

## New / changed modules

| Module | Change |
|---|---|
| `AiroAgent.HFCache` | **new** — shared `repo_and_revision/1` + snapshot scanning, extracted from `LlamaCpp` |
| `AiroAgent.Engine.Vllm` | **new** — the adapter (callbacks above) |
| `AiroAgent.Engine` | register `:vllm` in `@adapters` |
| `AiroAgent.Engine.LlamaCpp` | use `HFCache` (no behaviour change) |
| `AiroAgent.GPU` | unified-memory strategy (`/proc/meminfo` fallback) |
| `config/runtime.exs` | `AIRO_AGENT_ENGINES`, `vllm_image`, `engine_bin[:vllm]` |
| `priv/engine/vllm-slot` | **new** — podman wrapper (named container, CDI, mount, trap) |
| `bin/docker-build/Dockerfile` | precompiled-OTP multi-arch base |
| `bin/deploy.sh` | per-host `PLATFORM` for `buildx` |

## Phased plan

- **Phase 0 — engine-neutral prep (no vLLM yet).** `HFCache` extraction;
  `AIRO_AGENT_ENGINES` per-host config; GPU unified-memory telemetry; builder →
  precompiled multi-arch + `deploy.sh PLATFORM`. Outcome: an **arm64 agent runs
  on sparky** and reports correct telemetry, serving nothing.
- **Phase 1 — vLLM inventory + launch (manual engine).** `Engine.Vllm`
  inventory / `launch_spec` / readiness / capabilities, validated against a
  **hand-started** vLLM container on sparky. Agent loads/unloads a model via the
  podman wrapper.
- **Phase 2 — spawn hardening.** Named-container reconcile + crash recovery; CDI
  GPU passthrough validated on GB10; `runtime_props` scrape (`/v1/models` +
  `/metrics`).
- **Phase 3 — provenance & telemetry polish.** `--served-model-name = id`;
  vLLM ctx semantics in `SlotInfo`; unified-memory budget reported to Airo with
  headroom.

## Out of scope (and who owns it)

- **Model acquisition** (HF download to the cache) — host bootstrap, as today.
- **vLLM image / `nvidia-container-toolkit` / podman install on sparky** — host
  bootstrap.
- **Airo's placement policy for unified memory** — Airo-side; this doc only makes
  the agent *report* a usable budget.
- **Tensor-parallel across multiple GB10s** — single-host TP only for now.
- **TGI** — drops into the same seam later; nothing here blocks it.
