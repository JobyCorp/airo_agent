# Launch payloads

Known-good `POST /load` bodies, one per model that is fiddly enough to be worth
pinning. Not used by any script — these are for a human (or an agent) restoring
a host by hand after a deploy, a reboot, or an agent restart.

```sh
scp deploy/payloads/dspark.json sparky:/tmp/
ssh sparky 'curl -s -X POST localhost:4400/load \
  -H "Content-Type: application/json" --data-binary @/tmp/dspark.json'
```

Each file is the **effective** profile — the agent's defaults already applied,
exactly as `GET /slots` reports it. So it is explicit and reproducible rather
than relying on whatever the adapter defaults happen to be at the time. They
round-trip: what you `POST` is what `/slots` gives back.

They deliberately carry **no sampling** (`temperature`, `top_p`, penalties).
Airo sets sampling per request, and per-request values override server defaults
anyway. Baking `temperature: 1.1` + `repetition_penalty: 1.05` into the DSpark
load produced gibberish — repetition penalty interacts badly with its
speculative decoding — so `--generation-config vllm` keeps the server neutral.
Note the vLLM adapter maps no sampling keys at all (see
`Engine.Vllm.honored_profile_keys/0`); a genuine server-side default has to ride
in `extra_argv` as `--override-generation-config`.

No secrets belong in here. Unlike `deploy/hosts/*.env` (gitignored, may carry
`AIRO_AGENT_TOKEN`), these files are tracked — keep them to model ids and launch
knobs.

## Files

| File | Host | Notes |
| --- | --- | --- |
| `dspark-0731.json` | `sparky` (+ `sparky2` as rank 1) | **Current.** Official `DeepSeek-V4-Flash-0731`, which ships DSpark folded in — there is no separate `-0731-DSpark` repo. `num_speculative_tokens: 5`, see below. |
| `dspark.json` | `sparky` (+ `sparky2` as rank 1) | Fallback: the abliterated preview checkpoint, still on disk on both Sparks. Kept for the uncensored variant; `k` here is 3 (bump it to 5 if you fall back — see below). |
| `forge-qwen36.json` | `forge` | Single-host vLLM on one RTX 5090. `--attention-backend FLASHINFER` is load-bearing, and `parallel: 4` is a measured ceiling not a guess — see below. |

Both dspark payloads are two-host TP loads (`nnodes: 2`). Reloading is a
**cluster** operation: the head's wrapper starts rank 1 on the worker over SSH,
so posting to sparky alone brings up both ranks. Any agent restart on either
Spark drops the load and it must be reposted. Weights must already exist at the
same snapshot path on both hosts — rsync over the 200G link before a first load.

## `forge-qwen36.json`: keep `--attention-backend FLASHINFER`

Measured 2026-08-07 on forge's 5090 (sm_120), Qwen3.6-35B-A3B — a hybrid with 10
full-attention and 30 linear-attention (GDN) layers. Decode steps/s against
prompt depth, everything else held identical:

| prompt tokens | `TRITON_ATTN` | `FLASHINFER` |
| --- | --- | --- |
| 590 | 95.5/s | 106.2/s |
| 9,171 | 80.4/s | 104.9/s |
| 42,972 | **40.3/s** | **105.0/s** |

Triton loses 58% of its decode rate as context deepens; FlashInfer loses 1%. It
is not speculative decoding — tokens/step is flat at ~2.4 for both — and not
paging: the forced 2128-token attention block size is identical in both runs.
FlashInfer wins every measured cell at c=1 and c=4, short and long, by up to
166%.

**The trap:** enabling MTP forces FlashInfer down from `FULL_AND_PIECEWISE` to
`PIECEWISE` cudagraphs, because it declares only `UNIFORM_SINGLE_TOKEN_DECODE`
and spec-decode needs `UNIFORM_BATCH` (`vllm/config/compilation.py`).
`TRITON_ATTN` declares `ALWAYS` and so *keeps* full graphs — which reads like
the better choice and is not. Full graphs are worth ~8% at c=4; the Triton
kernel costs up to 166% at depth. Do not "fix" the PIECEWISE warning by
switching back.

`--disable-hybrid-kv-cache-manager` is **not usable** on this model: it aborts
with `Hybrid KV cache manager is disabled but failed to convert the KV cache
specs to one unified type`, because the full-attention and GDN layers cannot
share one pool. That also makes the 2128-token attention page permanent.

### `parallel: 4` — do not raise it to 5

Tried and reverted 2026-08-07. The KV pool is not the constraint: it holds
**5.35x** full-depth streams at `ctx: 51200` (273,989 tokens), unchanged between
`parallel: 4` and `5`, and five 43K streams sit resident at 78% of the pool with
no preemption at all. Five *fit*. They just make the box slower.

A synthetic sweep made 5 look good, because it ran **homogeneous** batches:

| | `parallel: 4` | `parallel: 5` |
| --- | --- | --- |
| all-short (~1.8K) aggregate | 728 tok/s | 893 tok/s (+23%) |
| all-long (~43K) aggregate | 107 tok/s | 103 tok/s |
| all-long tail TTFT | 8.3s | 11.4s |

Real agent traffic is **heterogeneous** — decode-heavy worker streams running
next to digest calls that carry large prefills — and there the fifth slot is a
clear loss. Same prompt, same repo, near-identical work (37 vs 39 model calls,
87 vs 81 tool calls), four fan-out sub-agents:

| | `parallel: 4` | `parallel: 5` |
| --- | --- | --- |
| sub-agent wall times | 88s / 70s / 80s / 38s | 117s / 151s / 105s / 110s |

Per-call latency roughly doubled *uniformly* — not spikes, which rules out a
queue or cache pathology and points at batch-wide compute contention. The
mechanism is chunked prefill: a large-prefill request interleaves its prefill
chunks into the same batches as every resident stream's decode. At 4 slots a
digest call waits for a slot and costs one stream's latency; at 5 it is admitted
and taxes all of them.

The lesson generalises past this box: **admission capacity is not throughput
capacity.** A slot that fits in KV can still be net-negative if what lands in it
is prefill-heavy. Benchmark mixed shapes, not uniform ones — a homogeneous sweep
cannot see this.

Model choice is constrained by VRAM: the `-Fast` build is 22.02 GiB and yields
**273,989 KV tokens = 5.35x concurrency at 51,200**. Plain
`unsloth/…-NVFP4` is 24.67 GiB and drops that to 2.16x; `nvidia/…-NVFP4` is
lighter still but ships **no MTP module** and its `modelopt` format forces the
MARLIN MoE backend (~2.5x slower per unsloth). Never set `--moe-backend`
by hand — compressed-tensors auto-selects `FLASHINFER_CUTLASS`.

## `num_speculative_tokens: 5`

`dspark_block_size` is 5 for DeepSeek-V4-Flash (preview *and* 0731), and the
drafter emits a full block per pass regardless of `k` — so `k: 3` computes five
draft tokens and verifies three, discarding two already paid for. Measured
2026-07-31 on 0731, same profile otherwise: mean 56.4 → 60.4 tok/s (no-think),
53.8 → 57.5 (think), tok/step 3.07 → 3.82. Per-position acceptance for the first
three positions is unchanged, confirming positions 3–4 are pure harvest.

The headline *acceptance rate* falls (69% → 56%) because it now averages the two
hardest positions in — that is expected, not a regression; `tok/step` is the
figure that matters. Prose is the one loss (−3.8%): positions 3–4 accept at only
~33–43% on unpredictable text, so the extra verification isn't repaid. Valid `k`
is **≤ 5, or a multiple of 5**; `k: 7` passes the boot guard and then crashes on
first generation with a tensor-shape mismatch.

This applies to the preview checkpoints too — the block size is a property of the
architecture, not the release.
