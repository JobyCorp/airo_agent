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

Both are two-host TP loads (`nnodes: 2`). Reloading is a **cluster** operation:
the head's wrapper starts rank 1 on the worker over SSH, so posting to sparky
alone brings up both ranks. Any agent restart on either Spark drops the load and
it must be reposted. Weights must already exist at the same snapshot path on
both hosts — rsync over the 200G link before a first load.

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
