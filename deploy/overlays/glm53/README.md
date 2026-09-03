# GLM-5.3-Flash EXL3 overlay files

Adaptation of the MiaAI recipe for this kit, driven through airo_agent's
normal `POST /load` path instead of the recipe's `start.sh`:

    https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks  @ eb0469fb
    (image + runtime patches; base adaptation was done @ 79f10b9 — see
    "2026-09-02 update" below for what moved and what was deliberately kept)

The payload is `deploy/payloads/glm53-flash-exl3.json`. These files are
copies of the recipe's `overlay/` runtime patches plus `files/
chat_template.jinja`, staged on BOTH sparks at
`/home/jody/airo-agent-overlays/glm53/` (the vllm-slot wrapper mounts
`overlay_files` sources by host path on both ranks, so the worker needs the
same files at the same path). `boot-shape-warmup.sh` is the recipe's
post-ready warmup — vendored for reference/manual use; it is not part of the
load.

`airo-entry.sh` is ours, and is the one piece of adaptation glue: the recipe
runs `python3 /opt/glm53/patch_*.py` on both ranks before `vllm serve` (the
patches are also baked into the image at build — the runtime re-run is an
idempotent guard that only matters when the mounted copies are newer than the
image; every patch logs "already present" otherwise). The wrapper
has no pre-serve hook, so the payload sets `entrypoint: bash` +
`cmd_prefix: /opt/glm53/airo-entry.sh`, and the container runs
`bash airo-entry.sh serve <snapshot-dir> …` → patches → `exec vllm "$@"`.

Everything else the recipe's start.sh does maps onto existing seams:

- weights: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` pinned at revision
  `25a44fdbf16862a46b7cc9921142c6c81350af2f` (the byte-identical mirror of
  brandonmusic snapshot 5ab363a8), downloaded to sparky's HF cache and
  rsynced to sparky2 — plus `incoai/GLM-5.3-Flash-DFlash2` (snapshot
  `dc77ff1c…`), the k=7 draft model the payload's speculative-config
  references by absolute snapshot path (draft TP=1, kept on rank 0).
- clustering: the recipe's HEAD/WORKER + NCCL pinning is the wrapper's
  cluster mode (`AIRO_VLLM_CLUSTER_*` in `deploy/hosts/sparky.env`); our kit
  uses GID index 5 (`::ffff:192.168.100.x` RoCEv2 entries), not the
  recipe's default 3.
- serve flags/env: recipe `.env` + inner-script args → payload
  `container_env` / `extra_argv`. Deliberate deltas from the recipe:
  `NCCL_IB_ADDR_FAMILY=AF_INET` added (load-bearing for DSpark on this
  fabric), and no triton/tilelang host cache mounts (the wrapper has no
  per-profile mount seam; `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` absorbs
  cold JIT instead — warm restarts re-JIT, which the recipe's cache mounts
  would avoid).

## 2026-09-02 update — prefill/TTFT bundle from recipe @ eb0469fb

Vendored from recipe HEAD `eb0469fb` on top of the 79f10b9 base (the six
original runtime patches are unchanged between the two pins):

- image: `glm53-flash-sm121:eb0469fb` (id `sha256:8002fa1f…`), built locally
  on sparky 2026-09-02 04:34 PDT from `~/glm53-recipe` (recipe checkout at
  `eb0469f`) with its Dockerfile — exllamav3 pinned at `c5d9c657`, all nine
  patches applied and their `test_*.py` run at build — then `docker save` /
  `docker load`ed onto sparky2 (same image id on both). It replaces the pulled
  GHCR `:exl3` (2026-08-28). It is on NO registry: a load on a spark that
  lacks the tag fails at `docker pull`, so rebuild-or-ship before changing the
  payload's `image`.
- `EXL3_FAT_KERNEL=1` (recipe PR #77, E2 fat-expert prefill kernel, ~+20%
  uncached prefill): only meaningful on the rebuilt image above, whose
  exllamav3 has `exl3_fat_gemm`. Engagement receipt from the live boot
  2026-09-02 11:47 (Worker_TP0): `exl3 e2 diag … configured_tier=kernel
  effective_tier=kernel tier_reason=kernel_ok sym_fat_gemm=1 cap=12.1
  cap_ok=1` and `exl3 fat-expert P0: layers=42 fat_layers=42 (100.0%)` on the
  first prefill. If a future boot shows `effective_tier` ≠ `kernel`, the flag
  is silently no-op'ing again.

- `chat_template.jinja`: recipe PR #63 one-liner — emit the
  `Reasoning Effort:` head line unconditionally so thinking on/off shapes
  share the prompt head. Before this, every thinking toggle invalidated the
  entire prefix cache (full re-prefill on agent traffic).
- `patch_kpool_tail_slotmap.py` (recipe #50): K-pool tail slot mapping
  pinned to the one-block circular scratch. Unconditional fix, baked into the
  rebuilt image; the runtime re-run is the idempotent guard described above.
- `patch_spinwait.py` (recipe #96): SpinCondition reader busy-loop window,
  opt-in via `GLM53_SPINWAIT_MS` (payload sets 16 — the recipe kit's frozen
  sweep value on the same 2× GB10 hardware).
- `patch_indexer_workspace.py` (recipe PR #86): sparse-MLA indexer prefill
  workspace right-sizing, opt-in via `GLM53_INDEXER_WORKSPACE=rightsize`
  (payload opts in). Stock sizing is `max_model_len × 40` entries at 132 B
  each; at our 512k ctx cap the live receipt (2026-09-02 11:42, Worker_TP0)
  is `stock 20971520 → 524296 entries, ~2574 MiB reclaimed` per rank (the
  recipe's ~4.5 GiB figure is at its 1M `max_model_len`). Fail-closed;
  `stock` = byte-identical.
- payload `extra_argv` adds `--long-prefill-token-threshold 1024` (native
  vLLM flag, recipe issue #110): caps how much of the per-step token budget
  one chunked prefill may claim, so a long cold prefill no longer freezes
  every other session for its full duration (measured 440 s → ~20 s turns,
  ~+10% on the contended prefill only).

`airo-entry.sh` applies the three new patches after the original six, in
the recipe's start.sh order (kpool_tail → spinwait → indexer_workspace).

Deliberately NOT taken from HEAD (pending separate work):

- MNBT 7168 (new maintainer default at MAX_NUM_SEQS=4): the maintainer chose
  it on the E2 kernel, which we now run, but keep 2048 until we A/B it on
  this kit. Never 8192 (indexer smem).
- `DFLASH_DRAFT_TP=2` (recipe #48, decode-side): we keep draft TP=1.
- `GLM53_MIXED_PREFILL_CHUNK`: recipe default is now `skip`; we keep our
  measured `1024`. Issue #110's controls showed this knob is orthogonal to
  the starvation fix above.
