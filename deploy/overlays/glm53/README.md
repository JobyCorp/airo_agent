# GLM-5.3-Flash EXL3 overlay files

Adaptation of the MiaAI recipe for this kit, driven through airo_agent's
normal `POST /load` path instead of the recipe's `start.sh`:

    https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks  @ 79f10b9

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
patches are also baked into the `:exl3` image at build — the runtime re-run
only matters when the mounted copies are newer than the image). The wrapper
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
