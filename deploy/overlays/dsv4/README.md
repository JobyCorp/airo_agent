# DeepSeek-V4-Flash-Vision-Exp (DSpark) overlay files

Adaptation of the MiaAI recipe for the Spark pair, driven through
airo_agent's normal `POST /load` path instead of the recipe's compose file:

    https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark  @ f3d7645
    (checked out on sparky at ~/dsv4-recipe; base adaptation was done @ 9923a9b,
    the opt-in gates @ 7440c53, the C128A gate @ f3d7645)

The payloads are `deploy/payloads/dsv4-vision-exp.json` (the recipe's
default shape: 6 seqs, gpu util 0.835, no opt-ins) and
`deploy/payloads/dsv4-vision-exp-longcode.json` (the README's long-coding
shape: 4 seqs, 16k batched tokens, gpu util 0.87, plus the opt-in hotfixes
for tool calls and prefix-cache alignment, and the C128A prefill cache).

Staged on BOTH sparks at `/home/jody/airo-agent-overlays/dsv4/`:

- `airo-entry.sh` — this file, the one piece of adaptation glue. The recipe's
  compose command copies the checkpoint's tokenizer encoding over the
  image's, applies its hotfixes with python3/bash on both ranks, then execs
  `vllm serve`. The vllm-slot wrapper has no pre-serve hook, so the payload
  sets `entrypoint: bash` + `cmd_prefix: /opt/dsv4/airo-entry.sh` and the
  container runs `bash airo-entry.sh serve <snapshot-dir> …` → patches →
  `exec vllm "$@"`. Steps 1–3 are the compose default path; step 4 runs the
  recipe's `DSPARK_ENABLE_*` opt-ins, each gated by the same env the compose
  file reads, in compose order. Keep step 4 in step with the compose file.
- `patches/` — a copy of the recipe checkout's `patches/` tree, mounted as
  `/opt/dspark-patches`. `RECIPE_REV` records the recipe commit it was
  copied from.

Updating to a newer recipe:

    ssh sparky 'cd ~/dsv4-recipe && git pull --ff-only origin main'
    ssh sparky 'rsync -a ~/dsv4-recipe/patches/ /home/jody/airo-agent-overlays/dsv4/patches/'
    ssh sparky 'rsync -a /home/jody/airo-agent-overlays/dsv4/patches/ jody@192.168.100.11:/home/jody/airo-agent-overlays/dsv4/patches/'
    # then diff docker-compose.dspark.yml against step 4 here, add any new
    # DSPARK_ENABLE_* gate, update RECIPE_REV on both hosts, scp this script
    # to both, and re-POST the payload (the container only reads these at boot).

Not mirrored from the compose file: `hotfix-dsv4-runtime-ablation.py`
(inert without `ABLATE=1`, which these payloads never set).
