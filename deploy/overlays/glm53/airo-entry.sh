#!/bin/bash
# airo-entry.sh — in-container launcher for the GLM-5.3-Flash EXL3 slot.
#
# The MiaAI recipe (GLM-5.3-Flash-EXL3-2x-DGX-Sparks) applies its runtime
# patches with `python3 /opt/glm53/patch_*.py` before exec'ing `vllm serve` on
# BOTH ranks — the patches are also baked into the :exl3 image at build, so
# these re-runs only matter when the mounted copies are newer than the image.
# The vllm-slot wrapper can't run pre-serve hooks, so the payload mounts this
# script via overlay_files and points entrypoint=bash + cmd_prefix at it:
# the container runs `bash /opt/glm53/airo-entry.sh serve <snapshot-dir> …`.
#
# Keep the patch list in step with the recipe's inner scripts (start.sh,
# write_inner_scripts) — a patch applied on one rank only skews the ranks.
set -euo pipefail

for p in \
    patch_glm_video_placeholders \
    patch_suppress_stops_in_reasoning \
    patch_scheduler_decode_floor \
    patch_glm5_drafter_group \
    patch_hybrid_prefix_hit \
    patch_xgrammar_termination; do
    python3 "/opt/glm53/${p}.py"
done

exec vllm "$@"
