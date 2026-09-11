#!/bin/bash
# airo-entry.sh — in-container launcher for the DeepSeek-V4-Flash-Vision-Exp
# DSpark slot on the Anemll 0.1.1 image.
#
# The MiaAI recipe (DeepSeek-v4-Flash-DSpark-2x-DGX-Spark, docker-compose.dspark.yml)
# copies the checkpoint's own tokenizer encoding over the image's, then applies
# its startup hotfixes with python3/bash before exec'ing `vllm serve` on BOTH
# ranks. The vllm-slot wrapper can't run pre-serve hooks, so the payload mounts
# this script plus the recipe's patches/ tree (as /opt/dspark-patches) via
# overlay_files and points entrypoint=bash + cmd_prefix at it: the container
# runs `bash /opt/dsv4/airo-entry.sh serve <snapshot-dir> …`.
#
# Steps 1-3 reproduce the compose command's DEFAULT path (no API keys, TP=2);
# step 4 adds the recipe's opt-in DSPARK_ENABLE_* hotfixes, off unless the
# payload's container_env sets them to 1. Keep the list in step with
# the compose file — a patch applied on one rank only skews the ranks.
set -euo pipefail

export PATH="/usr/local/cuda/bin:/usr/local/bin:${PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-$CUDA_HOME}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
for v in NCCL_IB_MERGE_NICS NCCL_IB_SUBNET_AWARE_ROUTING NCCL_IB_SUBNET_PREFIX_LEN \
         NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ NCCL_DMABUF_ENABLE; do
  [ -z "${!v:-}" ] && unset "$v"
done

P=/opt/dspark-patches
TOK=/usr/local/lib/python3.12/dist-packages/vllm/tokenizers

# 1. Checkpoint encoding over the image's copy (argv is `serve <snapshot-dir> …`).
snapshot="${2:-}"
enc="${snapshot}/encoding/encoding_dsv4.py"
if [ -f "$enc" ]; then
  cp "$enc" "$TOK/deepseek_v4_encoding.py"
  python3 - <<'PY'
from pathlib import Path
p = Path("/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4.py")
s = p.read_text()
old = 'elif reasoning_effort in ("max", "xhigh"):\n                reasoning_effort = "max"\n            else:\n                reasoning_effort = "high"'
new = 'elif reasoning_effort in ("max", "xhigh"):\n                reasoning_effort = "max"\n            elif reasoning_effort == "high":\n                reasoning_effort = "high"\n            else:\n                reasoning_effort = "low"'
u = s.replace(old, new); assert new in u; p.write_text(u)
PY
  python3 "$P/hotfix-encoding-dsv4-issue21.py"
else
  echo "airo-entry: encoding_dsv4.py not found at $enc" >&2
  exit 78
fi
python3 "$P/hotfix-dsv4-issue55-tool-truncation.py"

# 2. Kernel / runtime hotfixes (compose defaults: none of the SKIP flags set).
bash "$P/hotfix-nvfp4-ds-mla-issue22.sh"
bash "$P/hotfix-gb10-spin-wait.sh"
python3 "$P/hotfix-vllm-issue117-shm-ring-buffer.py"
python3 "$P/hotfix-vllm-issue117-shm-ring-buffer.py" --status
for hf in hotfix-dsv4-mtp-buffer-50312.sh hotfix-dsv4-skip-topk-49486.sh \
          hotfix-dsv4-dense-prefill-indexer-48407.sh hotfix-dsv4-skip-empty-c128-48957.sh \
          hotfix-dsv4-flashmla-workspace-50298.sh hotfix-dsv4-grammar-advance.sh; do
  bash "$P/$hf"
done
export VLLM_ENABLE_RESPONSES_API_STORE=0

# 3. Vision-Exp tower + scheduler / decode hotfixes (all default-on in compose).
python3 "$P/hotfix-dsv4-vision-exp.py"
python3 "$P/hotfix-vllm-empty-encoder-output.py"
python3 "$P/hotfix-dsv4-issue27-partial-prefill-concurrency.py"
python3 "$P/hotfix-dsv4-issue43-decode-fairness-and-diag.py"
python3 "$P/hotfix-dsv4-issue26-hybrid-swa-min.py"
python3 "$P/hotfix-dsv4-issue133-triton-specialization.py"
python3 "$P/hotfix-dsv4-suppress-stops-in-reasoning.py"

# 4. Opt-in hotfixes (recipe @ f3d7645 order), each gated by the same env the
#    compose file reads; the payload's container_env turns them on.
on() { [ "${!1:-0}" = "1" ]; }
on DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX      && python3 "$P/hotfix-dsv4-assistant-final-continuation.py"
on DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN       && python3 "$P/hotfix-dsv4-issue144-effort-align.py"
on DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX    && python3 "$P/hotfix-vllm-issue136-xgrammar-termination.py"
on DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED && python3 "$P/hotfix-vllm-issue191-toolcall-failclosed.py"
on DSPARK_ENABLE_DSPARK_BLOCK_K              && python3 "$P/hotfix-vllm-dspark-block-k.py"
on DSPARK_ENABLE_ROPE_SWA_FIX                && python3 "$P/hotfix-vllm-rope-swa-fix.py"
on DSPARK_ENABLE_DSPARK_SWA_PREFIX           && python3 "$P/hotfix-vllm-dspark-swa-prefix.py"
on DSPARK_ENABLE_DSML_RECOVERY               && python3 "$P/hotfix-vllm-dsml-recovery.py"
on DSPARK_ENABLE_MXFP4_INDEXER_CACHE         && python3 "$P/hotfix-vllm-mxfp4-indexer-cache.py"
on DSPARK_ENABLE_C128A_PREFILL_CACHE         && python3 "$P/hotfix-vllm-c128a-prefill-cache.py"
# (--async-scheduling and the MXFP4 --attention-config arg are payload argv, not env.)
# Not mirrored: hotfix-dsv4-runtime-ablation.py. The compose file runs it unconditionally,
# but it is inert without ABLATE=1 and this payload never sets ABLATE, so it is left out.

exec /usr/local/bin/vllm "$@"
