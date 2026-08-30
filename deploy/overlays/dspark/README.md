# DSpark overlay files — frozen copy of the known-good config

Vendored 2026-08-29 from `sparky:/home/jody/airo-agent-overlays/` — the exact
files mounted into the live slot container (`airo-slot-8081`) serving
DeepSeek-V4-Flash-0731:fp8 across sparky+sparky2. The payload
(`deploy/payloads/dspark-0731.json`) references them by their **host** paths;
this directory is the backup of record, not what the agent reads. To restore a
host, copy these to `/home/jody/airo-agent-overlays/` on sparky.

Serving image, pinned by digest (payload pins the tag, tags can move):

    ghcr.io/anemll/dspark-vllm-gx10:0.1.1
    @sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8

What each overlay does (diffs are against that image):

- `flashmla_sparse.py` — nvfp4_ds_mla dispatch fix (MiaAI #22, commit
  56e4ca9): widens the fast sparse-MLA kernel gate to accept `nvfp4_ds_mla`
  alongside `fp8_ds_mla`, avoiding a ~16x long-context decode collapse.
- `scheduler.py` — decode-starvation admission gate (issue #27, commit
  58a1242): 15-line `[issue27-hotfix]` enforcing `max_num_partial_prefills`
  on admission, which the image's v1 scheduler otherwise never reads.
- `deepseek_v4_encoding.py` — tokenizer encoding mounted via the payload's
  `encoding_file` (commit bdcf316).
