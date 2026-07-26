# Cluster slots — reporting both ranks of a multi-node load

How the agent reports a tensor-parallel load that spans two hosts, and why it is
shaped this way. Companion to `DESIGN-vllm.md` § multi-node, which covers how
such a load is *launched*; this covers how it is *observed*.

The Airo side (`Airo.Serving`, `Airo.Agents.Ingest`, `Airo.Agents.SlotState`)
consumes what this describes.

## The problem it solves

A two-host tensor-parallel load used to be invisible to Airo on the worker:
`sparky2` reported 118 GB of 124 GB VRAM used and **zero slots**, so Airo
rendered it as a host 95% full with nothing on it. Any scheduler reading that
concludes the box is mysteriously occupied and may try to place work there.

That followed from how such a load is launched. Per `lib/airo_agent/engine/vllm.ex`:

> rank 0 (the API node) launches here; the `vllm-slot` wrapper derives rank 1
> from the SAME argv and runs it on the worker over SSH, so both ranks live and
> die with the one supervised wrapper process. **Fleet/Instance are none the wiser.**

And `sparky2:/etc/airo-agent.env`:

```
# No serving slots — worker-only (see header).
AIRO_AGENT_SLOTS=
```

So rank 1 is a container that `sparky`'s wrapper starts on `sparky2` over SSH.
`sparky2`'s own agent did not start it, does not supervise it, and has no
`Fleet` slot for it. Its agent runs there purely for GPU telemetry and inventory.

## What Airo expects

Every rank reports a slot. Each carries the shared load id plus its own rank.
The peer's slot needs no model of its own:

```json
"resident": {
  "model": "fraserprice/DeepSeek-V4-Flash-Abliterated-DSpark:fp8",
  "deployment_id": "dep-7f3a",
  "tp_rank": 1,
  "tp_size": 2
}
```

Three new fields on the slot push, alongside the existing `resident_model` /
`status` / `ctx` / … . Airo reads them in `Airo.Agents.Ingest.cluster_attrs/1`.

### What Airo handles, so the agent doesn't

- `deployment_id` is carried internally as `cluster_id`. It is a **load** id and
  has nothing to do with `Airo.Config.Deployment`; Airo renames it on the way in
  so the two can't be confused in the payload.
- **Rank > 0 never reconciles a Model.** Airo's `Provenance` keys models as
  `<host>_<model>_<port>`, so an unguarded peer would mint a second canonical
  Model for one logical load. Guarded in `Ingest.apply_slot/3`.
- **Rank > 0 never gets a deployment bound**, and one bound by hand is forced
  down with reason `tp_peer`.
- **Peer failure propagates to the head.** Head health is the worst rank's,
  covering a peer that is `down`, one that is `loading` (head → `unknown`), and
  one that has gone *silent* — membership is counted against the declared
  `tp_size`, because a lost rank leaves no slot state behind.
- `/v1/serving` exposes `resident.cluster`, `slot.serves_api`, and a top-level
  `clusters[]` joining the ranks. `/metrics` exposes `airo_cluster_serving`,
  `airo_cluster_complete`, `airo_cluster_members`, `airo_slot_tp_rank`,
  `airo_slot_serves_api`.

The agent's whole job is to emit the three fields from both hosts, accurately.

## Decision 1 — `tp_size` means **nodes**, not TP degree

The easiest thing here to get wrong, and the current fleet cannot catch it.
There is a regression test pinning it (`nnodes: 2, tensor_parallel_size: 8`).

The launch argv carries both:

```
--tensor-parallel-size 2      # total GPUs across the world
--nnodes 2                    # hosts participating
--node-rank 0                 # this host's index
```

On the DSpark pair these are **both 2**, so wiring `tp_size` to either makes
every test pass. They diverge the moment there is a 2-node × 4-GPU host
(`nnodes: 2`, `tensor_parallel_size: 8`).

`tp_rank`/`tp_size` are **`--node-rank` / `--nnodes`**. Two reasons:

1. The requirement is "both *nodes* report a slot, each carrying its own rank" —
   the unit is a host, not a GPU.
2. Airo divides weights by `tp_size` (`Airo.Agents.Capacity.shard_bytes/2`) to
   charge each host its share. A host holds 1/`nnodes` of the weights. Using the
   TP degree would under-report the footprint 4× on the example above.

## Decision 2 — the container label is the source of truth

The worker agent must learn the cluster identity of a container it did not
start. Options considered:

- *Head reports both slots.* Airo keys slots as `host_id:port` from the pushing
  host, so rank 1 would be attributed to `sparky` — wrong host for VRAM, and it
  needs a channel-contract change to let a slot name a different host. Rejected.
- *Worker infers from argv.* `docker inspect .Args` does carry `--node-rank`, but
  not the cluster id, and parsing argv is brittle. Rejected as the primary path.
- **Worker reads labels stamped by the wrapper.** Chosen. The wrapper is the one
  component spanning both ranks, so it is the natural place to mint identity, and
  a label survives an agent restart on either host because it lives on the
  container rather than in agent memory.

The label namespace was free to take: containers carried only *image* labels
(`org.opencontainers.*`, `ai.vllm.build.*`), nothing from `docker run`.

Label set (both ranks):

```
--label airo.cluster.id=<id>          # dep-7f3a
--label airo.cluster.rank=<0|1>       # --node-rank
--label airo.cluster.size=<n>         # --nnodes
--label airo.cluster.model=<id>       # --served-model-name
--label airo.slot.port=<port>
```

## Decision 3 — a peer slot is discovered, not configured, and never loadable

`sparky2` must **not** simply get `AIRO_AGENT_SLOTS=8081`. That would give
`Fleet` a slot it believes it owns, and `POST /load` would happily try to launch
a model onto a GPU already holding rank 1 — the load would fail late and
confusingly, after evicting nothing and OOMing.

Instead, peer ranks are *discovered* from the container runtime and merged into
`slots/0` as read-only. `load`/`unload` on a port occupied by a foreign rank
returns `{:error, :peer_rank_resident}`.

## Where it lives

| Concern | Code |
| --- | --- |
| Fields on the wire (`deployment_id`/`tp_rank`/`tp_size`) | `slot_info.ex` — `to_payload/1` does the `cluster_id` → `deployment_id` rename; `fleet/event.ex` carries them on transitions |
| Minting the id | `engine/vllm.ex` — `cluster_id/1`, exposed to Fleet via the optional `Engine.cluster_info/2` callback |
| Stamping the labels | `priv/engine/vllm-slot` — `cluster_labels` array, with `airo.cluster.rank` appended per invocation |
| Observing a foreign rank | `peer_ranks.ex` |
| Merging + refusing loads | `fleet.ex` — `slots/0` appends `PeerRanks.slots/0`; `load`/`unload` return `{:error, :peer_rank_resident}` (409) |

`AIRO_AGENT_SLOTS` stays **empty** on a worker: peer ranks are discovered, not
configured. Giving the worker a configured slot would make Fleet believe it owns
the port and let `POST /load` launch onto a GPU already carrying a rank.

## A related hazard this fixed

`reap_orphans/0` used to force-remove *every* local `airo-slot-*` container at
boot. On a worker — which owns no slots but holds the head's rank-1 container —
that meant an agent restart silently killed a live rank and took the whole
cluster down with it. The sweep is now scoped to this host's own configured slot
ports, so a worker reclaims nothing.

## Verifying it

Labels are stamped at `docker run`, so a load that predates this work carries
none and its peer stays undiscovered until the cluster is next reloaded.

```bash
curl -sH "Authorization: Bearer <mgmt-key>" \
  https://airo.local.joby.gg/v1/serving | jq '.clusters'
```

Expect one entry, `complete: true`, `serving: true`, two members — `sparky`
rank 0 `serves_api: true`, `sparky2` rank 1 `serves_api: false`.

Then the propagation that has teeth:

```bash
ssh sparky2 'docker stop airo-slot-8081'
```

The head's deployment goes `:down` — before this, it stayed `:up` and kept
taking traffic through a dead cluster. `airo_cluster_serving` drops to 0.

**Deploy order matters.** Deploying to the head restarts its agent, which reaps
rank 0 (correctly — it owns it) and, via `reap_worker_orphans`, rank 1: the load
dies and needs reloading. Deploying to the worker first is safe.
