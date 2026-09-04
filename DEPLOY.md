# Deploying airo_agent

Agent-facing deployment guide. Read top-to-bottom before deploying.

`airo_agent` runs **on each GPU serving host**, co-located with `llama-server`,
the GPU drivers, and the HF model cache — because it *spawns* the engine as a
host child process, binds the host's slot ports, and scans the host's model
cache. So this is a **fleet** deploy (one instance per host), built as a
self-contained OTP release and pushed over SSH. It is **not** a single-VM or
container deploy like airo.

## TL;DR

```bash
git commit -am "…"          # the build is exactly committed HEAD
bin/deploy.sh               # build once, roll out to every deploy/hosts/*.env host
bin/deploy.sh gpu-01        # …or just one host (ssh alias)
```

Builds a prod release inside a glibc-matched container, then rolls it out **one
host at a time** with a health gate between hosts. Exit 0 = every targeted host
is healthy.

## The one thing that makes this different: a deploy is a host drain

Each `llama-server` is a `MuonTrap.Daemon` child of the agent's BEAM, so when
the agent stops, **muontrap reaps every engine it spawned**. Restarting the
agent therefore drops that host to **zero resident models** until Airo's
placement policy re-fills it. The agent does not re-adopt engines on restart
(by design — it manages only engines it spawned).

Consequences, baked into the tooling:

- **There is no zero-downtime deploy** without changing the engine-lifetime
  model. A deploy is a brief serving drain on one host, not a hot-swap.
- `bin/deploy.sh` rolls **one host at a time** and waits for `/health` before
  moving on, so aggregate fleet capacity degrades gracefully and Airo can shift
  load to the other hosts.
- The blast radius is *serving availability on one host for ~a model reload*,
  not data loss (there's no DB; the agent is pure control plane).

## Architecture

- Release lives at `/opt/airo-agent/`, **versioned with a symlink**:
  - `releases/airo_agent-<stamp>-<sha>/` — each deploy extracts here, untouched.
  - `current -> releases/airo_agent-<stamp>-<sha>` — flipped **atomically**
    (`ln` to a temp name, then `mv -T`). The running release is never destroyed
    by a deploy, and rollback is a symlink flip (`bin/rollback.sh`).
- Run by systemd **system** unit `airo-agent.service`
  (`/opt/airo-agent/current/bin/airo_agent start`) as user `airo-agent` (in
  `video`/`render` for `nvidia-smi`).
- Runtime env from **`/etc/airo-agent.env`** (0640, root:airo-agent) — see below.
  This is **per host**.
- The release **bundles ERTS**, so serving hosts need no Erlang/Elixir installed.
- No database, no assets, no Traefik. Airo reaches the control API directly on
  the LAN at `http://<advertise-host>:<port>`.

## Per-host config (`/etc/airo-agent.env`)

Unlike airo (one env file), **each host's config differs** — advertise address,
slot count (VRAM-dependent), host id, engine paths. These live as flat files in
`deploy/hosts/<ssh-alias>.env`; `bin/deploy.sh` installs each on its host as
`/etc/airo-agent.env`. The filename stem **is** the SSH alias.

Real host files carry `AIRO_AGENT_TOKEN` and are **gitignored** — only
`deploy/hosts/example.env` is tracked. Copy it to start a new host:

```bash
cp deploy/hosts/example.env deploy/hosts/gpu-01.env   # edit, then deploy
```

| Var | Per-host? | Purpose |
|---|---|---|
| `AIRO_AGENT_ADVERTISE_HOST` | ✅ | LAN IP/FQDN Airo reaches this host + its engines at. Non-loopback ⇒ exposes on `0.0.0.0`. |
| `AIRO_AGENT_SLOTS` | ✅ | Serving slot ports (CSV); one resident model each. Size to VRAM. |
| `AIRO_AGENT_HOST_ID` | ✅ | Stable id Airo keys the host's agent on. |
| `AIRO_AGENT_PORT` | ✅ | Control API port (default 4400; `bin/deploy.sh` reads it for the health gate). |
| `AIRO_AGENT_MODEL_ROOT` | ✅ | Where to scan for GGUF models. |
| `AIRO_AGENT_TOKEN` | maybe | Bearer auth for control API + channel join. Secret. |
| `AIRO_SOCKET_URL` | shared | The controller Airo's `/agent` socket; unset ⇒ log-only. |
| `AIRO_OBSERVER_SOCKET_URLS` | ✅ | CSV of observer Airo sockets (e.g. a dev airo on the workstation). Observers see everything, command nothing. |
| `LLAMA_SERVER_BIN` / `LLAMA_CPP_LIB` | ✅ | Engine binary + its shared libs on this host. |

## Build: per-arch container (the fleet is multi-arch)

A Mix release bundles ERTS (`beam.smp`), which links the **build host's** glibc
*and is arch-specific*. The fleet has both x86 (jobycorp) and **arm64** (DGX
Spark / sparky), so each host's release is built **for its own arch** inside a
container, then shipped. `bin/deploy.sh` does this automatically: it resolves
each host's platform (`PLATFORM=…` in its env file, else `ssh <host> uname -m`)
and builds per-arch, caching so two hosts of the same arch share one build.

`bin/docker-build/Dockerfile` is the builder — a **precompiled, multi-arch**
`hexpm/elixir` base (no OTP-from-source) + `build-essential`. That's the whole
trick that makes emulated arm64 builds cheap: **airo_agent has zero NIFs and
exactly one tiny C file** (`muontrap.c`); everything else is
architecture-independent BEAM bytecode. So an arm64 build emulated on the x86
build host only compiles that one `cc` + bundles the arm64 ERTS — seconds, not
the multi-hour horror an emulated OTP-from-source build would be.

The base is pinned to `hexpm/elixir:1.19.5-erlang-28.4.3-debian-bookworm-…`
(verified multi-arch). **Why debian-bookworm, not ubuntu-noble:** the bundled
ERTS links the build image's glibc, and bookworm's 2.36 is *older* than every
target (sparky is Ubuntu 24.04 / glibc 2.39), so the release stays
forward-compatible everywhere. hexpm has no noble image for this combo, and
trixie/resolute would link a glibc too new for sparky.

**One-time on the build host** (enables cross-arch emulation):

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
```

```bash
ELIXIR_IMAGE=hexpm/elixir:…  bin/deploy.sh   # re-pin the builder base tag
REBUILD_IMAGE=1 bin/deploy.sh                # after editing the Dockerfile
```

> The Dockerfile's `ELIXIR_IMAGE` is pinned to a verified multi-arch tag. When
> bumping Elixir/OTP, re-pin to another **multi-arch, ≤-target-glibc** tag and
> confirm with `docker manifest inspect <tag>` (look for both `linux/amd64` and
> `linux/arm64`).

## Prerequisites (per host)

1. **SSH alias works with passwordless sudo**: `ssh <alias> sudo -n true` exits 0.
2. **`deploy/hosts/<alias>.env` exists** (copy from `example.env`).
3. **`llama-server` is installed on the host** at `LLAMA_SERVER_BIN`. The agent
   only *spawns* the engine — provisioning llama.cpp + CUDA + the HF model cache
   is **out of band** (host bootstrap), not this deploy's job.
4. Docker on the build machine, with binfmt registered (see Build) for any
   cross-arch host.

First deploy to a host auto-creates the `airo-agent` system user, the
`/opt/airo-agent` tree, and installs+enables the systemd unit.

## What `bin/deploy.sh` does

1. Resolves the host list (args, else every `deploy/hosts/*.env` but `example`).
2. `git archive HEAD` → clean source staged once.
3. **Per host, sequentially:** resolve the host's arch → build (or reuse a
   cached) per-arch release inside the container (`mix deps.get/compile` +
   `mix release` → `airo_agent-<stamp>-<sha>-<arch>.tar.gz`) → scp tarball +
   unit + env → extract to `releases/<release>/` → install `/etc/airo-agent.env`
   + the unit → flip `current` atomically → `systemctl restart` → **poll
   `/health` (≤60s)** → prune old releases (keep `KEEP_RELEASES`, default 5) →
   next host. A host that fails the health gate stops the rollout (later hosts
   untouched).

### Flags

```bash
bin/deploy.sh [hosts…]            # default: all deploy/hosts/*.env
REBUILD_IMAGE=1 bin/deploy.sh     # rebuild the per-arch builder image(s)
ELIXIR_IMAGE=… bin/deploy.sh      # override the builder base image tag
SKIP_ENV=1 bin/deploy.sh          # ship release only, leave /etc/airo-agent.env
KEEP_RELEASES=5 bin/deploy.sh     # retained old release dirs per host
HEALTH_TIMEOUT=60 bin/deploy.sh   # seconds to wait for /health
SETTLE=30 bin/deploy.sh           # pause after each host is green (let Airo re-place)
```

## Verification

```bash
ssh gpu-01 'systemctl is-active airo-agent.service'                       # active
ssh gpu-01 'curl -s -o /dev/null -w "%{http_code}\n" localhost:4400/health'  # 200
ssh gpu-01 'curl -s localhost:4400/slots | head'                          # slots back up
ssh gpu-01 'ls -l /opt/airo-agent/current'                                # -> releases/<release>
```

## Rollback

Instant — the previous release is still on disk:

```bash
bin/rollback.sh gpu-01            # flip to the previous release + restart
bin/rollback.sh gpu-01 --list    # show available releases (current marked *)
bin/rollback.sh gpu-01 airo_agent-20260623-120000-abc1234   # to a specific one
```

(Rollback also restarts → re-drains that host's engines, same as a deploy.)

## Common failure modes

- **`beam.smp: ... GLIBC_x.yz not found`** on a host — the `BUILD_IMAGE` is
  newer than that host's distro. Rebuild with a matching `BUILD_IMAGE`
  (`REBUILD_IMAGE=1`).
- **`/health` never passes** — the release didn't boot. Check
  `ssh <host> journalctl -u airo-agent -n50`. The bad release is still under
  `releases/`; `current` was flipped, so `bin/rollback.sh <host>` reverts.
  (Common cause: `/etc/airo-agent.env` missing a required var, or
  `LLAMA_SERVER_BIN` wrong so engines can't launch — though the agent itself
  still serves `/health`.)
- **Engines didn't come back after deploy** — expected briefly; Airo re-places
  models. If they stay down, the engine binary/paths in `/etc/airo-agent.env`
  are wrong for that host, or VRAM is exhausted.
- **`sudo: a password is required`** — the alias lacks passwordless sudo.

## What you must NOT do

- Don't deploy the whole fleet in parallel — it drains every host at once. The
  one-at-a-time roll is the point.
- Don't edit `/opt/airo-agent/current/` in place — always deploy a tarball.
- Don't commit `deploy/hosts/*.env` (tokens) or `*.tar.gz`.
- Don't expect the deploy to install `llama-server` — that's host bootstrap.
