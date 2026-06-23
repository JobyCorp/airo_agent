#!/usr/bin/env bash
# Build an airo_agent prod release in a glibc-matched container and roll it out
# to GPU serving hosts over SSH — ONE HOST AT A TIME.
#
# Why one at a time: each llama-server is a muontrap child of the agent's BEAM,
# so restarting the agent drains every resident model on that host until Airo
# re-places them. Rolling sequentially with a health gate keeps fleet capacity
# degrading gracefully instead of dropping every host at once.
#
# Layout on each host (versioned + symlink — instant rollback, never destroys
# the running release):
#   /opt/airo-agent/releases/airo_agent-<stamp>-<sha>/   each deploy lands here
#   /opt/airo-agent/current -> releases/airo_agent-<stamp>-<sha>   (atomic flip)
#
# The fleet is MULTI-ARCH: x86 boxes (jobycorp) + arm64 (DGX Spark / sparky).
# Each host's release is built for its own arch via `docker build --platform`
# (arm64 emulated on an x86 build host through binfmt/qemu — cheap here, since
# airo_agent has no NIFs and one tiny C file). Builds are cached per-arch, so two
# arm hosts share one build. Each host's platform comes from PLATFORM=… in its
# deploy/hosts/<host>.env, else `ssh <host> uname -m`.
#
# One-time on the build host: register binfmt for cross-arch emulation:
#   docker run --privileged --rm tonistiigi/binfmt --install arm64
#
# Usage:
#   bin/deploy.sh                       # deploy to every deploy/hosts/*.env host
#   bin/deploy.sh gpu-01 sparky         # deploy to just these (ssh aliases)
#   REBUILD_IMAGE=1 bin/deploy.sh       # force-rebuild the builder image(s)
#   ELIXIR_IMAGE=… bin/deploy.sh        # override the builder base image tag
#   SKIP_ENV=1 bin/deploy.sh            # don't push /etc/airo-agent.env (release only)
#   KEEP_RELEASES=5 bin/deploy.sh       # how many old release dirs to retain
#
# Each host's SSH alias == the stem of its deploy/hosts/<alias>.env file, which
# is installed remotely as /etc/airo-agent.env. Passwordless sudo required.

set -euo pipefail

APP="airo_agent"
REMOTE_DIR="${REMOTE_DIR:-/opt/airo-agent}"
REMOTE_TMP="${REMOTE_TMP:-/tmp}"
SERVICE="${SERVICE:-airo-agent.service}"
RUN_USER="${RUN_USER:-airo-agent}"
KEEP_RELEASES="${KEEP_RELEASES:-5}"
# Builder base image (precompiled Elixir/OTP, multi-arch). Empty = the Dockerfile
# default; override to re-pin without editing the Dockerfile.
ELIXIR_IMAGE="${ELIXIR_IMAGE:-}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-60}"   # seconds to wait for /health after restart
SETTLE="${SETTLE:-0}"                    # extra pause after a host is green, before the next

cd "$(dirname -- "$0")/.."
[ -f mix.exs ] || { echo "error: run from the project root" >&2; exit 1; }
command -v docker >/dev/null || { echo "error: docker not found" >&2; exit 1; }

# Hosts to deploy: explicit args, else every deploy/hosts/*.env except the example.
if [ "$#" -gt 0 ]; then
  HOSTS=("$@")
else
  HOSTS=()
  for f in deploy/hosts/*.env; do
    [ -e "$f" ] || continue
    stem="$(basename "$f" .env)"
    [ "$stem" = "example" ] && continue
    HOSTS+=("$stem")
  done
fi
[ "${#HOSTS[@]}" -gt 0 ] || { echo "error: no hosts (pass aliases or add deploy/hosts/<alias>.env)" >&2; exit 1; }

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
# date is host-side (not in the release), so a stamp here is fine.
STAMP="$(date +%Y%m%d-%H%M%S)"
RELEASE="${APP}-${STAMP}-${GIT_SHA}"   # on-host dir name (arch-agnostic)

echo "▸ Releasing ${RELEASE} to: ${HOSTS[*]}"

SRC_DIR="$(mktemp -d)"
OUT_DIR="$(mktemp -d)"
trap 'rm -rf "${SRC_DIR}" "${OUT_DIR}"' EXIT

echo "▸ Staging clean source from HEAD (${GIT_SHA})…"
# Exactly committed HEAD — commit before deploying or your change won't ship.
git archive HEAD | tar -x -C "${SRC_DIR}"

# arch token (amd64|arm64) for a host: PLATFORM=… in its env file, else uname -m.
platform_for() {
  local host="$1" envfile="deploy/hosts/$1.env" p=""
  [ -f "$envfile" ] && p="$(grep -E '^PLATFORM=' "$envfile" 2>/dev/null | tail -1 | cut -d= -f2)"
  if [ -z "$p" ]; then
    case "$(ssh "$host" uname -m 2>/dev/null)" in
      aarch64|arm64) p="linux/arm64" ;;
      x86_64|amd64)  p="linux/amd64" ;;
      *) echo "error: can't determine arch for ${host}; set PLATFORM= in ${envfile}" >&2; return 1 ;;
    esac
  fi
  echo "$p"
}

# Build the release for one platform, once. Memoized: the tarball path is
# deterministic per arch, so a second host of the same arch reuses it.
build_for() {
  local platform="$1" arch="${1##*/}"
  local tag="airo-agent-builder:${arch}"
  local tarball="${APP}-${STAMP}-${GIT_SHA}-${arch}.tar.gz"
  [ -f "${OUT_DIR}/${tarball}" ] && { echo "${tarball}"; return 0; }

  if [ "${REBUILD_IMAGE:-0}" = "1" ] || ! docker image inspect "${tag}" >/dev/null 2>&1; then
    echo "▸ Building builder image ${tag} (${platform})…" >&2
    docker build --platform "${platform}" \
      ${ELIXIR_IMAGE:+--build-arg ELIXIR_IMAGE="${ELIXIR_IMAGE}"} \
      -t "${tag}" -f bin/docker-build/Dockerfile bin/docker-build >&2
  fi

  echo "▸ Building ${arch} release (${platform})…" >&2
  docker run --rm --platform "${platform}" \
    -v "${SRC_DIR}:/build" \
    -v "${OUT_DIR}:/out" \
    -e MIX_ENV=prod \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    "${tag}" \
    bash -c '
      set -euo pipefail
      cd /build
      mix deps.get --only prod
      mix deps.compile
      mix release --overwrite
      tar -C _build/prod/rel/airo_agent -czf "/out/'"${tarball}"'" .
      chown -R "${HOST_UID}:${HOST_GID}" /build /out
    ' >&2
  [ -f "${OUT_DIR}/${tarball}" ] || { echo "error: ${arch} build produced no ${tarball}" >&2; return 1; }
  echo "${tarball}"
}

# ── Roll out, one host at a time (build its arch on demand) ───────────────────
for host in "${HOSTS[@]}"; do
  echo
  echo "════ ${host} ════"

  envfile="deploy/hosts/${host}.env"
  if [ "${SKIP_ENV:-0}" != "1" ]; then
    [ -f "$envfile" ] || { echo "error: ${envfile} missing (need it for /etc/airo-agent.env)" >&2; exit 1; }
  fi
  # Port the host's /health is on (for the gate). Default 4400.
  port="$(grep -E '^AIRO_AGENT_PORT=' "$envfile" 2>/dev/null | tail -1 | cut -d= -f2)"
  port="${port:-4400}"

  platform="$(platform_for "${host}")" || exit 1
  TARBALL="$(build_for "${platform}")" || exit 1

  echo "  • shipping ${TARBALL} + unit (${platform})"
  scp -q "${OUT_DIR}/${TARBALL}" "${host}:${REMOTE_TMP}/${TARBALL}"
  scp -q "deploy/airo-agent.service" "${host}:${REMOTE_TMP}/airo-agent.service"
  if [ "${SKIP_ENV:-0}" != "1" ]; then
    echo "  • shipping ${envfile} -> /etc/airo-agent.env"
    scp -q "$envfile" "${host}:${REMOTE_TMP}/airo-agent.env"
  fi

  ssh "${host}" bash -s -- \
    "${REMOTE_TMP}/${TARBALL}" "${REMOTE_DIR}" "${RELEASE}" "${SERVICE}" \
    "${RUN_USER}" "${KEEP_RELEASES}" "${HEALTH_TIMEOUT}" "${port}" "${SKIP_ENV:-0}" <<'REMOTE'
set -euo pipefail
TARBALL="$1"; DIR="$2"; RELEASE="$3"; SERVICE="$4"
RUN_USER="$5"; KEEP="$6"; HEALTH_TIMEOUT="$7"; PORT="$8"; SKIP_ENV="$9"

# First-time host: create the run user (no login, no home churn) + dirs + unit.
if ! id "${RUN_USER}" >/dev/null 2>&1; then
  echo "  • creating system user ${RUN_USER}"
  sudo useradd --system --create-home --shell /usr/sbin/nologin "${RUN_USER}"
  sudo usermod -aG video,render "${RUN_USER}" 2>/dev/null || true
fi
sudo install -d -o "${RUN_USER}" -g "${RUN_USER}" -m 0755 "${DIR}" "${DIR}/releases"

if [ "${SKIP_ENV}" != "1" ]; then
  echo "  • installing /etc/airo-agent.env"
  sudo install -o root -g "${RUN_USER}" -m 0640 "/tmp/airo-agent.env" /etc/airo-agent.env
fi

echo "  • extracting ${RELEASE}"
sudo install -d -o "${RUN_USER}" -g "${RUN_USER}" -m 0755 "${DIR}/releases/${RELEASE}"
sudo tar -xzf "${TARBALL}" -C "${DIR}/releases/${RELEASE}"
sudo chown -R "${RUN_USER}:${RUN_USER}" "${DIR}/releases/${RELEASE}"

# Install/refresh the systemd unit (shipped alongside the release).
if [ -f /tmp/airo-agent.service ]; then
  sudo install -o root -g root -m 0644 /tmp/airo-agent.service /etc/systemd/system/airo-agent.service
  sudo systemctl daemon-reload
fi
sudo systemctl enable "${SERVICE}" >/dev/null 2>&1 || true

# Atomic symlink flip — ln to a temp name then rename over `current`.
echo "  • flipping current -> ${RELEASE}"
sudo ln -sfn "${DIR}/releases/${RELEASE}" "${DIR}/current.tmp"
sudo mv -T "${DIR}/current.tmp" "${DIR}/current"

# Restart drains this host's engines; the gate below waits for it to come back.
echo "  • restarting ${SERVICE} (drains engines on this host)"
sudo systemctl restart "${SERVICE}"

echo "  • waiting for /health (≤${HEALTH_TIMEOUT}s)"
ok=0
for _ in $(seq 1 "${HEALTH_TIMEOUT}"); do
  if curl -sf -o /dev/null "http://127.0.0.1:${PORT}/health"; then ok=1; break; fi
  sleep 1
done
if [ "${ok}" != "1" ]; then
  echo "  ✗ ${SERVICE} did not pass /health — leaving current symlink for inspection" >&2
  sudo systemctl --no-pager --lines=20 status "${SERVICE}" || true
  exit 1
fi
echo "  ✓ healthy"

# Prune old release dirs, keeping the newest ${KEEP} (plus whatever `current` points at).
cur="$(readlink -f "${DIR}/current")"
ls -1dt "${DIR}/releases/"*/ 2>/dev/null | tail -n "+$((KEEP+1))" | while read -r old; do
  [ "$(readlink -f "${old}")" = "${cur}" ] && continue
  echo "  • pruning $(basename "${old}")"
  sudo rm -rf "${old}"
done

rm -f "${TARBALL}" /tmp/airo-agent.env /tmp/airo-agent.service
REMOTE

  echo "  ✓ ${host} done"
  [ "${SETTLE}" -gt 0 ] && { echo "  • settling ${SETTLE}s before next host"; sleep "${SETTLE}"; }
done

echo
echo "✓ Rollout complete (${RELEASE})"
