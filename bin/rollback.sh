#!/usr/bin/env bash
# Roll a host back to its previous release — instant, no rebuild. The versioned
# layout keeps the last few releases on disk; this just flips the `current`
# symlink and restarts (which re-drains/reloads the engines on that host).
#
# Usage:
#   bin/rollback.sh gpu-01                 # flip to the previous release
#   bin/rollback.sh gpu-01 --list          # show available releases, do nothing
#   bin/rollback.sh gpu-01 airo_agent-20260623-...-<sha>   # flip to a specific one

set -euo pipefail
host="${1:-}"; target="${2:-}"
[ -n "${host}" ] || { echo "usage: bin/rollback.sh <ssh-alias> [--list | <release-dir>]" >&2; exit 1; }

REMOTE_DIR="${REMOTE_DIR:-/opt/airo-agent}"
SERVICE="${SERVICE:-airo-agent.service}"

ssh "${host}" bash -s -- "${REMOTE_DIR}" "${SERVICE}" "${target}" <<'REMOTE'
set -euo pipefail
DIR="$1"; SERVICE="$2"; TARGET="$3"
cur="$(readlink -f "${DIR}/current" 2>/dev/null || true)"

# Newest-first list of release dirs.
mapfile -t rels < <(ls -1dt "${DIR}/releases/"*/ 2>/dev/null | sed 's:/*$::')
[ "${#rels[@]}" -gt 0 ] || { echo "no releases under ${DIR}/releases" >&2; exit 1; }

if [ "${TARGET}" = "--list" ]; then
  for r in "${rels[@]}"; do
    mark=" "; [ "$(readlink -f "$r")" = "${cur}" ] && mark="*"
    echo "${mark} $(basename "$r")"
  done
  exit 0
fi

if [ -n "${TARGET}" ]; then
  dest="${DIR}/releases/${TARGET}"
  [ -d "${dest}" ] || { echo "no such release: ${TARGET}" >&2; exit 1; }
else
  # Previous = first release dir that isn't the current one.
  dest=""
  for r in "${rels[@]}"; do
    [ "$(readlink -f "$r")" = "${cur}" ] && continue
    dest="$r"; break
  done
  [ -n "${dest}" ] || { echo "no previous release to roll back to" >&2; exit 1; }
fi

echo "rolling back to $(basename "${dest}") (restart drains engines on this host)"
sudo ln -sfn "${dest}" "${DIR}/current.tmp"
sudo mv -T "${DIR}/current.tmp" "${DIR}/current"
sudo systemctl restart "${SERVICE}"
sudo systemctl --no-pager --lines=10 status "${SERVICE}" || true
REMOTE
