# Builder image for airo_agent prod releases — MULTI-ARCH.
#
# Rebased off a PRECOMPILED Elixir/OTP image (no more kerl/OTP-from-source) so
# cross-arch builds stay cheap. The arm64 target (DGX Spark / GB10) is built via
# qemu emulation on an x86 host, and airo_agent has ZERO NIFs and exactly one
# tiny native C file (deps/muontrap/c_src/muontrap.c). So the only emulated
# native work is that single `cc` plus bundling the target-arch ERTS — seconds,
# not the multi-hour horror an emulated OTP-from-source build would be. The x86
# build gets faster too (no first-time kerl compile).
#
# Built per-arch by bin/deploy.sh via `docker build --platform`:
#   docker build --platform linux/arm64 -t airo-agent-builder:arm64 \
#     -f bin/docker-build/Dockerfile bin/docker-build
#
# Base: a verified multi-arch (amd64+arm64) hexpm/elixir tag, OTP 28.4.3 (matches
# the dev host). Why **debian-bookworm** and not ubuntu-noble: the bundled ERTS
# links the build image's glibc, and bookworm's 2.36 is OLDER than every target
# (sparky is Ubuntu 24.04 / glibc 2.39), so the release runs everywhere —
# forward-compatible. hexpm doesn't publish a noble image for this combo, and
# trixie/resolute would link a glibc too NEW for sparky. Re-pin with
# `docker manifest inspect <tag>`; override per build with --build-arg ELIXIR_IMAGE=…
ARG ELIXIR_IMAGE=hexpm/elixir:1.19.5-erlang-28.4.3-debian-bookworm-20260610
FROM ${ELIXIR_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# build-essential: muontrap compiles c_src/muontrap.c via elixir_make at release
# time. git + ca-certificates: hex/deps fetch. Nothing else is native.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mix local.hex --force \
    && mix local.rebar --force

WORKDIR /build
