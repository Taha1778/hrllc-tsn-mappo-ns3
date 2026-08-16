#!/usr/bin/env bash
set -euo pipefail

# Native-Linux equivalent of setup_ns3_wsl.ps1. It is intentionally local to
# one clone so cluster workers never share a mutable ns-3 build directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS3_BRANCH="${NS3_BRANCH:-ns-3.44}"
NS3_DIR="$PROJECT_ROOT/external/$NS3_BRANCH"
SOURCE="$PROJECT_ROOT/ns3/scratch/hrllc_tsn_mappo.cc"

for command in git cmake g++ python3; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "Missing required command: $command" >&2
    echo "Install the Linux build prerequisites, then retry." >&2
    exit 1
  }
done

mkdir -p "$PROJECT_ROOT/external"
if [[ ! -d "$NS3_DIR/.git" ]]; then
  git clone --depth 1 --branch "$NS3_BRANCH" https://gitlab.com/nsnam/ns-3-dev.git "$NS3_DIR"
fi

install -m 0644 "$SOURCE" "$NS3_DIR/scratch/hrllc_tsn_mappo.cc"
cd "$NS3_DIR"
./ns3 configure --enable-modules=core --disable-python --disable-tests --disable-examples
./ns3 build hrllc_tsn_mappo
echo "Native Linux ns-3 setup complete: $NS3_DIR"
