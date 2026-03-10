#!/usr/bin/env bash
set -euo pipefail

# prepare_asqi_pkg.sh
# Prepares asqi_pkg/ inside a container directory for Docker builds.
# Containers that depend on the asqi package reference asqi_pkg/ in their
# Dockerfile. This script copies src/asqi from the repo root into the
# container's build context so it can be installed as a local dependency.
#
# Usage:
#   scripts/prepare_asqi_pkg.sh <container_dir>
# Examples:
#   scripts/prepare_asqi_pkg.sh test_containers/hf_vision_tester
#   scripts/prepare_asqi_pkg.sh sdg_containers/my_generator


CONTAINER_DIR="${1:?Usage: $0 <container_dir>}"

if [[ ! -f "$CONTAINER_DIR/Dockerfile" ]]; then
  echo "No Dockerfile found in $CONTAINER_DIR, skipping."
  exit 0
fi

if ! grep -q 'asqi_pkg/' "$CONTAINER_DIR/Dockerfile" 2>/dev/null; then
  echo "Dockerfile in $CONTAINER_DIR does not reference asqi_pkg/, skipping."
  exit 0
fi

mkdir -p "$CONTAINER_DIR/asqi_pkg/src"
cp -r src/asqi "$CONTAINER_DIR/asqi_pkg/src/asqi"

cat > "$CONTAINER_DIR/asqi_pkg/pyproject.toml" << 'PYPROJECT'
[project]
name = "asqi-engineer"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = ["datasets[vision]>=4.4.1","pydantic>=2.11.7","pyyaml>=6.0.2"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/asqi"]
PYPROJECT

echo "Prepared asqi_pkg for $(basename "$CONTAINER_DIR")"
