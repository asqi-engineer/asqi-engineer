#!/usr/bin/env bash
# Release helper: bumps version in pyproject.toml and prepends changelog.
# Usage: ./scripts/release.sh [version]
# Example: ./scripts/release.sh 0.5.0
# If version is omitted, git-cliff suggests a version which you can confirm or override.
set -euo pipefail

VERSION="${1:-}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
PYPROJECT="${REPO_ROOT}/pyproject.toml"
CHANGELOG="${REPO_ROOT}/CHANGELOG.md"
CLIFF_CONFIG="${REPO_ROOT}/cliff.toml"

CURRENT=$(python3 -c "import tomllib; print(tomllib.load(open('${PYPROJECT}','rb'))['project']['version'])")

# Auto-detect recommended version from conventional commits if not provided
if [[ -z "$VERSION" ]]; then
  RAW_VERSION=$(git-cliff --config "${CLIFF_CONFIG}" --bumped-version 2>/dev/null || true)
  DETECTED="${RAW_VERSION#v}"
  if [[ -z "$DETECTED" ]]; then
    echo "Could not auto-detect recommended version."
  fi
  echo "Current version:     ${CURRENT}"
  echo "Recommended version: ${DETECTED:-<unknown>}"
  read -rp "Version to release [${DETECTED}]: " INPUT_VERSION
  VERSION="${INPUT_VERSION:-$DETECTED}"
  if [[ -z "$VERSION" ]]; then
    echo "Error: no version provided."
    exit 1
  fi
fi

# Bump version in pyproject.toml
python3 -c "
import re, pathlib
p = pathlib.Path('${PYPROJECT}')
p.write_text(re.sub(r'^version = \".*\"', 'version = \"${VERSION}\"', p.read_text(), count=1, flags=re.MULTILINE))
"
echo "Bumped pyproject.toml: ${CURRENT} → ${VERSION}"

# Prepend changelog
if [[ ! -f "${CHANGELOG}" ]]; then
  echo "# Changelog" > "${CHANGELOG}"
  echo "Created ${CHANGELOG}"
fi

git-cliff --config "${CLIFF_CONFIG}" \
  --unreleased \
  --tag "v${VERSION}" \
  --prepend "${CHANGELOG}"

echo "Prepended ${VERSION} changelog to CHANGELOG.md"
echo ""
echo "Next steps:"
echo "  1. Review and edit CHANGELOG.md"
echo "  2. Commit: git add pyproject.toml CHANGELOG.md && git commit -m \"chore(release): asqi-engineer ${VERSION}\""
