#!/usr/bin/env bash
# Release helper for asqi-engineer.
# Usage: ./scripts/release.sh <version>
# Example: ./scripts/release.sh 0.5.0
set -euo pipefail

VERSION="${1:-}"

if [[ -z "$VERSION" ]]; then
  CURRENT=$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
  echo "Current version: ${CURRENT}"
  echo "Usage: $0 <version>"
  exit 1
fi

# Bump version in pyproject.toml
python3 -c "
import re, pathlib
p = pathlib.Path('pyproject.toml')
p.write_text(re.sub(r'^version = \".*\"', 'version = \"${VERSION}\"', p.read_text(), count=1, flags=re.MULTILINE))
"
echo "Bumped pyproject.toml → ${VERSION}"

# Regenerate lockfile to reflect version change
uv lock
echo "Updated uv.lock"

# Prepend changelog
if [[ ! -f CHANGELOG.md ]]; then
  echo "# Changelog" > CHANGELOG.md
fi
git-cliff --unreleased --tag "v${VERSION}" --prepend CHANGELOG.md
echo "Updated CHANGELOG.md"

# Commit, tag, push
git add pyproject.toml uv.lock CHANGELOG.md
git commit -m "chore(release): asqi-engineer ${VERSION}"
git tag "v${VERSION}"
git push origin HEAD "v${VERSION}"
