#!/usr/bin/env bash
set -eu -o pipefail

# build_test_containers.sh
# Builds all Docker images found under test_containers/* where a Dockerfile is present.
# Image names follow the pattern: <repository>:<folder>-<tag>
# Defaults: repository="my-user/my-project", tag="latest"
#
# Usage:
#   scripts/build_test_containers.sh [-r REPO] [-t TAG]
# Examples:
#   scripts/build_test_containers.sh                      # builds my-user/my-project:<name>-latest
#   scripts/build_test_containers.sh -t v0.1.0            # builds my-user/my-project:<name>-v0.1.0
#   scripts/build_test_containers.sh -r myrepo/one -t v1  # builds myrepo/one:<name>-v1

REPO="asqiengineer/test-container"
TAG="latest"

print_help() {
  cat <<EOF
Build all Docker images under test_containers/

Options:
  -r, --repository  Docker repository to use (default:
                  ${REPO})
  -t, --tag       Version/tag suffix to use (default: ${TAG})
  -h, --help      Show this help and exit

The resulting image names will be: <repository>:<folder>-<tag>
For example, folder "garak/" becomes:
  ${REPO}:garak-latest (default)
  ${REPO}:garak-v0.1.0 (with -t v0.1.0)
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--repository)
      REPO="$2"; shift 2 ;;
    -t|--tag)
      TAG="$2"; shift 2 ;;
    -h|--help)
      print_help; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      print_help
      exit 1 ;;
  esac
done

# Check dependencies
if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not in PATH" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
CONTAINERS_DIR="${ROOT_DIR}/test_containers"

if [[ ! -d "${CONTAINERS_DIR}" ]]; then
  echo "Error: test_containers directory not found at ${CONTAINERS_DIR}" >&2
  exit 1
fi

shopt -s nullglob

BUILD_COUNT=0
SKIP_COUNT=0
ERROR_COUNT=0

for dir in "${CONTAINERS_DIR}"/*/ ; do
  # Get folder name without trailing slash
  name="$(basename "${dir%/}")"
  dockerfile_path="${dir}Dockerfile"
  if [[ ! -f "${dockerfile_path}" ]]; then
    echo "[SKIP] ${name}: No Dockerfile found at ${dockerfile_path}"
    (( SKIP_COUNT = SKIP_COUNT + 1 ))
    continue
  fi

  image_tag="${REPO}:${name}-${TAG}"
  echo "[BUILD] ${name} -> ${image_tag}"
  echo "        Context: ${dir}"

  if docker build -t "${image_tag}" "${dir}" ; then
    echo "[OK] Built ${image_tag}"
    (( BUILD_COUNT = BUILD_COUNT + 1 ))
  else
    echo "[ERR] Failed to build ${image_tag}" >&2
    (( ERROR_COUNT = ERROR_COUNT + 1 ))
  fi
  echo

done

shopt -u nullglob

echo "Summary: built=${BUILD_COUNT} skipped=${SKIP_COUNT} errors=${ERROR_COUNT}"

if [[ ${ERROR_COUNT} -gt 0 ]]; then
  exit 1
fi
