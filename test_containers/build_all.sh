
#!/usr/bin/env bash
set -euo pipefail

# build_test_containers.sh
# Builds all Docker images found under test_containers/* where a Dockerfile is present.
# Image names follow the pattern: <registry>/<folder>:<tag>
# Defaults: registry="my-registry", tag="latest"
#
# Usage:
#   scripts/build_test_containers.sh [-r REGISTRY] [-t TAG]
# Examples:
#   scripts/build_test_containers.sh                 # builds my-registry/<name>:latest
#   scripts/build_test_containers.sh -r myrepo -t v1 # builds myrepo/<name>:v1

REGISTRY="my-registry"
TAG="latest"

print_help() {
  cat <<EOF
Build all Docker images under test_containers/

Options:
  -r, --registry  Docker registry/namespace to use (default: ${REGISTRY})
  -t, --tag       Docker tag/label to use (default: ${TAG})
  -h, --help      Show this help and exit

The resulting image names will be: <registry>/<folder>:<tag>
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--registry)
      REGISTRY="$2"; shift 2 ;;
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
    ((SKIP_COUNT++))
    continue
  fi

  image_tag="${REGISTRY}/${name}:${TAG}"
  echo "[BUILD] ${name} -> ${image_tag}"
  echo "        Context: ${dir}"

  if docker build -t "${image_tag}" "${dir}" ; then
    echo "[OK] Built ${image_tag}"
    ((BUILD_COUNT++))
  else
    echo "[ERR] Failed to build ${image_tag}" >&2
    ((ERROR_COUNT++))
  fi
  echo

done

shopt -u nullglob

echo "Summary: built=${BUILD_COUNT} skipped=${SKIP_COUNT} errors=${ERROR_COUNT}"

if [[ ${ERROR_COUNT} -gt 0 ]]; then
  exit 1
fi
