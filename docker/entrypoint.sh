#!/bin/sh
set -e

provider="${CV_PROVIDER:-local}"
provider="$(printf '%s' "$provider" | tr '[:upper:]' '[:lower:]')"

if [ "$provider" != "local" ]; then
  if [ ! -x /venv/bin/python ]; then
    echo "ERROR: CV_PROVIDER=$provider requires the runtime-cv Docker image." >&2
    echo "Your container was built with the slim runtime target (no /venv, no baked models)." >&2
    echo "Fix: COMPOSE_PARALLEL_LIMIT=1 docker compose build api && docker compose up -d api" >&2
    echo "Ensure .env has DOCKER_BUILD_TARGET=runtime-cv and CV_EXTRAS=cv-yolo" >&2
    exit 1
  fi
  export PATH="/venv/bin:$PATH"
fi

exec "$@"
