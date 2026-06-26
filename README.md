# DID Backend API

**DID** — **Duplicate Image Detection** — is a FastAPI backend for environmental and public-concern reporting. It flags likely duplicate submissions using image similarity, GPS proximity, issue category, and time windows, with admin review for uncertain matches.

## Stack

- **API:** FastAPI, Uvicorn
- **Database:** PostgreSQL + PostGIS + pgvector
- **Cache:** Redis (upload rate limits, exact-image hash cache)
- **CV (optional):** local heuristics, CLIP embeddings, YOLOv11 object detection

## Local development

```bash
cp .env.example .env
docker compose up --build
```

| URL | Purpose |
|-----|---------|
| http://localhost:8000/ | Report upload UI |
| http://localhost:8000/docs | OpenAPI docs |
| http://localhost:8000/health | Health check (public) |

Compose runs migrations on startup. Uploads are stored under `./uploads`.

## Duplicate detection

On each new report the service:

1. Checks for an **exact image match** (SHA-256) near the same location and category.
2. Searches **nearby candidates** (PostGIS radius + pgvector embedding similarity).
3. Classifies as **duplicate**, **possible duplicate** (queued for review), or **new**.

Signals include perceptual hash, CLIP/YOLO embeddings, distance, category, and configurable similarity thresholds (`DUPLICATE_*` in `.env`).

## Computer vision

Set `CV_PROVIDER` in `.env`:

| Value | Description | Extra install |
|-------|-------------|---------------|
| `local` | Heuristic color/edge detection + histogram embeddings (default) | none |
| `embedding` | CLIP semantic embeddings + local heuristics for object labels | `pip install -e ".[cv-embedding]"` |
| `yolov11` | YOLOv11 object detection + CLIP embeddings | `pip install -e ".[cv-yolo]"` |

Custom YOLO weights: `CV_YOLO_MODEL=/path/to/weights.pt`.

### Irrelevant image rejection

When `CV_REJECT_IRRELEVANT_IMAGES=true` (default), uploads that do not appear to show an environmental or public concern are rejected with HTTP 400. Detection uses mapped YOLO/heuristic labels; CLIP zero-shot scoring is used when the provider includes embeddings.

| Variable | Default | Purpose |
|----------|---------|---------|
| `CV_REJECT_IRRELEVANT_IMAGES` | `true` | Enable/disable rejection |
| `CV_RELEVANCE_THRESHOLD` | `0.15` | Minimum CLIP concern score |
| `CV_RELEVANCE_MIN_DETECTION_CONFIDENCE` | `0.25` | Minimum object-detection confidence |

### Docker CV image

For Docker, set `CV_EXTRAS` and `DOCKER_BUILD_TARGET` to match your provider, then rebuild:

| `CV_PROVIDER` | `CV_EXTRAS` | `DOCKER_BUILD_TARGET` |
|---------------|-------------|------------------------|
| `local` | *(leave empty)* | *(default `runtime`)* |
| `embedding` | `cv-embedding` | `runtime-cv` |
| `yolov11` | `cv-yolo` | `runtime-cv` |

The `runtime-cv` target prefetches CLIP and YOLO weights into `/var/cache/did-backend-api` at build time and reuses a cached PyTorch layer so torch is not re-downloaded on every rebuild.

```bash
# Example: YOLOv11 + CLIP
CV_PROVIDER=yolov11
CV_EXTRAS=cv-yolo
DOCKER_BUILD_TARGET=runtime-cv
COMPOSE_PARALLEL_LIMIT=1 docker compose build api
docker compose up -d api
```

If models download again at runtime, confirm the container has `/venv/bin/python` (runtime-cv image) and that `DOCKER_BUILD_TARGET=runtime-cv` is set — compose reads this name exactly.

If the CV build fails with **cannot allocate memory**, increase Docker Desktop memory (8 GB+ recommended) and use `COMPOSE_PARALLEL_LIMIT=1`.

## Security

All routes except `/` and `/health` require the client IP to be in `ALLOWED_IPS` (supports CIDR ranges).

When exposing the API via **ngrok** or another tunnel on the same host, keep `TRUSTED_PROXY_IPS` to loopback only (`127.0.0.1,::1`). Docker bridge IPs (`172.16.0.0/12`) belong in `ALLOWED_IPS`, not `TRUSTED_PROXY_IPS`, so ngrok's `X-Forwarded-For` header does not bypass the whitelist.

Upload rate limiting: `UPLOAD_RATE_LIMIT_PER_MINUTE` (Redis-backed).

## Tests

Unit tests (no database required):

```bash
pytest -m "not integration"
```

Integration tests (Postgres with PostGIS and pgvector):

```bash
docker compose up -d postgres
pytest -m integration
```

## Migrations

```bash
docker compose exec api alembic upgrade head
```

The compose Postgres image includes PostGIS and pgvector.
