# syntax=docker/dockerfile:1

# One apt pass for build tools (shared by slim + CV builders).
FROM python:3.12-slim AS build-deps

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Cached layer: PyTorch CPU wheels (~1GB download once, reused on rebuild).
FROM build-deps AS cv-base

ENV PIP_DEFAULT_TIMEOUT=600
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

FROM build-deps AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app

ENV PIP_DEFAULT_TIMEOUT=600
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install .

FROM cv-base AS builder-cv

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app

ARG CV_EXTRAS=
ARG CV_EMBEDDING_MODEL=ViT-B-32
ARG CV_EMBEDDING_PRETRAINED=openai
ARG CV_YOLO_MODEL=/var/cache/did-backend-api/yolo11n.pt
ENV PIP_DEFAULT_TIMEOUT=600 \
    CV_EXTRAS=${CV_EXTRAS} \
    CV_EMBEDDING_MODEL=${CV_EMBEDDING_MODEL} \
    CV_EMBEDDING_PRETRAINED=${CV_EMBEDDING_PRETRAINED} \
    CV_YOLO_MODEL=${CV_YOLO_MODEL}
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[${CV_EXTRAS}]"

COPY docker/prefetch_cv_models.py /tmp/prefetch_cv_models.py
RUN python /tmp/prefetch_cv_models.py

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/uploads /var/cache/did-backend-api

COPY --from=builder /install /usr/local
COPY pyproject.toml README.md ./
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY docker/entrypoint.sh /docker/entrypoint.sh
RUN chmod +x /docker/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/docker/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Copy CV artifacts first so this stage waits for builder-cv (avoids parallel apt with build-deps).
FROM python:3.12-slim AS runtime-cv

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
WORKDIR /app

COPY --from=builder-cv /venv /venv
COPY --from=builder-cv /var/cache/did-backend-api /var/cache/did-backend-api

ENV PATH=/venv/bin:$PATH \
    HF_HOME=/var/cache/did-backend-api/huggingface \
    HUGGINGFACE_HUB_CACHE=/var/cache/did-backend-api/huggingface/hub

# Install runtime libs after heavy build stages finish; one package at a time to lower peak RAM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && apt-get install -y --no-install-recommends libgl1 \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get install -y --no-install-recommends libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/uploads /var/cache/did-backend-api

COPY pyproject.toml README.md ./
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY docker/entrypoint.sh /docker/entrypoint.sh
RUN chmod +x /docker/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/docker/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
