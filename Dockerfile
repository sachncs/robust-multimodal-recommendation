"""Multi-stage Dockerfile for the morel inference service.

Stage 1 (builder): install full Python toolchain + build deps.
Stage 2 (runtime): slim image with just the runtime deps + the app.
"""

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY pyproject.toml ./
RUN pip install --upgrade pip build && \
    pip install ".[serve]"

COPY morel ./morel
COPY README.md ./README.md
COPY LICENSE ./LICENSE
RUN python -m build

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOREL_DATA_DIR=/data

RUN groupadd --system morel && \
    useradd --system --gid morel --uid 1000 --create-home --home-dir /home/morel morel

WORKDIR /app

COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

USER morel

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=3).read()" || exit 1

ENTRYPOINT ["python", "-m", "morel"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
