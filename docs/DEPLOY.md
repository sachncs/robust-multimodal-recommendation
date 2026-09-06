# morel — Deployment

## Docker

A multi-stage `Dockerfile` produces a slim runtime image:

```bash
docker build -t morel:latest .
docker run -p 8080:8080 morel:latest morel serve --host 0.0.0.0 --port 8080
```

The runtime image runs as a non-root user and includes a `HEALTHCHECK`.

## docker-compose

```bash
docker compose up morel-serve
```

Mounts the local `data/` and `checkpoints/` read-only.

## Environment variables

- `MOREL_DATA_DIR` — root data directory (default `./data`).
- `MOREL_AUTH_TOKEN` — bearer-token for the inference API. When unset, auth
  is disabled.
- `MOREL_AUTH_ENABLED` — set to `1` to require a token even in environments
  that otherwise disable auth.
- `LOG_LEVEL` — log level (default `INFO`).

## Health and metrics

- `GET /health` — JSON `{"status": "ok", "version": "..."}`.
- `GET /metrics` — Prometheus exposition.
- `GET /v1/complete` (POST `{items, modalities}`) — completed modality
  vectors.
- `GET /v1/recommend` (POST `{user, top}`) — ranked items.

## Scaling

The current single-process `uvicorn` worker is suitable for moderate
throughput. For high-throughput deployments:

1. Front the service with a load balancer.
2. Place a sidecar model loader that warms the LRU before traffic.
3. Enable Prometheus scraping on `/metrics`.
4. Pin the model checkpoint via `MOREL_MODEL_PATH`.

## Kubernetes (sketch)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: morel
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: serve
        image: morel:latest
        args: ["serve", "--host", "0.0.0.0", "--port", "8080"]
        readinessProbe:
          httpGet: { path: /health, port: 8080 }
        livenessProbe:
          httpGet: { path: /health, port: 8080 }
          initialDelaySeconds: 30
        env:
        - name: MOREL_AUTH_TOKEN
          valueFrom: { secretKeyRef: { name: morel, key: token } }
```
