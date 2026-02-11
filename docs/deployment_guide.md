# Deployment Guide

## Prerequisites

- Docker + Docker Compose
- Kubernetes cluster (on-prem) with GPU nodes for inference
- `kubectl` and `kustomize`

## Local Development (Docker Compose)

1. Copy env template:
   - `config/.env.example` -> `.env`
2. Start stack:
   - `docker compose up --build`
3. Access services:
   - Gateway: `http://localhost:8000`
   - MLflow: `http://localhost:5000`
   - Grafana: `http://localhost:3000`

## Kubernetes (On-Prem)

1. Create namespace and secrets:
   - `kubectl apply -f infra/k8s/base/namespace.yaml`
   - `kubectl apply -f infra/k8s/base/secrets.yaml`
2. Deploy base services:
   - `kubectl apply -k infra/k8s/base`
3. Apply production overlay:
   - `kubectl apply -k infra/k8s/overlays/prod`

## Configuration Management

- All configs are provided via environment variables.
- Sensitive values are stored in K8s secrets (or external secret manager).

## Horizontal Scaling

- HPA enabled for Gateway and Batch Worker.
- Inference service scales based on GPU utilization.

## Rolling Updates

- Rolling update strategy configured for all deployments.
- Readiness and liveness probes in place.

## GPU Scheduling

- Inference deployment uses `nvidia.com/gpu: 1` resource requests.
- Node selectors and tolerations for GPU nodes.

## Storage

- MinIO for artifact storage (use external S3 if available).
- PostgreSQL for metadata (use managed Postgres if available).

## Troubleshooting

- Check service logs via `kubectl logs`.
- Validate health endpoints: `/healthz` and `/readyz`.
- Inspect metrics in Grafana for saturation or error spikes.
