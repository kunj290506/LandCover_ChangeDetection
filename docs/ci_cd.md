# CI/CD Integration

## CI

- Linting via `ruff` on each pull request.
- Docker image builds for each microservice.

## CD

- Use your preferred deployment system (Argo CD, Flux, Jenkins).
- Recommended flow:
  1. Build and push images to registry.
  2. Update K8s image tags via GitOps overlay.
  3. Rollout and validate with health checks.

## Rollback

- Revert image tags in overlay.
- Promote previous model version in MLflow registry.
