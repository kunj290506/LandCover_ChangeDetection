# Operations Runbook

## SLO Targets

- Gateway availability: 99.9%
- Inference p95 latency: < 2s per 256x256 patch
- Batch throughput: > 500 pairs/hour per GPU node

## Daily Checks

- Grafana dashboards for error rate and latency
- MLflow for failed runs or stalled artifacts
- Disk and object storage utilization

## Incident Response

- **High error rate**: Check gateway logs, rate limiting, JWT issues
- **Inference latency spike**: Check GPU utilization and model load
- **Batch backlog**: Scale worker replicas and check Redis

## Rollback Procedure

1. Promote previous model version in MLflow registry.
2. Restart inference pods to reload model version.
3. Verify with smoke tests.

## Disaster Recovery

- Nightly backups of PostgreSQL and MLflow artifacts.
- Object storage replication if available.

## Security Operations

- Rotate JWT secret and API keys quarterly.
- Audit RBAC policies monthly.
- Validate container image signatures.

## Maintenance

- Patch base images monthly.
- Run model health checks weekly.
