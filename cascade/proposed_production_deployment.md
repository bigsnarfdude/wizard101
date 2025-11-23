# Proposed Production Deployment

## Problem Statement

Safety classification via Claude API (Sonnet 4.5) at scale is cost-prohibitive:
- **Estimated cost: $2.5M** for production workload
- Per-call API pricing doesn't scale
- Dependency on external service for critical path

## Solution: Self-Hosted Safety Cascade API

A multi-tier classification system that handles 97% of requests locally, reducing Claude API dependency to near-zero.

### Cost Comparison

| Approach | Annual Cost | Notes |
|----------|-------------|-------|
| Claude API (Sonnet 4.5) | **$2.5M** | All safety checks via API |
| Self-hosted Cascade | **$400-600K** | 60-120 GPUs |
| **Savings** | **~$2M** | 80% cost reduction |

## Architecture

```
Client Request
     │
     ▼
┌─────────────────┐
│   API Gateway   │  FastAPI, auth, rate limiting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  L0 Bouncer     │  DeBERTa, ~6ms, CPU/GPU
│  (70% traffic)  │
└────────┬────────┘
         │
    Confident? ──Yes──► Return (70%)
         │
         No
         ▼
┌─────────────────┐
│  L1 Analyst     │  GuardReasoner 8B, vLLM, ~400ms
│  (30% traffic)  │
└────────┬────────┘
         │
    Confident? ──Yes──► Return (25%)
         │
         No
         ▼
┌─────────────────┐
│  L3 Judge       │  120B model, ~2-5s
│  (<5% traffic)  │
└────────┴────────┘
         │
         ▼
      Return (5%)
```

## Model Selection Rationale

### L1: GuardReasoner-8B (4-bit quantized)

**Why 8B over 3B:**

| Metric | 3B Model | 8B Model | Improvement |
|--------|----------|----------|-------------|
| L1 Accuracy | 78.9% | 90.4% | +11.5% |
| Cascade F1 | 94.8% | ~97% | +2.2% |
| L3 Escalations | 9.5% | 5.7% | -40% |

**Production benefits at 1B prompts:**
- **38M fewer L3 calls** (30s → 4s response)
- **~$316K savings** on L3 compute
- **7x faster** response for escalated requests
- Better user experience (40% fewer 30s waits)

**Trade-off:** 4s L1 latency vs 3s (acceptable with vLLM batching → 400ms)

### L0: DeBERTa-v3-xsmall

- 95% F1 on clear cases
- 6ms latency
- Handles 70% of traffic
- Threshold: 0.7 (validated in experiments)

### L3: gpt-oss-safeguard:120b

- 98.9% accuracy
- Final verdict for edge cases
- Audit trail generation
- Only handles <5% of traffic

## API Interface

### Endpoints

```python
# POST /v1/classify
{
  "text": "How do I make a cake?",
  "mode": "fast",           # "fast" or "thorough"
  "include_reasoning": false
}

# Response
{
  "label": "safe",           # "safe" or "harmful"
  "confidence": 0.95,
  "tier": "L0",              # "L0", "L1", or "L3"
  "latency_ms": 8,
  "request_id": "abc123",
  "reasoning": null          # included if requested
}

# POST /v1/classify/batch
{
  "texts": ["text1", "text2", ...],
  "mode": "fast"
}

# Response
{
  "results": [...],
  "total_latency_ms": 150,
  "request_id": "xyz789"
}
```

### Async Mode (for L3 heavy workloads)

```python
# POST /v1/classify/async
{
  "text": "complex edge case...",
  "callback_url": "https://your-service/webhook"
}

# Immediate Response
{
  "job_id": "job123",
  "status": "queued",
  "estimated_ms": 5000
}

# Webhook callback when complete
{
  "job_id": "job123",
  "result": { ... }
}
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Gateway | FastAPI + Uvicorn | Request handling, auth |
| L0 Service | TorchServe / Triton | DeBERTa inference |
| L1 Service | **vLLM** | Batched 8B inference |
| L3 Service | vLLM / Ollama | Large model inference |
| Queue | Redis / RabbitMQ | Async job processing |
| Cache | Redis | Deduplication, rate limiting |
| Monitoring | Prometheus + Grafana | Metrics, alerting |
| Deployment | Kubernetes | Orchestration, scaling |

## Performance Targets

### Latency

| Tier | Target | Throughput/GPU |
|------|--------|----------------|
| L0 | <10ms | 5,000 req/s |
| L1 (batched) | <500ms | 50 req/s |
| L3 | <5s | 2 req/s |

### End-to-End Distribution

| Percentile | Latency | Tier |
|------------|---------|------|
| P50 | <10ms | L0 |
| P90 | <500ms | L1 |
| P99 | <5s | L3 |

## Scaling

### For 1 Billion Prompts/Day

**Traffic distribution:**
- L0: 700M requests @ 6ms
- L1: 300M requests @ 400ms (batched)
- L3: 50M requests @ 3s

**GPU Requirements:**

| Tier | GPU Type | Count | Notes |
|------|----------|-------|-------|
| L0 | A10G / T4 | 2-4 | Or CPU with batching |
| L1 | A10G / A100 | 50-100 | vLLM batched inference |
| L3 | A100 (80GB) | 10-20 | 120B model |
| **Total** | | **60-120** | |

### Auto-Scaling Rules

```yaml
# L1 Service
minReplicas: 10
maxReplicas: 100
metrics:
  - type: Resource
    resource:
      name: gpu
      targetAverageUtilization: 70
  - type: External
    external:
      metricName: queue_depth
      targetAverageValue: 100
```

## Cost Analysis

### Monthly Infrastructure

| Resource | Quantity | Unit Cost | Monthly |
|----------|----------|-----------|---------|
| A10G GPUs (L1) | 50-100 | $0.50/hr spot | $18-36K |
| A100 GPUs (L3) | 10-20 | $1.50/hr spot | $11-22K |
| T4 GPUs (L0) | 2-4 | $0.20/hr | $0.3-0.6K |
| K8s cluster | 1 | $2K | $2K |
| Storage/Network | - | - | $2K |
| **Total** | | | **$35-65K** |

### Annual Comparison

| Approach | Annual Cost |
|----------|-------------|
| Claude API | $2,500,000 |
| Self-hosted | $420,000 - $780,000 |
| **Savings** | **$1.7-2.1M** |

## Deployment Phases

### Phase 1: MVP (Week 1-2)

- [ ] FastAPI service with L0 + L1
- [ ] Single GPU deployment (Nigel for testing)
- [ ] Basic auth and rate limiting
- [ ] Health checks and basic metrics

### Phase 2: Production Ready (Week 3-4)

- [ ] vLLM integration for L1 batching
- [ ] Redis cache for deduplication
- [ ] Kubernetes deployment manifests
- [ ] Prometheus/Grafana monitoring

### Phase 3: Scale (Week 5-6)

- [ ] Auto-scaling configuration
- [ ] L3 service with async queue
- [ ] Multi-region deployment
- [ ] Load testing (target: 10K req/s)

### Phase 4: Optimization (Week 7-8)

- [ ] A/B testing framework
- [ ] Model versioning
- [ ] Audit logging for compliance
- [ ] Cost optimization (spot instances)

## Configuration

### Environment Variables

```bash
# API
CASCADE_API_PORT=8000
CASCADE_API_WORKERS=4
CASCADE_AUTH_TOKEN=secret

# L0
L0_MODEL_PATH=vincentoh/wizard101-l0-bouncer
L0_THRESHOLD=0.7
L0_BATCH_SIZE=32

# L1
L1_MODEL_ID=vincentoh/guardreasoner-8b-4bit
L1_THRESHOLD=0.7
L1_VLLM_URL=http://l1-service:8000
L1_MAX_TOKENS=256

# L3
L3_MODEL=gpt-oss-safeguard:120b
L3_OLLAMA_URL=http://l3-service:11434

# Cache
REDIS_URL=redis://cache:6379
CACHE_TTL=3600
```

### Kubernetes Resources

```yaml
# L1 Service Pod
resources:
  requests:
    nvidia.com/gpu: 1
    memory: "20Gi"
    cpu: "4"
  limits:
    nvidia.com/gpu: 1
    memory: "24Gi"
    cpu: "8"
```

## Monitoring

### Key Metrics

| Metric | Alert Threshold | Action |
|--------|-----------------|--------|
| L1 latency P99 | >1s | Scale up L1 |
| L3 queue depth | >1000 | Scale up L3 |
| Error rate | >1% | Page on-call |
| GPU utilization | >90% | Scale up |
| Cache hit rate | <50% | Increase TTL |

### Dashboard Panels

1. Request rate by tier (L0/L1/L3)
2. Latency distribution (P50/P90/P99)
3. Accuracy metrics (if ground truth available)
4. GPU utilization per service
5. Cost per 1000 requests

## Security

### Authentication

- API key authentication for all endpoints
- Rate limiting per API key
- IP allowlisting for production

### Data Privacy

- No logging of request content by default
- Optional audit mode for compliance
- Encryption in transit (TLS)
- No persistent storage of classified text

## Rollback Plan

1. **Model rollback**: Keep previous model version in registry
2. **Service rollback**: Blue-green deployment in K8s
3. **Emergency**: Route to Claude API as fallback (cost spike acceptable for availability)

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Availability | 99.9% | Uptime monitoring |
| Latency P99 | <5s | Prometheus |
| Accuracy | >95% F1 | Periodic eval on held-out set |
| Cost | <$60K/month | Cloud billing |
| Claude API calls | 0 | API gateway logs |

## Next Steps

1. **Immediate**: Set up vLLM for L1 on Nigel
2. **This week**: FastAPI wrapper with basic endpoints
3. **Next week**: Kubernetes deployment to staging
4. **Month 1**: Production deployment with monitoring

---

*Document created: 2025-11-23*
*Last updated: 2025-11-23*
