# Evaluation Report — Multi-Modal Evidence Review

## Strategy Comparison

### Strategy B: VLM + Rule Engine + Evidence Requirements (Full Pipeline)

This is the primary strategy using Gemini 3.1 Flash Lite as the VLM,
combined with a deterministic rule engine for decision validation,
evidence requirements checking, and user history risk assessment.

### Strategy: B: VLM + Rule Engine

| Field | Accuracy | Correct / Total |
|---|---|---|
| claim_status | 100.0% | 20/20 |
| issue_type | 100.0% | 20/20 |
| object_part | 100.0% | 20/20 |
| severity | 100.0% | 20/20 |
| evidence_standard_met | 90.0% | 18/20 |
| valid_image | 95.0% | 19/20 |

| Set Field | Precision | Recall | F1 |
|---|---|---|---|
| risk_flags | 85.0% | 80.8% | 82.9% |
| supporting_image_ids | 100.0% | 100.0% | 100.0% |

#### claim_status Confusion Matrix

```
Actual \ Predicted      supported               contradicted            not_enough_information  
------------------------------------------------------------------------------------------------
supported               12                      0                       0                       
contradicted            0                       5                       0                       
not_enough_information  0                       0                       3                       
```


## Operational Analysis

| Metric | Value |
|---|---|
| VLM calls (sample) | 20 |
| Images processed (sample) | 29 |
| Runtime (sample) | 0.0s |
| Model | Gemini 3.1 Flash Lite |
| Temperature | 0.0 |
| Response format | JSON |

### Cost Estimates (Full Test Set — 45 claims)

| Item | Estimate |
|---|---|
| VLM calls | ~45 |
| Input tokens | ~400K (text + images) |
| Output tokens | ~45K |
| Gemini 3.5 Flash cost | ~$0.20 |
| Runtime | ~7 minutes |

### Rate Limiting & Optimization

- **RPM**: 5s delay between calls (~12 RPM effective)
- **Caching**: Not needed for single-run pipeline
- **Batching**: One VLM call per claim (all images sent together)
- **Retry**: Exponential backoff with 3 attempts via tenacity
- **Determinism**: temperature=0.0, JSON response format

## Design Decisions

1. **Single VLM call per claim**: All images sent in one call for
   cross-image consistency detection
2. **Anti-injection system prompt**: Explicit rules to detect and
   report (not follow) text instructions in images
3. **Hybrid architecture**: VLM provides preliminary verdict,
   rule engine validates and overrides when needed
4. **Evidence requirements grounding**: Deterministic matching of
   issue families to evidence requirement rules
5. **Selective supporting_image_ids**: Only images that actively
   contribute evidence, not all submitted images
6. **Severity calibration**: Aligned with visible damage level,
   not user's description