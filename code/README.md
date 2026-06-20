# Multi-Modal Evidence Review — Solution

## Overview

A VLM-powered claim verification system that analyzes damage claims using submitted images, claim conversations, user history, and evidence requirements. Built for the HackerRank Orchestrate June 2026 hackathon.

## Architecture

```
Claims CSV → Per-Row Pipeline
                │
    ┌───────────┼────────────────┐
    │           │                │
    ▼           ▼                ▼
Claim        User History    Evidence
Extractor    (CSV lookup)    Requirements
(NLP parse)                  (CSV lookup)
    │           │                │
    └─────┬─────┘                │
          ▼                      │
  Vision Analyzer                │
  (Gemini 2.5 Flash)            │
  ├─ Per-image analysis          │
  ├─ Cross-image consensus       │
  ├─ Claim comparison            │
  └─ Anti-injection defense      │
          │                      │
          ▼                      ▼
  ┌────────────────────────────────┐
  │      Decision Engine           │
  │  (Rule + VLM Hybrid)          │
  │  13 validation rules          │
  └────────────────────────────────┘
          │
          ▼
  Output CSV Generator
  (Pydantic validation)
```

## Key Design Decisions

1. **Single VLM call per claim** — All images sent together for cross-image consistency detection
2. **Anti-injection system prompt** — Detects and reports (never follows) text instructions in images
3. **Hybrid VLM + Rule Engine** — VLM provides preliminary verdict; rules validate and override
4. **Evidence requirements grounding** — Deterministic matching of claim families to requirement rules
5. **Selective supporting_image_ids** — Only images that actively contribute evidence
6. **Multi-language support** — Handles Hindi, Spanish, Chinese conversations

## Setup

```bash
cd code
pip install -r requirements.txt
```

Create a `.env` file in the repo root:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

### Run on test claims (produce output.csv)
```bash
python code/main.py
```

### Run on sample claims (for evaluation)
```bash
python code/main.py --input dataset/sample_claims.csv --output sample_output.csv
```

### Run evaluation
```bash
python code/evaluation/main.py
```

### CLI options
```bash
python code/main.py --help
python code/main.py --model gemini-2.5-flash --delay 2.0
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Pipeline orchestrator and CLI entry point |
| `schemas.py` | Pydantic models, enums, and validation |
| `claim_extractor.py` | NLP parsing of claim conversations |
| `vision_analyzer.py` | Gemini 2.5 Flash VLM engine |
| `evidence_checker.py` | Evidence requirements matching |
| `risk_engine.py` | User history risk assessment |
| `decision_engine.py` | Hybrid Rule + VLM decision logic |
| `output_generator.py` | CSV output with schema validation |
| `evaluation/main.py` | Evaluation entry point |
| `evaluation/evaluate.py` | Evaluation runner and report generator |
| `evaluation/metrics.py` | Accuracy, F1, confusion matrix metrics |

## Model

- **Model**: Google Gemini 2.5 Flash
- **Temperature**: 0.0 (deterministic)
- **Response format**: Structured JSON
- **Cost**: ~$0.20 for full test set (45 claims)

## Evaluation

The evaluation compares predictions against `sample_claims.csv` ground truth across:
- claim_status accuracy
- issue_type accuracy
- object_part accuracy
- severity accuracy
- evidence_standard_met accuracy
- risk_flags F1
- supporting_image_ids F1

Results are written to `evaluation/evaluation_report.md`.
