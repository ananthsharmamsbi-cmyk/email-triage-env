---
title: Customer Support Email Triage
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - customer-support
  - nlp
  - rl
license: mit
---

# 📧 Customer Support Email Triage — OpenEnv

> A real-world OpenEnv environment where AI agents learn to handle customer support emails:  
> classify them, extract structured metadata, and draft professional responses.

---

## Why This Environment?

Every SaaS company and e-commerce platform runs a support inbox. Agents that can accurately triage emails, extract ticket data, and draft on-brand responses **reduce resolution time, improve CSAT scores, and scale support operations**. This environment provides:

- **Immediate practical value**: Skills transfer directly to production support systems
- **Rich evaluation signal**: Partial credit at every step, not just binary success
- **Difficulty ladder**: From simple 5-class classification → complex multi-criteria response evaluation
- **Reproducible benchmarking**: Fixed datasets + deterministic graders (+ optional LLM judge for Task 3)

---

## Environment Description

An agent works through a batch of customer support emails in a single episode. Each episode is one task × N emails. The agent acts once per email and receives a reward signal immediately.

```
reset(task="email_classify", seed=42)
  → observation (email #1)
  
step(action)
  → next observation, reward, done=False, info

step(action)
  → next observation, reward, done=False, info
  ...

step(action)
  → observation=None, reward, done=True, info
```

---

## Tasks

### Task 1 — Email Classification (`email_classify`) · **Easy**

| Property | Value |
|---|---|
| Episode length | 5 emails |
| Difficulty | Easy |
| Score range | 0.0 – 1.0 per step |

**Objective**: Given a single customer email, output its **category** and **urgency**.

**Categories**: `billing` · `technical` · `shipping` · `general` · `spam`  
**Urgency levels**: `low` · `medium` · `high` · `critical`

**Scoring**:
- +0.50 for correct category (exact match)
- +0.50 for correct urgency (exact match)
- +0.25 partial credit for urgency ±1 tier (e.g. predicted `high`, label `critical`)

**Action format**:
```json
{
  "category": "billing",
  "urgency": "high",
  "reasoning": "optional chain-of-thought"
}
```

---

### Task 2 — Ticket Metadata Extraction (`ticket_extraction`) · **Medium**

| Property | Value |
|---|---|
| Episode length | 3 threads |
| Difficulty | Medium |
| Score range | 0.0 – 1.0 per step |

**Objective**: Given a multi-message email thread (1–3 messages), extract 5 structured fields.

**Scoring**: Weighted token-overlap F1 across fields:
| Field | Weight |
|---|---|
| `customer_name` | 20% |
| `issue_summary` | 30% |
| `product_mentioned` | 20% |
| `resolution_needed` | 20% |
| `estimated_effort` | 10% (exact + ±1 partial) |

**Action format**:
```json
{
  "customer_name": "Lisa Park",
  "issue_summary": "Bulk CSV export produces empty files for >10,000 rows",
  "product_mentioned": "DataSync Pro",
  "resolution_needed": "Fix export to correctly generate large CSV files",
  "estimated_effort": "medium"
}
```

---

### Task 3 — Response Drafting (`response_drafting`) · **Hard**

| Property | Value |
|---|---|
| Episode length | 2 scenarios |
| Difficulty | Hard |
| Score range | 0.0 – 1.0 per step |

**Objective**: Draft a complete professional support response to a complex complaint, given the email thread, customer account history, and relevant company policy snippets.

**Scoring** (composite):
| Component | Weight | Method |
|---|---|---|
| Must-include coverage | 25% | Keyword presence check |
| LLM judge quality | 50% | GPT-based rubric (professionalism, empathy, accuracy, completeness) |
| Structural quality | 25% | Response length, all fields populated, escalation flag |

**Action format**:
```json
{
  "response_subject": "Re: Your recent billing concern — action taken",
  "response_body": "Dear Robert, Thank you for bringing this to our attention...",
  "internal_note": "Enterprise customer, SLA breach confirmed. Issued 10% credit.",
  "escalate": true
}
```

---

## Observation Space

All observations share these common fields:

| Field | Type | Description |
|---|---|---|
| `task` | string | Active task ID |
| `step_index` | int | Current step (0-indexed) |
| `total_steps` | int | Total steps in episode |
| `score_so_far` | float | Running average score |

**Task 1** adds: `email` (message_id, sender, subject, body, timestamp, metadata)

**Task 2** adds: `thread` (array of email messages)

**Task 3** adds: `thread`, `customer_history` (tier, age, open_tickets), `policy_snippets` (title, content)

---

## Reward Object

Every `step()` returns a structured `Reward`:

```json
{
  "score": 0.75,
  "breakdown": {
    "category": 0.5,
    "urgency": 0.25,
    "expected_category": "shipping",
    "predicted_category": "shipping",
    "expected_urgency": "critical",
    "predicted_urgency": "high"
  },
  "penalty": 0.0,
  "message": "Category: ✓ | Urgency: ~"
}
```

---

## API Reference

The environment runs as a FastAPI service on port **7860**.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/tasks` | List all tasks with metadata |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Submit action, receive reward |
| `GET` | `/state` | Current episode state |
| `GET` | `/openenv.yaml` | OpenEnv spec file |
| `GET` | `/docs` | Interactive Swagger UI |

### Reset request
```json
POST /reset
{
  "task": "email_classify",
  "seed": 42
}
```

### Step request
```json
POST /step
{
  "action": {
    "category": "billing",
    "urgency": "high"
  }
}
```

---

## Setup & Usage

### Option 1 — Docker (recommended)

```bash
# Build
docker build -t email-triage-env .

# Run (with API key for full Task 3 LLM judging)
docker run -p 7860:7860 \
  -e HF_TOKEN=your_key_here \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  email-triage-env

# Health check
curl http://localhost:7860/
```

### Option 2 — Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Set credentials
export HF_TOKEN=your_key_here
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini

# Start server
uvicorn app:app --host 0.0.0.0 --port 7860

# Or run inference directly
python inference.py --seed 42
```

### Quick API test
```bash
# Reset to task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "email_classify", "seed": 42}'

# Submit action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"category": "billing", "urgency": "high"}}'

# Check state
curl http://localhost:7860/state
```

---

## Running the Baseline

```bash
# All tasks (requires HF_TOKEN / OPENAI_API_KEY)
python inference.py --seed 42

# Single task
python inference.py --task email_classify --seed 42
```

### Expected Baseline Scores (gpt-4o-mini, seed=42)

| Task | Avg Score | Notes |
|---|---|---|
| `email_classify` | ~0.82 | Strong on category; occasional urgency off-by-one |
| `ticket_extraction` | ~0.71 | Good on name/product; summaries lose points on brevity |
| `response_drafting` | ~0.68 | Solid structure; sometimes misses specific policy citations |
| **Overall** | **~0.74** | |

> Scores are approximate and depend on the model configuration. Task 3 uses LLM judging, so scores may vary slightly across runs.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier for agent + Task 3 judge |
| `HF_TOKEN` | Yes | API key (passed as OpenAI API key) |
| `OPENAI_API_KEY` | Fallback | Used if `HF_TOKEN` not set |

---

## Project Structure

```
.
├── app.py                        # FastAPI server (OpenEnv REST API)
├── inference.py                  # Baseline LLM agent script
├── openenv.yaml                  # OpenEnv specification
├── requirements.txt
├── Dockerfile
├── README.md
├── data/
│   └── emails.json               # Synthetic email dataset (10+5+3 scenarios)
└── environment/
    ├── __init__.py
    ├── env.py                    # Core EmailTriageEnv class
    ├── models.py                 # Typed Pydantic models
    └── tasks/
        ├── __init__.py
        ├── task1_classify.py     # Easy grader
        ├── task2_extract.py      # Medium grader (token F1)
        └── task3_respond.py      # Hard grader (rule + LLM judge)
```

---

## Design Decisions

**Why email triage?**  
Every support team does this. The three sub-tasks (classify → extract → respond) form a natural pipeline that increases in cognitive complexity, mirroring how real agents would be deployed progressively.

**Why token F1 for extraction?**  
Exact string match is too strict for summaries written differently but semantically correct. Token F1 rewards getting the right concepts without over-penalising paraphrases.

**Why LLM judging for Task 3?**  
Response quality is fundamentally multidimensional and hard to capture with heuristics alone. The LLM judge evaluates professionalism, empathy, accuracy, and completeness — dimensions that matter in production. The judge prompt is deterministic (temperature=0), making scores reproducible.

**Partial rewards throughout**  
Every step provides a signal. No step is purely binary. This makes the environment useful for RL training, not just evaluation.

---

## License

MIT
