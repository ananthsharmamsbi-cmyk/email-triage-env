# Customer Support Email Triage — Complete Project Guide

---

## 1. One-Line Summary

> "A standardised AI training environment that teaches and evaluates language models on three levels of
> customer support tasks — from simple email classification to writing full professional responses —
> hosted on Hugging Face Spaces and compatible with any RL training framework."

---

## 2. What Problem Does It Solve?

Every company — banks, e-commerce, SaaS — gets thousands of customer support emails daily.
Human agents must:
1. Read each email and decide what type it is and how urgent
2. Pull out key information (who, what, how urgent, what product)
3. Write a professional reply

This is time-consuming, expensive, and inconsistent. AI can help — but AI needs to be
**trained and evaluated** on this task first. That is exactly what this environment does.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Agent (inference.py)                       │
│                  OpenAI Client / gpt-4o-mini                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP (POST /reset, POST /step)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│             Hugging Face Spaces — Docker, Port 7860             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  FastAPI Server (app.py)                 │   │
│  │   POST /reset   POST /step   GET /state   GET /health   │   │
│  └───────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │              EmailTriageEnv (env.py)                     │   │
│  │         Episode Manager — seed, state, history           │   │
│  └────┬──────────────────┬──────────────────┬──────────────┘   │
│       │                  │                  │                    │
│  ┌────▼────┐        ┌────▼────┐        ┌────▼────┐            │
│  │ Task 1  │        │ Task 2  │        │ Task 3  │            │
│  │Classify │        │Extract  │        │ Draft   │            │
│  │  Easy   │        │ Medium  │        │  Hard   │            │
│  └────┬────┘        └────┬────┘        └────┬────┘            │
│       │                  │                  │                    │
│       └──────────────────┴──────────┬───────┘                  │
│                                     │                            │
│  ┌──────────────────────────────────▼───────────────────────┐  │
│  │              Pydantic Models (models.py)                  │  │
│  │         Observation | Action | Reward | EpisodeState      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              data/emails.json (18 scenarios)              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                           │
                           │  Task 3 LLM Judge (temperature=0)
                           ▼
                    ┌──────────────┐
                    │   LLM API    │
                    │  (OpenAI /   │
                    │  compatible) │
                    └──────────────┘
```

### How the agent loop works step by step

```
1. Agent calls POST /reset  {"task": "email_classify", "seed": 42}
   ← Environment returns first email as observation

2. Agent reads the email, decides: {"category": "billing", "urgency": "high"}
   Agent calls POST /step  {"action": {"category": "billing", "urgency": "high"}}
   ← Environment returns: score=0.75, next email, done=false

3. Agent repeats step 2 for each email in the episode

4. When all emails are done, environment returns done=true with final info
```

---

## 4. Project File Structure

```
ScalarHackathon/
├── app.py                         FastAPI server — all API endpoints
├── inference.py                   Baseline agent script (runs GPT against env)
├── openenv.yaml                   OpenEnv specification file
├── requirements.txt               Python dependencies
├── Dockerfile                     Container definition for HF Spaces
├── README.md                      Full documentation
├── .env.example                   Template for environment variables
├── .github/
│   └── workflows/
│       └── deploy.yml             Auto-deploy to HF Spaces on git push
├── data/
│   └── emails.json                18 realistic email scenarios with labels
└── environment/
    ├── __init__.py
    ├── env.py                     Core environment — episode management
    ├── models.py                  All Pydantic models (Observation/Action/Reward)
    └── tasks/
        ├── __init__.py
        ├── task1_classify.py      Easy grader — exact match + partial credit
        ├── task2_extract.py       Medium grader — token F1 per field
        └── task3_respond.py       Hard grader — rule + LLM judge + structural
```

---

## 5. The Three Tasks In Detail

### Task 1 — Email Classification (Easy)
- **What the agent sees:** A single customer email (sender, subject, body)
- **What the agent must do:** Label it with a category and urgency level
- **Categories:** billing / technical / shipping / general / spam
- **Urgency:** low / medium / high / critical
- **How it's scored:**
  - +0.50 for correct category (exact match)
  - +0.50 for correct urgency (exact match)
  - +0.25 partial credit if urgency is one level off (e.g. said high, answer was critical)
  - Maximum score per email: 1.0
- **Episode length:** 5 emails

### Task 2 — Ticket Metadata Extraction (Medium)
- **What the agent sees:** A thread of 1-3 emails between customer and support
- **What the agent must do:** Fill in 5 structured fields like a CRM ticket
  - customer_name — who is the customer
  - issue_summary — what is the problem in 1-2 sentences
  - product_mentioned — which product or feature
  - resolution_needed — what action is required
  - estimated_effort — quick / medium / extensive
- **How it's scored:** Token-overlap F1 (like academic NLP benchmarks)
  - Rewards getting the right concepts even if wording differs
  - Weighted: issue_summary counts 30%, others 20% each, effort 10%
- **Episode length:** 3 threads

### Task 3 — Response Drafting (Hard)
- **What the agent sees:** A complex complaint email + customer account history + company policy snippets
- **What the agent must do:** Write a complete professional response including:
  - response_subject — email subject line
  - response_body — full email to send to customer (150-400 words)
  - internal_note — internal CRM note for the team
  - escalate — true/false decision
- **How it's scored (composite):**
  - 25% — Rule-based: are required phrases present? Are forbidden phrases absent?
  - 50% — LLM judge: scores professionalism, empathy, accuracy, completeness
  - 25% — Structural: correct length, all fields populated, escalation flag correct
- **Episode length:** 2 complaint scenarios

---

## 6. API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | /health | Health check — returns {"status": "healthy"} |
| GET | / | Root — returns env info and task list |
| GET | /tasks | Lists all 3 tasks with metadata |
| POST | /reset | Start new episode — body: {task, seed} |
| POST | /step | Submit action — body: {action: {...}} |
| GET | /state | Current episode state |
| GET | /openenv.yaml | OpenEnv spec file |
| GET | /docs | Interactive Swagger API docs |

---

## 7. Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| API_BASE_URL | LLM API endpoint | https://api.openai.com/v1 |
| MODEL_NAME | Model to use | gpt-4o-mini |
| HF_TOKEN | API key (also used as OpenAI key) | hf_xxxx or sk-xxxx |

---

## 8. Reward Design

Every step returns a structured Reward object:
```json
{
  "score": 0.75,
  "breakdown": {
    "category": 0.5,
    "urgency": 0.25,
    "expected_urgency": "critical",
    "predicted_urgency": "high"
  },
  "penalty": 0.0,
  "message": "Category: ✓ | Urgency: ~"
}
```

**Why partial rewards matter:** In reinforcement learning, the agent needs signal
to know it is improving. Binary pass/fail gives almost no learning signal early on.
Partial credit means even an imperfect agent learns which direction to improve.

---

## 9. Expected Baseline Scores (gpt-4o-mini, seed=42)

| Task | Score | Notes |
|------|-------|-------|
| email_classify | ~0.82 | Strong on category, occasional urgency off-by-one |
| ticket_extraction | ~0.71 | Good on names/products, verbose on summaries |
| response_drafting | ~0.68 | Solid structure, sometimes misses policy citations |
| **Overall** | **~0.74** | |

---

## 10. Key Technical Decisions

| Decision | Reason |
|----------|--------|
| FastAPI over Flask | Automatic Pydantic validation, async support, auto docs |
| Pydantic models | Type safety, automatic validation, OpenEnv spec compliance |
| Token F1 for Task 2 | Rewards semantically correct extractions, not just exact string match |
| LLM judge for Task 3 | Response quality is multidimensional, hard to capture with rules alone |
| temperature=0 for judge | Makes LLM judgment deterministic and reproducible |
| Docker on HF Spaces | Containerised = reproducible anywhere, standard HF deployment |
| Seed in reset() | Same seed = same episode every time = fair comparison across models |

---

## 11. Judge Questions & Answers

**Q: Why customer support email triage?**
"It is a task every real company does daily at scale. It has a natural difficulty
progression — classifying is easy, extracting structure is medium, generating a
response is hard. The skills an AI learns here transfer directly to production.
I wanted something with immediate practical value, not a game."

**Q: How do your graders work? Are they fair?**
"Task 1 uses exact match with partial credit for urgency — one tier off still gets
50% credit. Task 2 uses token-overlap F1, the same technique used in SQuAD and
other NLP benchmarks. Task 3 combines rule-based checks, an LLM judge at
temperature=0 for reproducibility, and structural checks."

**Q: How do you ensure reproducibility?**
"The reset() endpoint accepts an integer seed. With the same seed, the same emails
are always selected in the same order. Tasks 1 and 2 graders are fully deterministic.
Task 3's LLM judge runs at temperature=0. The baseline script always uses seed=42."

**Q: What is the reward function doing?**
"Every step returns a score between 0 and 1 — never just pass/fail. This gives the
agent a gradient to follow during training. If it classifies the category correctly
but gets urgency wrong, it gets 0.5 and knows it is on the right track. Sparse
binary rewards make reinforcement learning much harder."

**Q: How does an AI agent actually use this environment?**
"It calls POST /reset with a task name, gets an observation — the email content.
It calls POST /step with its action. It gets back a score and the next email.
It keeps stepping until done=true. This is the same loop used in OpenAI Gym,
just for text tasks instead of games."

**Q: What is the OpenEnv spec and why does it matter?**
"OpenEnv is a standardised interface for text-based RL environments — like
Gymnasium but for language tasks. By following the spec, any training framework
such as TRL or OpenRLHF can connect to this environment without custom code.
The openenv.yaml file describes the environment metadata, action and observation
spaces, and task definitions."

**Q: What are your baseline scores?**
"On gpt-4o-mini with seed=42: Task 1 scores around 0.82, Task 2 around 0.71,
Task 3 around 0.68. Overall approximately 0.74. The model is strongest at
classification and weakest at drafting responses that cite specific policy details."

**Q: What would you improve with more time?**
"More email scenarios — currently 18 total. A WebSocket endpoint for more efficient
multi-agent training. Harder adversarial examples designed to trick the classifier.
A leaderboard to compare different models on the same tasks."

**Q: Why did you use Docker?**
"Docker ensures the environment runs identically everywhere — on my laptop, on HF
Spaces, on a judge's machine. No dependency conflicts, no OS differences. The
Dockerfile uses Python 3.11-slim to keep the image small and fast to build."

**Q: Does this work for RL training or just evaluation?**
"Both. For evaluation, you run inference.py and get scores. For RL training, a
framework like TRL's GRPOTrainer calls reset() and step() in a loop, collects
rewards, and updates model weights. The partial reward design makes it
particularly useful for RL — the agent gets meaningful signal at every step."

---

## 12. Key Numbers to Remember

| Item | Value |
|------|-------|
| Total email scenarios | 18 |
| Task 1 episode length | 5 emails |
| Task 2 episode length | 3 threads |
| Task 3 episode length | 2 complaints |
| Score range | 0.0 to 1.0 |
| Task 1 baseline score | ~0.82 |
| Task 2 baseline score | ~0.71 |
| Task 3 baseline score | ~0.68 |
| Overall baseline | ~0.74 |
| Server port | 7860 |
| Python version | 3.11 |
| Inference script timeout | under 20 minutes |
| Memory requirement | under 512 MB |

---

## 13. Submission Checklist

- [x] HF Space deploys and responds to /health
- [x] openenv.yaml present and valid
- [x] Dockerfile builds and runs
- [x] inference.py in root directory
- [x] inference.py uses OpenAI client
- [x] Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
- [x] 3 tasks with graders producing scores 0.0 to 1.0
- [x] Graders are deterministic
- [x] reset() returns clean initial state
- [x] step() returns observation, reward, done, info
- [x] state() returns current episode state
- [x] Partial progress rewards (not binary)
- [x] README with all required sections
- [x] Inference runs in under 20 minutes
- [x] Runs on 2 vCPU, 8 GB RAM machine
