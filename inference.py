"""
inference.py — Baseline inference script for the Customer Support Email Triage OpenEnv.

Runs a GPT model as an agent through all 3 tasks and reports reproducible scores.
Uses the OpenAI client with credentials read from environment variables.

Required environment variables:
  API_BASE_URL   — LLM API base URL  (e.g. https://api.openai.com/v1)
  MODEL_NAME     — Model identifier  (e.g. gpt-4o-mini)
  HF_TOKEN       — API key

Usage:
  python inference.py
  python inference.py --seed 42
  python inference.py --task email_classify
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from environment import EmailTriageEnv
from environment.models import TaskID, ResetRequest

# ---------------------------------------------------------------------------
# Configuration from env vars
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
API_KEY      = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

if not API_KEY:
    print("[ERROR] No API key found. Set HF_TOKEN or OPENAI_API_KEY.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Per-task system prompts and action parsers
# ---------------------------------------------------------------------------

_SYS_CLASSIFY = """You are an expert customer support email triage agent.
Given an email, output ONLY a JSON object with:
  - "category": one of ["billing", "technical", "shipping", "general", "spam"]
  - "urgency":  one of ["low", "medium", "high", "critical"]

Rules:
- billing: payment, invoice, charge, refund, subscription, pricing
- technical: bug, error, API, integration, broken feature, can't login
- shipping: delivery, order, package, tracking, return, wrong item
- spam: solicitation, prize, phishing, unrelated promotion
- general: sales inquiry, how-to, onboarding, general question
- urgency follows business impact: critical=production down/legal risk, high=core functionality, medium=annoying, low=informational

Return ONLY the JSON, no explanation."""

_SYS_EXTRACT = """You are an expert at extracting structured information from customer support email threads.
Given a thread of emails, output ONLY a JSON object with these exact fields:
  - "customer_name":     string — full name of the customer (from email body/signature)
  - "issue_summary":     string — concise 1-2 sentence description of the core problem (max 50 words)
  - "product_mentioned": string — the specific product/feature name mentioned
  - "resolution_needed": string — what action is needed to resolve this (max 30 words)
  - "estimated_effort":  one of ["quick", "medium", "extensive"]
    (quick=<30min, medium=30min-4hr, extensive=>4hr or complex investigation)

Return ONLY the JSON, no explanation."""

_SYS_RESPOND = """You are a senior customer support specialist. You always respond professionally,
empathetically, and with clear action steps.
Given a customer complaint thread, customer history, and company policies, output ONLY a JSON object:
  - "response_subject":  string — professional email subject line
  - "response_body":     string — complete email body to send to the customer (150-400 words)
                                   Address ALL issues raised. Use the customer's name. Offer concrete next steps.
  - "internal_note":     string — internal CRM note for the support team (30-80 words)
  - "escalate":          boolean — true if this needs senior agent / management involvement

Return ONLY the JSON, no markdown, no explanation."""


def _obs_to_prompt(task: TaskID, obs: Dict[str, Any]) -> str:
    """Convert a raw observation dict to a user prompt string."""

    if task == TaskID.EMAIL_CLASSIFY:
        email = obs["email"]
        return (
            f"CLASSIFY THIS EMAIL:\n\n"
            f"From: {email['sender']}\n"
            f"Subject: {email['subject']}\n\n"
            f"{email['body']}\n\n"
            f"(Received: {email['timestamp']})"
        )

    elif task == TaskID.TICKET_EXTRACT:
        parts = []
        for i, msg in enumerate(obs["thread"], 1):
            parts.append(
                f"--- Message {i} ---\n"
                f"From: {msg['sender']}\n"
                f"Subject: {msg['subject']}\n"
                f"{msg['body']}"
            )
        return "EXTRACT TICKET METADATA FROM THIS THREAD:\n\n" + "\n\n".join(parts)

    else:  # RESPONSE_DRAFT
        parts = []
        for i, msg in enumerate(obs["thread"], 1):
            parts.append(
                f"--- Customer Message {i} ---\n"
                f"From: {msg['sender']}\n"
                f"Subject: {msg['subject']}\n"
                f"{msg['body']}"
            )
        hist = obs["customer_history"]
        hist_text = (
            f"\nCUSTOMER ACCOUNT DETAILS:\n"
            f"  Tier: {hist['tier']} | Account age: {hist['account_age_days']} days | "
            f"Open tickets: {hist['open_tickets']}"
        )
        policies = "\n".join(
            f"\nPOLICY — {p['title']}:\n{p['content']}"
            for p in obs.get("policy_snippets", [])
        )
        return (
            "DRAFT A PROFESSIONAL SUPPORT RESPONSE:\n\n" +
            "\n\n".join(parts) +
            hist_text + "\n" +
            policies
        )


def _system_prompt(task: TaskID) -> str:
    return {
        TaskID.EMAIL_CLASSIFY: _SYS_CLASSIFY,
        TaskID.TICKET_EXTRACT: _SYS_EXTRACT,
        TaskID.RESPONSE_DRAFT: _SYS_RESPOND,
    }[task]


def _call_llm(system: str, user: str, retries: int = 3) -> str:
    """Call the LLM with retry on transient errors."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"  [WARN] LLM call failed (attempt {attempt+1}/{retries}): {exc}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def _parse_action(raw: str, task: TaskID) -> Dict[str, Any]:
    """Extract JSON from LLM output."""
    import re
    # Strip markdown code fences
    cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find the first JSON object in the output
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from LLM output: {raw!r}")


# ---------------------------------------------------------------------------
# Run one full task episode
# ---------------------------------------------------------------------------

def run_task(
    env: EmailTriageEnv,
    task: TaskID,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one complete episode and return results."""

    print(f"\n{'='*60}")
    print(f"TASK: {task.value.upper()}")
    print(f"{'='*60}")

    obs = env.reset(ResetRequest(task=task, seed=seed))
    system = _system_prompt(task)

    step_num     = 0
    step_scores: List[float] = []
    done         = False

    while not done:
        step_num += 1
        total     = obs.get("total_steps", "?")
        idx       = obs.get("step_index",  step_num - 1)

        print(f"\n  Step {step_num}/{total}")

        # Build prompt from observation
        user_prompt = _obs_to_prompt(task, obs)

        # Get LLM action
        raw_output  = _call_llm(system, user_prompt)
        action_dict = _parse_action(raw_output, task)

        if verbose:
            print(f"  Action: {json.dumps(action_dict, indent=2)[:300]}...")

        # Submit to environment
        result       = env.step(action_dict)
        reward       = result.reward
        done         = result.done
        obs          = result.observation or {}
        step_scores.append(reward.score)

        print(f"  Score: {reward.score:.3f} | {reward.message}")
        if verbose and reward.breakdown:
            breakdown_preview = {k: v for k, v in list(reward.breakdown.items())[:4]}
            print(f"  Breakdown: {breakdown_preview}")

    avg_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
    print(f"\n  EPISODE COMPLETE — Average score: {avg_score:.4f}")

    return {
        "task":         task.value,
        "seed":         seed,
        "step_scores":  [round(s, 4) for s in step_scores],
        "avg_score":    round(avg_score, 4),
        "total_steps":  len(step_scores),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline inference script for Email Triage OpenEnv"
    )
    parser.add_argument("--seed",  type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--task",  type=str, default=None,
                        choices=[t.value for t in TaskID],
                        help="Run only this task (default: all)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print("  Customer Support Email Triage — OpenEnv Baseline")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  API URL: {API_BASE_URL}")
    print(f"  Seed:    {args.seed}")
    print(f"{'#'*60}")

    # Create env with LLM judge enabled (for Task 3 grader)
    env = EmailTriageEnv(llm_client=client, llm_model=MODEL_NAME)

    tasks_to_run = (
        [TaskID(args.task)] if args.task
        else [TaskID.EMAIL_CLASSIFY, TaskID.TICKET_EXTRACT, TaskID.RESPONSE_DRAFT]
    )

    all_results: List[Dict[str, Any]] = []
    start_time = time.time()

    for task in tasks_to_run:
        result = run_task(env, task, seed=args.seed, verbose=args.verbose)
        all_results.append(result)

    elapsed = time.time() - start_time

    # ---- Summary -------------------------------------------------------
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  {'Task':<28} {'Steps':>5}  {'Avg Score':>10}")
    print(f"  {'-'*28} {'-'*5}  {'-'*10}")
    overall_scores = []
    for r in all_results:
        print(f"  {r['task']:<28} {r['total_steps']:>5}  {r['avg_score']:>10.4f}")
        overall_scores.append(r["avg_score"])
    overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    print(f"  {'OVERALL':.<28} {'':>5}  {overall:>10.4f}")
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Machine-readable output
    output = {
        "model":          MODEL_NAME,
        "api_base":       API_BASE_URL,
        "seed":           args.seed,
        "results":        all_results,
        "overall_score":  round(overall, 4),
        "elapsed_s":      round(elapsed, 1),
    }
    print(f"\n--- JSON RESULTS ---")
    print(json.dumps(output, indent=2))

    # Save to file for CI
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n[INFO] Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
