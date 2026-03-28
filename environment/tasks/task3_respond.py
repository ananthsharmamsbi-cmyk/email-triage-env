"""
Task 3 (Hard) — Response Drafting Grader

Scoring is multi-dimensional:
  1. Rule-based checks  (25%): must_include / must_not_include keyword presence
  2. LLM-judged quality (50%): professionalism, empathy, correctness, completeness
  3. Structural checks  (25%): escalation flag accuracy, non-empty fields, response length

When no LLM client is provided (offline mode), only rule-based + structural scores apply
and the total is rescaled to 0–1 range.
"""
from __future__ import annotations
import re
from typing import Any, Optional
from environment.models import DraftAction, Reward


# ---------------------------------------------------------------------------
# Rule-based helpers
# ---------------------------------------------------------------------------

def _keywords_present(text: str, keywords: list[str]) -> float:
    """Fraction of keywords found (case-insensitive) in the combined response text."""
    text_lower = text.lower()
    if not keywords:
        return 1.0
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def _rule_based_score(action: DraftAction, label: dict) -> tuple[float, dict]:
    combined_text = (
        action.response_subject + " " +
        action.response_body + " " +
        (action.internal_note or "")
    )

    must_include_score    = _keywords_present(combined_text, label.get("must_include", []))
    must_not_include_hits = _keywords_present(combined_text, label.get("must_not_include", []))
    must_not_score        = max(0.0, 1.0 - must_not_include_hits)   # penalise forbidden phrases

    rule_score = (must_include_score * 0.7) + (must_not_score * 0.3)
    return rule_score, {
        "must_include_coverage": round(must_include_score, 4),
        "forbidden_phrase_penalty": round(must_not_include_hits, 4),
    }


def _structural_score(action: DraftAction, label: dict) -> tuple[float, dict]:
    scores = {}

    # Response body length check (ideal: 100–600 words)
    word_count = len(action.response_body.split())
    if 80 <= word_count <= 700:
        length_score = 1.0
    elif word_count < 30 or word_count > 1200:
        length_score = 0.0
    else:
        length_score = 0.5
    scores["response_length_ok"] = round(length_score, 4)

    # Non-empty required fields
    fields_ok = all([
        bool(action.response_subject.strip()),
        bool(action.response_body.strip()),
        bool(action.internal_note.strip()),
    ])
    scores["fields_populated"] = 1.0 if fields_ok else 0.0

    # Escalation flag accuracy
    escalate_expected = label.get("escalate", False)
    escalate_correct  = (action.escalate == escalate_expected)
    scores["escalation_correct"] = 1.0 if escalate_correct else 0.0

    struct_score = (
        length_score         * 0.4 +
        scores["fields_populated"] * 0.3 +
        scores["escalation_correct"] * 0.3
    )
    return struct_score, scores


# ---------------------------------------------------------------------------
# LLM judge (called only when an OpenAI client is available)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert customer support quality evaluator. 
Score the following support email response on a scale of 0.0 to 1.0.

## Context
Customer email thread:
{thread}

## Agent's Response
Subject: {subject}
Body:
{body}

## Evaluation Criteria (score each 0.0–1.0):
1. **Professionalism** – Is the tone appropriate, respectful, and free of errors?
2. **Empathy** – Does it acknowledge the customer's frustration/concern appropriately?
3. **Accuracy** – Is the response factually correct and aligned with policy?
4. **Completeness** – Does it address ALL issues raised by the customer?

Return ONLY a JSON object (no markdown) in this exact format:
{{"professionalism": <float>, "empathy": <float>, "accuracy": <float>, "completeness": <float>}}
"""


def _llm_judge(
    action: DraftAction,
    thread_text: str,
    client: Any,
    model: str,
) -> tuple[float, dict]:
    """Call an LLM to judge response quality. Returns (score, breakdown)."""
    import json as _json

    prompt = _JUDGE_PROMPT.format(
        thread=thread_text,
        subject=action.response_subject,
        body=action.response_body,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        scores_dict = _json.loads(raw)
        llm_score   = sum(scores_dict.values()) / 4.0
        return round(llm_score, 4), {f"llm_{k}": round(v, 4) for k, v in scores_dict.items()}
    except Exception as exc:
        # Graceful fallback — don't crash the environment
        return 0.5, {"llm_judge_error": str(exc)}


# ---------------------------------------------------------------------------
# Public grader entry-point
# ---------------------------------------------------------------------------

def grade(
    action: DraftAction,
    label: dict,
    thread_text: str = "",
    llm_client: Optional[Any] = None,
    llm_model: str = "gpt-4o-mini",
) -> Reward:
    rule_score,   rule_bd   = _rule_based_score(action, label)
    struct_score, struct_bd = _structural_score(action, label)

    breakdown = {**rule_bd, **struct_bd}

    if llm_client is not None:
        llm_score, llm_bd = _llm_judge(action, thread_text, llm_client, llm_model)
        breakdown.update(llm_bd)
        total = rule_score * 0.25 + llm_score * 0.50 + struct_score * 0.25
    else:
        # Offline: rescale rule + structural across full range
        total = rule_score * 0.50 + struct_score * 0.50

    breakdown["rule_score"]   = round(rule_score, 4)
    breakdown["struct_score"] = round(struct_score, 4)

    return Reward(
        score=round(min(max(total, 0.0), 1.0), 4),
        breakdown=breakdown,
        penalty=0.0,
        message=f"Draft quality score: {total:.3f}"
    )
