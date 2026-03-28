"""
Task 2 (Medium) — Ticket Metadata Extraction Grader
Uses fuzzy token-overlap F1 for text fields + exact match for effort.
Score 0.0–1.0 with per-field breakdown.
"""
from __future__ import annotations
import re
from typing import Set
from environment.models import ExtractAction, EffortLevel, Reward

# Field weights must sum to 1.0
_FIELD_WEIGHTS = {
    "customer_name":     0.20,
    "issue_summary":     0.30,
    "product_mentioned": 0.20,
    "resolution_needed": 0.20,
    "estimated_effort":  0.10,
}

_STOP_WORDS = {"the", "a", "an", "is", "it", "in", "on", "for", "of", "to", "and", "or",
               "that", "this", "with", "by", "at", "from", "be", "are", "was", "were",
               "has", "have", "had", "not", "we", "our", "your", "my", "their"}


def _tokenize(text: str) -> Set[str]:
    """Lowercase, strip punctuation, remove stop words."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}


def _token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between two strings."""
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(gold)
    if not gold_toks:
        return 1.0 if not pred_toks else 0.0
    if not pred_toks:
        return 0.0
    overlap = len(pred_toks & gold_toks)
    precision = overlap / len(pred_toks)
    recall    = overlap / len(gold_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def grade(action: ExtractAction, label: dict) -> Reward:
    breakdown: dict = {}
    weighted_score = 0.0

    # ---- Text fields (token F1) ----------------------------------------
    for field in ("customer_name", "issue_summary", "product_mentioned", "resolution_needed"):
        pred  = getattr(action, field, "") or ""
        gold  = label.get(field, "") or ""
        f1    = _token_f1(pred, gold)
        w     = _FIELD_WEIGHTS[field]
        weighted_score += w * f1
        breakdown[f"{field}_f1"]   = round(f1, 4)
        breakdown[f"{field}_gold"] = gold

    # ---- Effort (exact match, 2-level partial) --------------------------
    effort_order   = [EffortLevel.QUICK, EffortLevel.MEDIUM, EffortLevel.EXTENSIVE]
    pred_eff       = action.estimated_effort
    gold_eff       = EffortLevel(label["estimated_effort"])
    dist           = abs(effort_order.index(pred_eff) - effort_order.index(gold_eff))
    effort_score   = 1.0 if dist == 0 else (0.5 if dist == 1 else 0.0)
    w              = _FIELD_WEIGHTS["estimated_effort"]
    weighted_score += w * effort_score
    breakdown["effort_score"]         = round(effort_score, 4)
    breakdown["effort_predicted"]     = pred_eff.value
    breakdown["effort_gold"]          = gold_eff.value

    return Reward(
        score=round(min(weighted_score, 1.0), 4),
        breakdown=breakdown,
        penalty=0.0,
        message=f"Weighted extraction F1: {weighted_score:.3f}"
    )
