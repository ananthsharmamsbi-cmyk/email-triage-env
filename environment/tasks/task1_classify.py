"""
Task 1 (Easy) — Email Classify Grader
Scores 0.0–1.0:  0.5 for correct category + 0.5 for correct urgency (partial if within 1 tier)
"""
from __future__ import annotations
from environment.models import ClassifyAction, EmailCategory, UrgencyLevel, Reward

# Ordinal order for urgency (used to measure "distance")
_URGENCY_ORDER = [UrgencyLevel.LOW, UrgencyLevel.MEDIUM, UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]


def grade(action: ClassifyAction, label: dict) -> Reward:
    """Return a Reward for a classification action compared to a ground-truth label."""
    expected_cat  = EmailCategory(label["category"])
    expected_urg  = UrgencyLevel(label["urgency"])

    # --- Category score (0.0 or 0.5) -----------------------------------
    cat_score = 0.5 if action.category == expected_cat else 0.0

    # --- Urgency score (0.0, 0.25, or 0.5) based on distance -----------
    pred_idx = _URGENCY_ORDER.index(action.urgency)
    true_idx = _URGENCY_ORDER.index(expected_urg)
    dist     = abs(pred_idx - true_idx)

    if dist == 0:
        urg_score = 0.5
    elif dist == 1:
        urg_score = 0.25   # partial credit for being 1 tier off
    else:
        urg_score = 0.0

    total = cat_score + urg_score

    return Reward(
        score=round(total, 4),
        breakdown={
            "category":         round(cat_score, 4),
            "urgency":          round(urg_score, 4),
            "expected_category": expected_cat.value,
            "predicted_category": action.category.value,
            "expected_urgency":  expected_urg.value,
            "predicted_urgency": action.urgency.value,
        },
        penalty=0.0,
        message=(
            f"Category: {'✓' if cat_score == 0.5 else '✗'} | "
            f"Urgency: {'✓' if urg_score == 0.5 else ('~' if urg_score == 0.25 else '✗')}"
        )
    )
