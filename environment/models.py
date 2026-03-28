"""
Typed Pydantic models for the Customer Support Email Triage OpenEnv environment.
All models follow the OpenEnv spec: Observation, Action, Reward.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TaskID(str, Enum):
    EMAIL_CLASSIFY   = "email_classify"      # Easy
    TICKET_EXTRACT   = "ticket_extraction"   # Medium
    RESPONSE_DRAFT   = "response_drafting"   # Hard


class EmailCategory(str, Enum):
    BILLING    = "billing"
    TECHNICAL  = "technical"
    SHIPPING   = "shipping"
    GENERAL    = "general"
    SPAM       = "spam"


class UrgencyLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class EffortLevel(str, Enum):
    QUICK     = "quick"
    MEDIUM    = "medium"
    EXTENSIVE = "extensive"


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class EmailMessage(BaseModel):
    """A single email message."""
    message_id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CustomerHistory(BaseModel):
    """Snapshot of a customer's account history."""
    customer_id: str
    account_age_days: int
    total_orders: int
    open_tickets: int
    tier: str  # "free" | "pro" | "enterprise"


class PolicySnippet(BaseModel):
    """A company policy excerpt relevant to the current scenario."""
    title: str
    content: str


# ---------------------------------------------------------------------------
# Observation models (what the agent sees)
# ---------------------------------------------------------------------------

class ClassifyObservation(BaseModel):
    """Observation for Task 1 – Email Classification."""
    task: TaskID = TaskID.EMAIL_CLASSIFY
    email: EmailMessage
    step_index: int
    total_steps: int
    score_so_far: float = 0.0


class ExtractObservation(BaseModel):
    """Observation for Task 2 – Ticket Metadata Extraction."""
    task: TaskID = TaskID.TICKET_EXTRACT
    thread: List[EmailMessage]          # 1-3 messages in a thread
    step_index: int
    total_steps: int
    score_so_far: float = 0.0


class DraftObservation(BaseModel):
    """Observation for Task 3 – Response Drafting."""
    task: TaskID = TaskID.RESPONSE_DRAFT
    thread: List[EmailMessage]
    customer_history: CustomerHistory
    policy_snippets: List[PolicySnippet]
    step_index: int
    total_steps: int
    score_so_far: float = 0.0


# Discriminated union for the API
Observation = ClassifyObservation | ExtractObservation | DraftObservation


# ---------------------------------------------------------------------------
# Action models (what the agent does)
# ---------------------------------------------------------------------------

class ClassifyAction(BaseModel):
    """Action for Task 1 – label the email."""
    task: TaskID = TaskID.EMAIL_CLASSIFY
    category: EmailCategory
    urgency: UrgencyLevel
    reasoning: Optional[str] = None   # optional chain-of-thought


class ExtractAction(BaseModel):
    """Action for Task 2 – extract structured metadata."""
    task: TaskID = TaskID.TICKET_EXTRACT
    customer_name: str
    issue_summary: str                 # ≤50 words
    product_mentioned: str
    resolution_needed: str             # ≤30 words
    estimated_effort: EffortLevel
    reasoning: Optional[str] = None


class DraftAction(BaseModel):
    """Action for Task 3 – draft a support response."""
    task: TaskID = TaskID.RESPONSE_DRAFT
    response_subject: str
    response_body: str                 # the email body sent to the customer
    internal_note: str                 # internal CRM note
    escalate: bool = False             # should this be escalated to a senior agent?
    reasoning: Optional[str] = None


# Discriminated union
Action = ClassifyAction | ExtractAction | DraftAction


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Structured reward returned by step()."""
    score: float = Field(..., ge=0.0, le=1.0, description="Per-step score 0.0–1.0")
    breakdown: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sub-scores and metadata per criterion"
    )
    penalty: float = Field(
        default=0.0, ge=0.0,
        description="Penalty applied (e.g. for invalid / destructive actions)"
    )
    message: str = ""


# ---------------------------------------------------------------------------
# Step / State response models
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    observation: Optional[Dict[str, Any]]   # next obs (None if done)
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EpisodeState(BaseModel):
    task: TaskID
    step_index: int
    total_steps: int
    cumulative_score: float
    done: bool
    history: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Reset request
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: TaskID = TaskID.EMAIL_CLASSIFY
    seed: Optional[int] = None
