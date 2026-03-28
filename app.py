"""
FastAPI application exposing the OpenEnv API for the Customer Support Email Triage env.

Endpoints:
  GET  /               — health check
  GET  /tasks          — list available tasks with metadata
  POST /reset          — start a new episode
  POST /step           — submit one action, receive observation + reward
  GET  /state          — inspect current episode state
  GET  /openenv.yaml   — serve the spec file

OpenEnv spec: https://openenv.dev
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
from pydantic import BaseModel

from environment import EmailTriageEnv
from environment.models import (
    TaskID, ResetRequest, StepResponse, EpisodeState,
)

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Support Email Triage — OpenEnv",
    description=(
        "A real-world OpenEnv environment where AI agents learn to triage, "
        "extract metadata from, and respond to customer support emails."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Build one shared LLM client if credentials are available
def _make_llm_client() -> Optional[Any]:
    api_key  = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("API_BASE_URL", "")
    if not api_key:
        return None
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return OpenAI(**kwargs)
    except Exception:
        return None


_llm_client = _make_llm_client()
_llm_model  = os.getenv("MODEL_NAME", "gpt-4o-mini")
env         = EmailTriageEnv(llm_client=_llm_client, llm_model=_llm_model)

# ---------------------------------------------------------------------------
# Pydantic request/response wrappers
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    action: Dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def root():
    return {
        "status":  "ok",
        "env":     "email-triage",
        "version": "1.0.0",
        "tasks":   [t.value for t in TaskID],
    }


@app.get("/health", tags=["health"])
def health():
    """Health check endpoint — required by OpenEnv spec and HF Spaces monitoring."""
    return {"status": "healthy"}


@app.get("/tasks", tags=["tasks"])
def list_tasks():
    """Return metadata for all available tasks."""
    return {
        "tasks": [
            {
                "task_id":    "email_classify",
                "difficulty": "easy",
                "description": (
                    "Classify each incoming email by category "
                    "(billing / technical / shipping / general / spam) and urgency "
                    "(low / medium / high / critical)."
                ),
                "episode_steps": 5,
                "score_components": ["category (0.5)", "urgency (0.5, partial credit)"],
            },
            {
                "task_id":    "ticket_extraction",
                "difficulty": "medium",
                "description": (
                    "Extract structured metadata from a multi-message email thread: "
                    "customer name, issue summary, product mentioned, resolution needed, "
                    "and estimated effort."
                ),
                "episode_steps": 3,
                "score_components": ["token F1 per field", "weighted 0.0–1.0"],
            },
            {
                "task_id":    "response_drafting",
                "difficulty": "hard",
                "description": (
                    "Draft a professional customer support response to a complex complaint, "
                    "given the email thread, customer history, and policy snippets. "
                    "Scored on rule-based coverage, structural quality, and LLM-judged quality."
                ),
                "episode_steps": 2,
                "score_components": [
                    "rule-based must_include (25%)",
                    "LLM judge professionalism/empathy/accuracy/completeness (50%)",
                    "structural checks (25%)",
                ],
            },
        ]
    }


@app.post("/reset", response_model=Dict[str, Any], tags=["openenv"])
def reset(request: ResetRequest):
    """
    Start a fresh episode.

    Body:
      - task: one of email_classify | ticket_extraction | response_drafting
      - seed: (optional) integer for reproducibility
    """
    try:
        obs = env.reset(request)
        return obs
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResponse, tags=["openenv"])
def step(request: StepRequest):
    """
    Submit one action and receive the next observation, reward, done flag, and info.

    The action shape depends on the active task:

    **email_classify:**
    ```json
    {"category": "billing", "urgency": "high"}
    ```

    **ticket_extraction:**
    ```json
    {
      "customer_name": "Jane Doe",
      "issue_summary": "User cannot log in due to 2FA failure",
      "product_mentioned": "Account Authentication",
      "resolution_needed": "Restore account access",
      "estimated_effort": "quick"
    }
    ```

    **response_drafting:**
    ```json
    {
      "response_subject": "Re: Your recent billing concern",
      "response_body": "Dear John, ...",
      "internal_note": "Verified charge, issuing refund",
      "escalate": false
    }
    ```
    """
    try:
        response = env.step(request.action)
        return response
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", response_model=EpisodeState, tags=["openenv"])
def state():
    """Return the current episode state (step index, cumulative score, history, done flag)."""
    try:
        return env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/metadata", tags=["openenv"])
def metadata():
    """OpenEnv required: returns environment name and description."""
    return {
        "name":        "email-triage-env",
        "version":     "1.0.0",
        "description": (
            "A real-world OpenEnv environment simulating customer support email triage. "
            "Agents classify emails, extract ticket metadata, and draft professional responses."
        ),
        "author":  "ScalarHackathon Team",
        "tasks":   [t.value for t in TaskID],
        "license": "MIT",
    }


@app.get("/schema", tags=["openenv"])
def schema():
    """OpenEnv required: returns action, observation, and state schemas."""
    return {
        "action": {
            "email_classify": {
                "type": "object",
                "required": ["category", "urgency"],
                "properties": {
                    "category": {"type": "string", "enum": ["billing", "technical", "shipping", "general", "spam"]},
                    "urgency":  {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "reasoning": {"type": "string"},
                }
            },
            "ticket_extraction": {
                "type": "object",
                "required": ["customer_name", "issue_summary", "product_mentioned", "resolution_needed", "estimated_effort"],
                "properties": {
                    "customer_name":     {"type": "string"},
                    "issue_summary":     {"type": "string"},
                    "product_mentioned": {"type": "string"},
                    "resolution_needed": {"type": "string"},
                    "estimated_effort":  {"type": "string", "enum": ["quick", "medium", "extensive"]},
                    "reasoning":         {"type": "string"},
                }
            },
            "response_drafting": {
                "type": "object",
                "required": ["response_subject", "response_body", "internal_note", "escalate"],
                "properties": {
                    "response_subject": {"type": "string"},
                    "response_body":    {"type": "string"},
                    "internal_note":    {"type": "string"},
                    "escalate":         {"type": "boolean"},
                    "reasoning":        {"type": "string"},
                }
            }
        },
        "observation": {
            "type": "object",
            "properties": {
                "task":        {"type": "string"},
                "step_index":  {"type": "integer"},
                "total_steps": {"type": "integer"},
                "score_so_far":{"type": "number"},
                "email":       {"type": "object", "description": "Present in email_classify task"},
                "thread":      {"type": "array",  "description": "Present in ticket_extraction and response_drafting tasks"},
                "customer_history":  {"type": "object", "description": "Present in response_drafting task"},
                "policy_snippets":   {"type": "array",  "description": "Present in response_drafting task"},
            }
        },
        "state": {
            "type": "object",
            "properties": {
                "task":             {"type": "string"},
                "step_index":       {"type": "integer"},
                "total_steps":      {"type": "integer"},
                "cumulative_score": {"type": "number"},
                "done":             {"type": "boolean"},
                "history":          {"type": "array"},
            }
        }
    }


@app.post("/mcp", tags=["openenv"])
async def mcp_endpoint(request: Request):
    """OpenEnv required: MCP (Model Context Protocol) JSON-RPC 2.0 endpoint."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    method  = body.get("method", "")
    req_id  = body.get("id", 1)
    jsonrpc = "2.0"

    # Handle standard MCP methods
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "email-triage-env", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "reset",
                    "description": "Start a new episode for a given task",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "enum": ["email_classify", "ticket_extraction", "response_drafting"]},
                            "seed": {"type": "integer"},
                        },
                        "required": ["task"],
                    }
                },
                {
                    "name": "step",
                    "description": "Submit one action and receive reward + next observation",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"action": {"type": "object"}},
                        "required": ["action"],
                    }
                },
            ]
        }
    else:
        result = {"message": "ok", "env": "email-triage-env"}

    return JSONResponse({"jsonrpc": jsonrpc, "id": req_id, "result": result})


@app.get("/openenv.yaml", tags=["spec"])
def serve_yaml():
    """Serve the openenv.yaml specification file."""
    yaml_path = Path(__file__).parent / "openenv.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    return FileResponse(str(yaml_path), media_type="text/yaml")


# ---------------------------------------------------------------------------
# Entry point (for local development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )
