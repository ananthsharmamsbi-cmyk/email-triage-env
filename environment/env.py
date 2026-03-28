"""
Core EmailTriageEnv — manages episode state, dispatches to task graders.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from environment.models import (
    Action, ClassifyAction, ExtractAction, DraftAction,
    ClassifyObservation, ExtractObservation, DraftObservation,
    CustomerHistory, EmailMessage, PolicySnippet,
    EpisodeState, Reward, StepResponse, TaskID, ResetRequest,
)
from environment.tasks import task1_classify, task2_extract, task3_respond

_DATA_PATH = Path(__file__).parent.parent / "data" / "emails.json"

# Number of scenarios presented per episode per task
_EPISODE_SIZE = {
    TaskID.EMAIL_CLASSIFY: 5,
    TaskID.TICKET_EXTRACT: 3,
    TaskID.RESPONSE_DRAFT: 2,
}


class EmailTriageEnv:
    """
    OpenEnv-compliant Customer Support Email Triage environment.

    Episodes:
      - Task 1 (Easy):   Classify N=5 individual emails.
      - Task 2 (Medium): Extract metadata from N=3 email threads.
      - Task 3 (Hard):   Draft responses to N=2 complex complaints.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        llm_model: str = "gpt-4o-mini",
    ):
        self._data        = self._load_data()
        self._llm_client  = llm_client
        self._llm_model   = llm_model
        self._state: Optional[EpisodeState]    = None
        self._scenarios:   List[dict]          = []
        self._current_idx: int                 = 0

    # ------------------------------------------------------------------
    # Public OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, request: ResetRequest) -> Dict[str, Any]:
        """Start a fresh episode for the given task. Returns the first observation."""
        rng = random.Random(request.seed)
        task = request.task
        pool = self._data.get(self._task_key(task), [])

        n = _EPISODE_SIZE[task]
        if len(pool) <= n:
            scenarios = pool[:]
        else:
            scenarios = rng.sample(pool, n)

        self._scenarios   = scenarios
        self._current_idx = 0
        self._state = EpisodeState(
            task=task,
            step_index=0,
            total_steps=len(scenarios),
            cumulative_score=0.0,
            done=False,
            history=[],
        )

        obs = self._build_observation(task, self._scenarios[0], 0)
        return obs.model_dump()

    def step(self, action_dict: Dict[str, Any]) -> StepResponse:
        """Process one agent action. Returns StepResponse."""
        if self._state is None or self._state.done:
            raise RuntimeError("Call reset() before step(), or episode is done.")

        task  = self._state.task
        idx   = self._current_idx
        scenario = self._scenarios[idx]

        # Parse + grade action
        action = self._parse_action(task, action_dict)
        reward = self._grade(task, action, scenario)

        # Update state
        self._state.cumulative_score += reward.score
        self._state.step_index        = idx + 1
        self._state.history.append({
            "step": idx,
            "action": action_dict,
            "reward": reward.model_dump(),
        })
        self._current_idx += 1

        if self._current_idx >= len(self._scenarios):
            self._state.done = True

        # Build next observation (or None if done)
        if not self._state.done:
            next_scenario = self._scenarios[self._current_idx]
            next_obs       = self._build_observation(
                task, next_scenario, self._current_idx
            )
            obs_dict = next_obs.model_dump()
        else:
            obs_dict = None

        avg_score = self._state.cumulative_score / self._state.step_index

        return StepResponse(
            observation=obs_dict,
            reward=reward,
            done=self._state.done,
            info={
                "step": self._state.step_index,
                "total": self._state.total_steps,
                "cumulative_score": round(self._state.cumulative_score, 4),
                "average_score":    round(avg_score, 4),
                "task": task.value,
            },
        )

    def state(self) -> EpisodeState:
        """Return current episode state."""
        if self._state is None:
            raise RuntimeError("No active episode. Call reset() first.")
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> Dict[str, Any]:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _task_key(self, task: TaskID) -> str:
        return {
            TaskID.EMAIL_CLASSIFY: "task1_classify",
            TaskID.TICKET_EXTRACT: "task2_extract",
            TaskID.RESPONSE_DRAFT: "task3_respond",
        }[task]

    def _parse_action(self, task: TaskID, action_dict: Dict[str, Any]) -> Action:
        # Inject task if not present
        action_dict = {**action_dict, "task": task.value}
        if task == TaskID.EMAIL_CLASSIFY:
            return ClassifyAction(**action_dict)
        elif task == TaskID.TICKET_EXTRACT:
            return ExtractAction(**action_dict)
        else:
            return DraftAction(**action_dict)

    def _grade(self, task: TaskID, action: Action, scenario: dict) -> Reward:
        label = scenario["label"]
        if task == TaskID.EMAIL_CLASSIFY:
            return task1_classify.grade(action, label)  # type: ignore
        elif task == TaskID.TICKET_EXTRACT:
            return task2_extract.grade(action, label)   # type: ignore
        else:
            thread_text = "\n\n".join(
                f"From: {m['sender']}\nSubject: {m['subject']}\n{m['body']}"
                for m in scenario.get("thread", [])
            )
            return task3_respond.grade(
                action,          # type: ignore
                label,
                thread_text=thread_text,
                llm_client=self._llm_client,
                llm_model=self._llm_model,
            )

    def _build_observation(
        self, task: TaskID, scenario: dict, idx: int
    ) -> ClassifyObservation | ExtractObservation | DraftObservation:
        total       = len(self._scenarios)
        score_so_far = (
            self._state.cumulative_score / idx
            if idx > 0 else 0.0
        )

        if task == TaskID.EMAIL_CLASSIFY:
            return ClassifyObservation(
                email=EmailMessage(**scenario["email"]),
                step_index=idx,
                total_steps=total,
                score_so_far=round(score_so_far, 4),
            )

        elif task == TaskID.TICKET_EXTRACT:
            thread = [EmailMessage(**m) for m in scenario["thread"]]
            return ExtractObservation(
                thread=thread,
                step_index=idx,
                total_steps=total,
                score_so_far=round(score_so_far, 4),
            )

        else:  # RESPONSE_DRAFT
            thread  = [EmailMessage(**m) for m in scenario.get("thread", [])]
            history = CustomerHistory(**scenario["customer_history"])
            policies = [PolicySnippet(**p) for p in scenario.get("policy_snippets", [])]
            return DraftObservation(
                thread=thread,
                customer_history=history,
                policy_snippets=policies,
                step_index=idx,
                total_steps=total,
                score_so_far=round(score_so_far, 4),
            )
