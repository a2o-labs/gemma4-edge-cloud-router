"""Multi-cloud routing — pick a cloud model based on the task type.

This is a thin policy layer that returns a model identifier (string)
given a CompactSchemaV15 envelope. Down the line, V15Pipeline will use
this to dispatch to one of several pre-loaded SoftPromptAdapter
instances — one per cloud model. For now the class is standalone and
returns model IDs only.

Default policy:
    task_type=code      -> claude-opus-4-7         (best at code synthesis)
    task_type=reason    -> claude-opus-4-7         (best at reasoning)
    task_type=summarize -> google/gemma-4-31B-it   (cheap + fast)
    task_type=translate -> google/gemma-4-31B-it
    task_type=qa        -> google/gemma-4-31B-it
    task_type=chat      -> google/gemma-4-31B-it
    task_type=other     -> google/gemma-4-31B-it   (default)

Override the entire policy by passing a dict to __init__.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .schema_v15 import CompactSchemaV15

TaskType = Literal["qa", "code", "summarize", "translate", "reason", "chat", "other"]

DEFAULT_POLICY: dict[str, str] = {
    "code": "claude-opus-4-7",
    "reason": "claude-opus-4-7",
    "summarize": "google/gemma-4-31B-it",
    "translate": "google/gemma-4-31B-it",
    "qa": "google/gemma-4-31B-it",
    "chat": "google/gemma-4-31B-it",
    "other": "google/gemma-4-31B-it",
}


@dataclass
class RouteDecision:
    """Result of a routing decision."""

    cloud_model: str
    task_type: str
    reason: str


class CloudRouter:
    """Pick a cloud model from a CompactSchemaV15."""

    def __init__(
        self,
        policy: dict[str, str] | None = None,
        complexity_override: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            policy: task_type -> cloud_model mapping. Defaults to DEFAULT_POLICY.
            complexity_override: if a schema has complexity="heavy",
                map task_type -> cloud_model from this dict instead of policy.
                Useful for routing heavy prompts to a more capable model.
                Example: {"qa": "claude-opus-4-7"}.
        """
        self.policy = dict(DEFAULT_POLICY)
        if policy:
            self.policy.update(policy)
        self.complexity_override = (
            dict(complexity_override) if complexity_override else {}
        )

    def select(self, schema: CompactSchemaV15) -> RouteDecision:
        tt = schema.task_type
        if schema.complexity == "heavy" and tt in self.complexity_override:
            return RouteDecision(
                cloud_model=self.complexity_override[tt],
                task_type=tt,
                reason=f"heavy override for {tt}",
            )
        cloud_model = self.policy.get(tt, self.policy["other"])
        return RouteDecision(
            cloud_model=cloud_model,
            task_type=tt,
            reason=f"task_type={tt}",
        )

    def __repr__(self) -> str:
        return f"CloudRouter(policy={list(self.policy.keys())})"
