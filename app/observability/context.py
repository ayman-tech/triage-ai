"""Per-request context for tracing and cost attribution."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass
class ActiveRun:
    run_id: str
    company_id: str
    trace_id: str | None = None
    case_id: str | None = None
    sequence: int = 0

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence


@dataclass
class ActiveStep:
    node_name: str
    sequence_number: int
    retry_number: int = 0


_active: ContextVar[ActiveRun | None] = ContextVar("workflow_active_run", default=None)
_active_step: ContextVar[ActiveStep | None] = ContextVar("workflow_active_step", default=None)


def get_active_run() -> ActiveRun | None:
    return _active.get()


def set_active_run(active: ActiveRun) -> Token:
    return _active.set(active)


def reset_active_run(token: Token) -> None:
    _active.reset(token)


def get_active_step() -> ActiveStep | None:
    return _active_step.get()


def set_active_step(active_step: ActiveStep) -> Token:
    return _active_step.set(active_step)


def reset_active_step(token: Token) -> None:
    _active_step.reset(token)


def set_case_id(case_id: str) -> None:
    ar = _active.get()
    if ar is not None:
        ar.case_id = case_id


def set_trace_id(trace_id: str) -> None:
    ar = _active.get()
    if ar is not None:
        ar.trace_id = trace_id
