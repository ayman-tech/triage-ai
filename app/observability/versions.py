"""Version strings stamped onto traces and workflow_runs."""

from __future__ import annotations

import os

from app.agents.llm_factory import default_model_name, get_provider


def workflow_version() -> str:
    return os.getenv("WORKFLOW_VERSION", "1.0.0")


def prompt_bundle_version() -> str:
    return os.getenv("PROMPT_BUNDLE_VERSION", "default")


def knowledge_pack_version() -> str:
    from app.knowledge.mock_company_pack import deployment_label

    return os.getenv("KNOWLEDGE_PACK_VERSION") or f"pack:{deployment_label()}"


def default_chat_model() -> str:
    return default_model_name()


def llm_provider() -> str:
    return get_provider()
