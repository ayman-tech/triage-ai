"""ASGI entry point for the complaint-processing API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import router as api_router
from app.db.session import init_db
from app.observability.logging import setup_logging
from app.observability.tracing import setup_tracing
from app.ui import router as ui_router

logger = logging.getLogger(__name__)


def _configure_langsmith() -> None:
    """Enable LangSmith tracing for Google ADK agents via the native integration."""
    tracing_on = (
        os.getenv("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")
        or os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes")
    )
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "(default project)"
    if api_key and not os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = api_key
    if project != "(default project)" and not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = project
    has_key = bool(api_key)

    if tracing_on and has_key:
        try:
            from langsmith.integrations.google_adk import configure_google_adk
            configure_google_adk(project_name=project if project != "(default project)" else None)
            logger.info("LangSmith ADK tracing enabled (project=%s)", project)
        except Exception as exc:
            logger.warning("LangSmith ADK integration failed to configure: %s", exc)
    elif tracing_on and not has_key:
        logger.warning(
            "LANGSMITH_TRACING is on but LANGSMITH_API_KEY is missing; "
            "LangSmith will not receive traces"
        )
    else:
        logger.info(
            "LangSmith off: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY "
            "to trace Google ADK agent calls in LangSmith"
        )


def _check_gemini_key() -> None:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        logger.warning(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set — "
            "LLM calls will fail at runtime"
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    _check_gemini_key()
    _configure_langsmith()
    setup_tracing()
    init_db()
    yield


app = FastAPI(
    title="Complaint classification agent",
    description="Google ADK + Gemini pipeline for consumer complaint intake, classification, risk, and resolution.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (CSS, JS)
static_dir = Path(__file__).resolve().parent / "app" / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# API routes
app.include_router(api_router)

# UI routes (HTML views)
app.include_router(ui_router)
