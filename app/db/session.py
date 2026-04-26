"""Database engine, session factory, and pgvector extension bootstrap."""

from __future__ import annotations

import logging
import os
import uuid

from dotenv import load_dotenv

load_dotenv()
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base
from app.utils.case_ids import ensure_case_public_id

logger = logging.getLogger(__name__)

# ── Connection URL ───────────────────────────────────────────────────────────
# Default points to a local Postgres with pgvector installed.
# Override via the DATABASE_URL environment variable.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/complaints",
)

_connect_args: dict = {}
if DATABASE_URL.startswith("postgresql"):
    _connect_args["connect_timeout"] = int(os.getenv("PG_CONNECT_TIMEOUT", "5"))

engine = create_engine(
    DATABASE_URL,
    echo=bool(os.getenv("SQL_ECHO", "")),
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    connect_args=_connect_args,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Enable the pgvector extension and create all tables (idempotent).

    Must be called once at application startup (or during migrations).
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        logger.info("pgvector extension ensured")

    # Create tables first — must exist before migration block runs
    Base.metadata.create_all(bind=engine)
    logger.info("All database tables created / verified")

    # When the demo repo evolves, new columns may be added to existing
    # tables. Since this project doesn't ship with Alembic migrations,
    # we do safe "add column if missing" for the known schema upgrades.
    with engine.connect() as conn:
        complaint_case_columns: list[tuple[str, str]] = [
            ("external_schema_json", "TEXT"),
            ("public_case_id", "VARCHAR(16)"),
            ("operational_mapping_json", "TEXT"),
            ("evidence_trace_json", "TEXT"),
            ("severity_class", "VARCHAR(40)"),
            ("team_assignment", "VARCHAR(120)"),
            ("sla_class", "VARCHAR(40)"),
            ("root_cause_hypothesis_json", "TEXT"),
            ("compliance_flags_json", "TEXT"),
            ("review_notes", "TEXT"),
            ("routed_to", "VARCHAR(120)"),
            ("jira_issue_key", "VARCHAR(32)"),
            ("jira_issue_url", "TEXT"),
            ("classification_audit_json", "TEXT"),
            ("user_id", "VARCHAR(64)"),
            ("intake_session_transcript_json", "TEXT"),
            ("document_gate_result_json", "TEXT"),
            ("document_consistency_json", "TEXT"),
            ("token_total", "INTEGER"),
            ("cost_estimate_usd", "DOUBLE PRECISION"),
        ]

        classification_columns: list[tuple[str, str]] = [
            ("review_recommended", "BOOLEAN DEFAULT FALSE"),
            ("reason_codes_json", "TEXT"),
            ("keywords_json", "TEXT"),
        ]
        case_document_columns: list[tuple[str, str]] = [
            ("case_id", "VARCHAR(32)"),
            ("intake_session_id", "VARCHAR(32)"),
            ("user_id", "VARCHAR(64)"),
            ("original_filename", "VARCHAR(255)"),
            ("mime_type", "VARCHAR(120)"),
            ("size_bytes", "INTEGER DEFAULT 0"),
            ("storage_uri", "TEXT"),
            ("checksum", "VARCHAR(128)"),
            ("upload_status", "VARCHAR(32) DEFAULT 'uploaded'"),
            ("parser_status", "VARCHAR(32) DEFAULT 'pending'"),
            ("extraction_status", "VARCHAR(32) DEFAULT 'pending'"),
            ("document_type", "VARCHAR(64) DEFAULT 'unknown'"),
            ("processing_error", "TEXT"),
            ("created_at", "TIMESTAMP"),
            ("updated_at", "TIMESTAMP"),
        ]
        document_artifact_columns: list[tuple[str, str]] = [
            ("document_id", "VARCHAR(32)"),
            ("raw_text", "TEXT"),
            ("normalized_text", "TEXT"),
            ("extracted_json", "TEXT"),
            ("parser_version", "VARCHAR(64)"),
            ("extraction_version", "VARCHAR(64)"),
            ("confidence", "FLOAT"),
            ("created_at", "TIMESTAMP"),
            ("updated_at", "TIMESTAMP"),
        ]
        document_embedding_columns: list[tuple[str, str]] = [
            ("document_id", "VARCHAR(32)"),
            ("case_id", "VARCHAR(32)"),
            ("chunk_index", "INTEGER DEFAULT 0"),
            ("content", "TEXT"),
            ("source_page", "INTEGER"),
            ("created_at", "TIMESTAMP"),
        ]
        source_dataset_columns: list[tuple[str, str]] = [
            ("name", "VARCHAR(160)"),
            ("company_id", "VARCHAR(64)"),
            ("source_type", "VARCHAR(64)"),
            ("description", "TEXT"),
            ("version", "VARCHAR(32) DEFAULT 'v1'"),
            ("status", "VARCHAR(32) DEFAULT 'active'"),
            ("sampling_strategy_json", "TEXT"),
            ("stats_json", "TEXT"),
            ("created_at", "TIMESTAMP"),
            ("updated_at", "TIMESTAMP"),
        ]
        source_dataset_item_columns: list[tuple[str, str]] = [
            ("dataset_id", "VARCHAR(32)"),
            ("external_id", "VARCHAR(64)"),
            ("split", "VARCHAR(32) DEFAULT 'evaluation'"),
            ("consumer_narrative", "TEXT"),
            ("product", "VARCHAR(120)"),
            ("sub_product", "VARCHAR(120)"),
            ("issue", "VARCHAR(120)"),
            ("sub_issue", "VARCHAR(120)"),
            ("company", "VARCHAR(200)"),
            ("state", "VARCHAR(2)"),
            ("submitted_via", "VARCHAR(20)"),
            ("date_received", "VARCHAR(16)"),
            ("company_response", "TEXT"),
            ("company_public_response", "TEXT"),
            ("metadata_json", "TEXT"),
            ("created_at", "TIMESTAMP"),
        ]
        knowledge_collection_columns: list[tuple[str, str]] = [
            ("company_id", "VARCHAR(64)"),
            ("source_dataset_id", "VARCHAR(32)"),
            ("name", "VARCHAR(160)"),
            ("knowledge_type", "VARCHAR(64)"),
            ("description", "TEXT"),
            ("version", "VARCHAR(32) DEFAULT 'v1'"),
            ("status", "VARCHAR(32) DEFAULT 'active'"),
            ("metadata_json", "TEXT"),
            ("created_at", "TIMESTAMP"),
            ("updated_at", "TIMESTAMP"),
        ]
        knowledge_entry_columns: list[tuple[str, str]] = [
            ("collection_id", "VARCHAR(32)"),
            ("title", "VARCHAR(200)"),
            ("content", "TEXT"),
            ("metadata_json", "TEXT"),
            ("created_at", "TIMESTAMP"),
            ("updated_at", "TIMESTAMP"),
        ]
        evaluation_dataset_columns: list[tuple[str, str]] = [
            ("source_dataset_id", "VARCHAR(32)"),
        ]
        workflow_run_columns: list[tuple[str, str]] = [
            ("llm_call_count", "INTEGER"),
        ]
        workflow_step_columns: list[tuple[str, str]] = [
            ("llm_call_count", "INTEGER"),
            ("prompt_tokens", "INTEGER"),
            ("completion_tokens", "INTEGER"),
            ("token_total", "INTEGER"),
            ("cost_estimate_usd", "DOUBLE PRECISION"),
        ]

        for table_name, columns in (
            ("complaint_cases", complaint_case_columns),
            ("classifications", classification_columns),
            ("case_documents", case_document_columns),
            ("document_artifacts", document_artifact_columns),
            ("document_embeddings", document_embedding_columns),
            ("source_datasets", source_dataset_columns),
            ("source_dataset_items", source_dataset_item_columns),
            ("knowledge_collections", knowledge_collection_columns),
            ("knowledge_entries", knowledge_entry_columns),
            ("evaluation_datasets", evaluation_dataset_columns),
            ("workflow_runs", workflow_run_columns),
            ("workflow_steps", workflow_step_columns),
        ):
            existing = conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    AND column_name = ANY(:cols)
                    """
                ),
                {"table_name": table_name, "cols": [c[0] for c in columns]},
            ).fetchall()
            existing_cols = {r[0] for r in existing}

            for col_name, col_type in columns:
                if col_name in existing_cols:
                    continue
                logger.info("Adding missing column to %s: %s", table_name, col_name)
                conn.execute(
                    text(
                        f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
                    )
                )
        conn.commit()

    try:
        from app.observability.backfill import backfill_cost_ledger_from_workflow_runs

        inserted = backfill_cost_ledger_from_workflow_runs()
        if inserted:
            logger.info("Historical cost ledger backfill inserted %d rows", inserted)
    except Exception as exc:
        logger.warning("Historical cost ledger backfill skipped: %s", exc)

    # Seed default user accounts (idempotent — skips if already present)
    _seed_default_users()
    _backfill_public_case_ids()


def _seed_default_users() -> None:
    """Insert the default admin, user, and team accounts if they don't exist."""
    from app.db.models import UserAccount  # local import to avoid circular deps

    # (email_local, team_assignment_value)
    # team_assignment_value must match exactly what routing.py / mock_company_pack writes
    # into complaint_cases.team_assignment.
    _TEAMS: list[tuple[str, str]] = [
        ("executivecomplaints",  "executive_complaints_team"),
        ("payments",             "payments_team"),
        ("debtcollection",       "debt_collection_team"),
        ("managementescalation", "management_escalation_team"),
        ("generalcomplaints",    "general_complaints_team"),
        ("studentloanservicing", "student_loan_servicing_team"),
        ("fraudaccessops",       "fraud_and_access_ops_team"),
        ("consumerlending",      "consumer_lending_team"),
        ("autoloan",             "auto_loan_team"),
        ("creditreporting",      "credit_reporting_team"),
        ("mortgageservicing",    "mortgage_servicing_team"),
        ("creditcard",           "credit_card_operations_team"),
    ]

    seeds = [
        {
            "email": "admin@triage.ai",
            "password": "admin123",
            "role": "admin",
            "company": None,
            "user_id": "admin-001",
        },
        {
            "email": "user@triage.ai",
            "password": "user123",
            "role": "user",
            "company": "Mock Bank",
            "user_id": "user-001",
        },
        *[
            {
                "email": f"{local}@triage.ai",
                "password": f"{local}123",
                "role": "team",
                "company": team_assignment_value,
                "user_id": f"team-{local}",
            }
            for local, team_assignment_value in _TEAMS
        ],
    ]

    session = SessionLocal()
    try:
        for s in seeds:
            exists = session.query(UserAccount).filter(UserAccount.email == s["email"]).first()
            if not exists:
                session.add(UserAccount(
                    id=uuid.uuid4().hex,
                    email=s["email"],
                    password=s["password"],
                    role=s["role"],
                    company=s["company"],
                    user_id=s["user_id"],
                ))
                logger.info("Seeded default user: %s", s["email"])
            elif exists.company != s["company"]:
                # Fix stale team_assignment values from previous seeds
                exists.company = s["company"]
                logger.info("Updated company/team for %s → %s", s["email"], s["company"])
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _backfill_public_case_ids() -> None:
    """Assign CASE00001-style public ids to complaint rows that predate the field."""
    from app.db.models import ComplaintCase

    session = SessionLocal()
    try:
        rows = (
            session.query(ComplaintCase)
            .filter(ComplaintCase.public_case_id.is_(None))
            .order_by(ComplaintCase.created_at.asc(), ComplaintCase.id.asc())
            .all()
        )
        changed = False
        for row in rows:
            ensure_case_public_id(session, row)
            changed = True
        if changed:
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()



@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Yield a transactional DB session and handle cleanup."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
