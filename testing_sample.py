# Batch pipeline test: load N CFPB-style CSV rows and run ``process_complaint`` on each.

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None  # type: ignore[assignment, misc]

if load_dotenv:
    load_dotenv()

try:
    from app.observability.logging import setup_logging
    from app.observability.tracing import setup_tracing
except ImportError:
    setup_logging = None  # type: ignore[assignment, misc]
    setup_tracing = None  # type: ignore[assignment, misc]

try:
    import pandas as pd
except ModuleNotFoundError as e:
    raise RuntimeError(
        "pandas is required. Install in this environment, e.g. "
        "'./.venv/bin/python -m pip install pandas'"
    ) from e

from app.knowledge.mock_company_pack import deployment_label
from app.orchestrator.workflow import process_complaint

CSV_PATH = os.getenv("TEST_CSV_PATH", "complaints.csv")
OUTPUT_CSV = os.getenv(
    "TEST_PIPELINE_OUTPUT_CSV",
    "testing_sample_pipeline_output.csv",
)
SAMPLE_COUNT = max(1, 5 )#int(os.getenv("TEST_SAMPLE_COUNT", "10")))


def get_first_existing(df: pd.DataFrame, col_candidates: list[str]) -> str | None:
    for c in col_candidates:
        if c in df.columns:
            return c
    return None


def _safe_json(obj: object) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, default=str)
    except TypeError:
        return str(obj)


def build_row_record(
    payload: dict,
    case: object,
    *,
    source_csv: str,
    sample_index: int,
    pipeline_error: str | None = None,
) -> dict[str, str | int | None]:
    """Flatten CaseRead (or dict) into one CSV row."""
    if hasattr(case, "model_dump"):
        c = case.model_dump()
    elif isinstance(case, dict):
        c = case
    else:
        c = {}

    cls = c.get("classification") or {}
    risk = c.get("risk_assessment") or {}
    res = c.get("proposed_resolution") or {}
    rc = c.get("root_cause_hypothesis") or {}

    st = c.get("status", "")
    if hasattr(st, "value"):
        st = st.value

    return {
        "sample_index": sample_index,
        "run_at_utc": datetime.utcnow().isoformat() + "Z",
        "source_csv": source_csv,
        "deployment": deployment_label(),
        "consumer_narrative": (c.get("consumer_narrative") or payload.get("consumer_narrative")),
        "routed_to": c.get("routed_to"),
        "status": str(st) if st is not None else "",
        "classification_product_category": cls.get("product_category"),
        "classification_issue_type": cls.get("issue_type"),
        "classification_confidence": cls.get("confidence"),
        "risk_level": (risk.get("risk_level") if isinstance(risk, dict) else None),
        "risk_score": risk.get("risk_score") if isinstance(risk, dict) else None,
        "root_cause_summary": rc.get("summary") if isinstance(rc, dict) else _safe_json(rc),
        "resolution_action": res.get("recommended_action") if isinstance(res, dict) else None,
        "resolution_confidence": res.get("confidence") if isinstance(res, dict) else None,
        "compliance_flags_json": _safe_json(c.get("compliance_flags")),
        "review_notes": c.get("review_notes"),
        "classification_json": _safe_json(cls),
        "risk_assessment_json": _safe_json(risk),
        "proposed_resolution_json": _safe_json(res),
        "root_cause_json": _safe_json(rc),
        "evidence_trace_json": _safe_json(c.get("evidence_trace")),
        "pipeline_error": pipeline_error or "",
    }


def _cell_str(row: pd.Series, col: str | None) -> str | None:
    if not col or col not in row.index:
        return None
    val = row[col]
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s or None


def row_to_payload(
    row: pd.Series,
    *,
    col_narrative: str | None,
    col_product: str | None,
    col_sub_product: str | None,
    col_sub_issue: str | None,
    col_company: str | None,
    col_state: str | None,
    col_zip: str | None,
    col_channel: str | None,
    col_response: str | None,
    col_date_received: str | None,
    col_issue: str | None,
) -> dict:
    channel_raw = (
        str(row[col_channel]).strip() if col_channel else "web"
    ).lower()
    channel_map = {
        "web": "web",
        "online": "web",
        "phone": "phone",
        "email": "email",
        "fax": "fax",
        "postal": "postal",
        "mail": "postal",
        "referral": "referral",
    }
    channel = channel_map.get(channel_raw, "web")

    submitted_at = None
    if col_date_received:
        val = row[col_date_received]
        if pd.notna(val):
            try:
                submitted_at = pd.to_datetime(val, errors="coerce").to_pydatetime()
            except Exception:
                submitted_at = None

    if col_narrative:
        nar_raw = row[col_narrative]
        nar_str = "" if pd.isna(nar_raw) else str(nar_raw).strip()
    else:
        nar_str = ""

    product_str = _cell_str(row, col_product)
    sub_product_str = _cell_str(row, col_sub_product)
    issue_str = _cell_str(row, col_issue)
    sub_issue_str = _cell_str(row, col_sub_issue)

    has_rich_narrative = len(nar_str) >= 10
    has_cfpb_core = bool(product_str and issue_str)
    if not has_rich_narrative and not has_cfpb_core:
        raise ValueError(
            "Row needs either consumer_narrative (>=10 chars) or Product+Issue for CFPB structured path"
        )

    return {
        "consumer_narrative": nar_str if nar_str else None,
        "product": product_str,
        "sub_product": sub_product_str,
        "cfpb_product": product_str,
        "cfpb_sub_product": sub_product_str,
        "cfpb_issue": issue_str,
        "cfpb_sub_issue": sub_issue_str,
        "company": _cell_str(row, col_company),
        "state": _cell_str(row, col_state),
        "zip_code": _cell_str(row, col_zip),
        "channel": channel,
        "submitted_at": submitted_at.isoformat() if submitted_at else None,
        "external_product_category": product_str,
        "external_issue_type": issue_str,
        "requested_resolution": _cell_str(row, col_response),
    }


def main() -> None:
    if setup_logging:
        setup_logging()
    if setup_tracing:
        setup_tracing()

    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key_env = "DEEPSEEK_API_KEY" if llm_provider == "deepseek" else "OPENAI_API_KEY"
    api_key = os.getenv(api_key_env)
    print(f"LLM_PROVIDER: {llm_provider}")
    print(f"{api_key_env} set:", bool(api_key))
    print("CSV_PATH:", CSV_PATH)
    print("DEFAULT_COMPANY_ID:", DEFAULT_COMPANY_ID)
    print("SAMPLE_COUNT:", SAMPLE_COUNT)
    print("Pipeline output CSV:", OUTPUT_CSV)

    df = pd.read_csv(CSV_PATH)
    print("Loaded row columns:")
    print(list(df.columns))

    col_narrative = get_first_existing(
        df,
        [
            "Consumer complaint narrative",
            "consumer_narrative",
            "narrative",
        ],
    )
    col_product = get_first_existing(df, ["Product", "product"])
    col_sub_product = get_first_existing(df, ["Sub-product", "sub_product"])
    col_company = get_first_existing(df, ["Company", "company"])
    col_state = get_first_existing(df, ["State", "state"])
    col_zip = get_first_existing(df, ["ZIP code", "Zip code", "zip_code", "ZIP"])
    col_channel = get_first_existing(df, ["Submitted via", "Channel", "channel"])
    col_response = get_first_existing(
        df, ["Company response to consumer", "Company response"]
    )
    col_date_received = get_first_existing(df, ["Date received", "date_received"])
    col_issue = get_first_existing(df, ["Issue", "issue"])
    col_sub_issue = get_first_existing(
        df, ["Sub-issue", "Sub-Issue", "sub_issue", "Sub issue"]
    )

    missing = [name for name, val in [("issue", col_issue)] if val is None]
    if missing:
        raise RuntimeError(
            "CSV is missing required columns for the test: " + ", ".join(missing)
        )

    assert col_issue is not None
    if col_narrative is None and not (col_product and col_issue):
        raise RuntimeError(
            "CSV needs a narrative column, or both Product and Issue columns, "
            "for structured-only CFPB rows."
        )

    print("Using columns:")
    print(
        {
            "narrative": col_narrative,
            "issue": col_issue,
            "product": col_product,
            "sub_product": col_sub_product,
            "company": col_company,
            "state": col_state,
            "zip_code": col_zip,
            "channel": col_channel,
            "date_received": col_date_received,
            "requested_resolution": col_response,
            "sub_issue": col_sub_issue,
        }
    )

    narrative_ok = (
        df[col_narrative].notna()
        & (df[col_narrative].astype(str).str.len() >= 10)
        if col_narrative
        else pd.Series(False, index=df.index)
    )
    structured_ok = pd.Series(False, index=df.index)
    if col_product and col_issue:
        structured_ok = (
            df[col_product].notna()
            & df[col_issue].notna()
            & (df[col_product].astype(str).str.strip() != "")
            & (df[col_issue].astype(str).str.strip() != "")
        )
    valid_mask = narrative_ok | structured_ok
    if not valid_mask.any():
        raise RuntimeError(
            "No valid rows: need narrative >= 10 characters or non-empty Product+Issue."
        )

    valid_df = df.loc[valid_mask]
    n_available = len(valid_df)
    n_run = min(SAMPLE_COUNT, n_available)
    sample_df = valid_df.sample(n=n_run)
    print(f"\nValid rows in file: {n_available}; running {n_run} random sample(s) (SAMPLE_COUNT={SAMPLE_COUNT}).")
    if n_run < SAMPLE_COUNT:
        print(f"Note: fewer than {SAMPLE_COUNT} valid rows; only {n_run} executed.")

    if not api_key:
        print(
            f"\n{api_key_env} is not set; skipping the LLM-powered pipeline run."
        )
        return

    out_rows: list[dict] = []

    for i, (_, row) in enumerate(sample_df.iterrows(), start=1):
        print(f"\n{'='*60}\n--- Complaint {i}/{n_run} ---\n{'='*60}")
        try:
            payload = row_to_payload(
                row,
                col_narrative=col_narrative,
                col_product=col_product,
                col_sub_product=col_sub_product,
                col_sub_issue=col_sub_issue,
                col_company=col_company,
                col_state=col_state,
                col_zip=col_zip,
                col_channel=col_channel,
                col_response=col_response,
                col_date_received=col_date_received,
                col_issue=col_issue,
            )
        except ValueError as e:
            print(f"Skip row {i}: {e}")
            continue

        trimmed = {
            k: (str(v)[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
            for k, v in payload.items()
        }
        print("Payload (trimmed):", trimmed)

        try:
            final_state = process_complaint(payload)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"Pipeline error: {err}")
            traceback.print_exc()
            out_rows.append(
                build_row_record(
                    payload,
                    {},
                    source_csv=CSV_PATH,
                    sample_index=i,
                    pipeline_error=err[:2000],
                )
            )
            continue

        case = final_state["case"]
        routed = getattr(case, "routed_to", None) or (
            case.get("routed_to") if isinstance(case, dict) else None
        )
        print("Routed to:", routed)

        if i == 1:
            cls = getattr(case, "classification", None) or (
                case.get("classification") if isinstance(case, dict) else None
            )
            print("\nClassification (sample detail for complaint 1):")
            print(json.dumps(cls, indent=2, default=str))

        out_rows.append(
            build_row_record(payload, case, source_csv=CSV_PATH, sample_index=i)
        )

    if out_rows:
        pd.DataFrame(out_rows).to_csv(OUTPUT_CSV, index=False)
        print(f"\nWrote {len(out_rows)} row(s) to {OUTPUT_CSV!r}")
    else:
        print("\nNo output rows to write.")


if __name__ == "__main__":
    main()
