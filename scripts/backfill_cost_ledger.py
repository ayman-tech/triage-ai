"""Manual backfill entrypoint for historical workflow cost aggregates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.session import init_db
from app.observability.backfill import backfill_cost_ledger_from_workflow_runs


def main() -> None:
    init_db()
    inserted = backfill_cost_ledger_from_workflow_runs()
    print(f"Inserted {inserted} historical cost ledger rows.")


if __name__ == "__main__":
    main()
