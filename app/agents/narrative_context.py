"""Build narrative (or fallback) text for downstream LLM agents."""

from __future__ import annotations

from app.schemas.case import CaseRead

RICH_NARRATIVE_MIN = 10

NO_NARRATIVE_BOILERPLATE = (
    "[No consumer narrative was submitted (or text is too short for free-text analysis). "
    "Rely on the operational classification, CFPB portal selections, and tools.]"
)


def narrative_for_agent_prompt(case: CaseRead) -> str:
    """Primary narrative for risk/resolution/compliance prompts."""
    nar = (case.consumer_narrative or "").strip()
    lines = []
    if len(nar) >= RICH_NARRATIVE_MIN:
        lines.append(nar)
    else:
        lines.append(NO_NARRATIVE_BOILERPLATE)
    if case.cfpb_product:
        lines.append(f"CFPB portal product: {case.cfpb_product}")
    if case.cfpb_sub_product:
        lines.append(f"CFPB portal sub-product: {case.cfpb_sub_product}")
    if case.cfpb_issue:
        lines.append(f"CFPB portal issue: {case.cfpb_issue}")
    if case.cfpb_sub_issue:
        lines.append(f"CFPB portal sub-issue: {case.cfpb_sub_issue}")
    if case.product and not case.cfpb_product:
        lines.append(f"Reported product (legacy field): {case.product}")
    if case.sub_product:
        lines.append(f"Reported sub-product: {case.sub_product}")
    doc_summary = case.case_document_summary or {}
    if doc_summary.get("total_documents"):
        facts = doc_summary.get("facts") or {}
        lines.append(
            "Attached documents: "
            f"{doc_summary.get('total_documents', 0)} total, "
            f"{doc_summary.get('processed_documents', 0)} processed, "
            f"{doc_summary.get('pending_documents', 0)} pending."
        )
        if facts.get("amounts"):
            lines.append(f"Document amounts: {', '.join(facts['amounts'][:8])}")
        if facts.get("dates"):
            lines.append(f"Document dates: {', '.join(facts['dates'][:8])}")
        if facts.get("signals"):
            lines.append(f"Document signals: {', '.join(facts['signals'][:8])}")
    gate = case.document_gate_result or {}
    if gate.get("required"):
        lines.append(
            "Document gate: "
            f"status={gate.get('status')}, "
            f"processed={gate.get('processed_documents', 0)}/{gate.get('total_documents', 0)}, "
            f"failed={gate.get('failed_documents', 0)}."
        )
    consistency = case.document_consistency or {}
    if consistency.get("status") and consistency.get("status") != "not_applicable":
        lines.append(f"Document consistency status: {consistency.get('status')}")
    conflicts = consistency.get("conflicts") or []
    for conflict in conflicts[:5]:
        lines.append(
            "Document conflict: "
            f"{conflict.get('field')} narrative={conflict.get('narrative')} documents={conflict.get('documents')}"
        )
    return "\n".join(lines)
