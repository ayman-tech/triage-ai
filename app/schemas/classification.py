"""Pydantic models for complaint classification output."""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

EnumT = TypeVar("EnumT", bound=Enum)


class ProductCategory(str, Enum):
    CREDIT_REPORTING = "credit_reporting"
    DEBT_COLLECTION = "debt_collection"
    MORTGAGE = "mortgage"
    CREDIT_CARD = "credit_card"
    CHECKING_SAVINGS = "checking_savings"
    STUDENT_LOAN = "student_loan"
    VEHICLE_LOAN = "vehicle_loan"
    PAYDAY_LOAN = "payday_loan"
    MONEY_TRANSFER = "money_transfer"
    PREPAID_CARD = "prepaid_card"
    OTHER = "other"


class IssueType(str, Enum):
    INCORRECT_INFO = "incorrect_information"
    COMMUNICATION_TACTICS = "communication_tactics"
    ACCOUNT_MANAGEMENT = "account_management"
    BILLING_DISPUTES = "billing_disputes"
    FRAUD_SCAM = "fraud_or_scam"
    LOAN_MODIFICATION = "loan_modification"
    PAYMENT_PROCESSING = "payment_processing"
    DISCLOSURE_TRANSPARENCY = "disclosure_transparency"
    CLOSING_CANCELLING = "closing_or_cancelling"
    OTHER = "other"


def _taxonomy_slug(raw: object) -> str:
    """Normalize LLM labels ('Credit Card', 'Fraud or Scam') to snake_case tokens."""
    if raw is None:
        return ""
    if isinstance(raw, Enum):
        return raw.value
    s = str(raw).strip().lower()
    s = s.replace("&", " ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _match_enum_value(raw: object, enum_cls: type[EnumT]) -> EnumT | None:
    if isinstance(raw, enum_cls):
        return raw
    slug = _taxonomy_slug(raw)
    if not slug:
        return None
    for member in enum_cls:
        if member.value == slug:
            return member
    return None


# Short / alternate phrasing the LLM uses that do not slug-match enum values.
_PRODUCT_ALIASES: dict[str, ProductCategory] = {
    "cc": ProductCategory.CREDIT_CARD,
    "checking": ProductCategory.CHECKING_SAVINGS,
    "savings": ProductCategory.CHECKING_SAVINGS,
    "bank_account": ProductCategory.CHECKING_SAVINGS,
    "auto_loan": ProductCategory.VEHICLE_LOAN,
    "car_loan": ProductCategory.VEHICLE_LOAN,
    "student": ProductCategory.STUDENT_LOAN,
    "reporting": ProductCategory.CREDIT_REPORTING,
    "credit_report": ProductCategory.CREDIT_REPORTING,
    "collections": ProductCategory.DEBT_COLLECTION,
    "collection": ProductCategory.DEBT_COLLECTION,
}

_ISSUE_ALIASES: dict[str, IssueType] = {
    "fraud": IssueType.FRAUD_SCAM,
    "scam": IssueType.FRAUD_SCAM,
    "fraudulent": IssueType.FRAUD_SCAM,
    "billing": IssueType.BILLING_DISPUTES,
    "payment": IssueType.PAYMENT_PROCESSING,
    "payments": IssueType.PAYMENT_PROCESSING,
    "disclosure": IssueType.DISCLOSURE_TRANSPARENCY,
    "communication": IssueType.COMMUNICATION_TACTICS,
    "incorrect": IssueType.INCORRECT_INFO,
    "misinformation": IssueType.INCORRECT_INFO,
    "account": IssueType.ACCOUNT_MANAGEMENT,
    "loan_mod": IssueType.LOAN_MODIFICATION,
    "modification": IssueType.LOAN_MODIFICATION,
    "closing": IssueType.CLOSING_CANCELLING,
    "cancellation": IssueType.CLOSING_CANCELLING,
    "cancel": IssueType.CLOSING_CANCELLING,
}


def _coerce_product_category(raw: object) -> ProductCategory:
    matched = _match_enum_value(raw, ProductCategory)
    if matched is not None:
        return matched
    slug = _taxonomy_slug(raw)
    if slug in _PRODUCT_ALIASES:
        return _PRODUCT_ALIASES[slug]
    # Substring hint: "fraud" appears only in fraud_or_scam for issues, not products.
    logger.warning(
        "Unknown product_category %r after normalization %r; coercing to other",
        raw,
        slug,
    )
    return ProductCategory.OTHER


def _coerce_issue_type(raw: object) -> IssueType:
    matched = _match_enum_value(raw, IssueType)
    if matched is not None:
        return matched
    slug = _taxonomy_slug(raw)
    if slug in _ISSUE_ALIASES:
        return _ISSUE_ALIASES[slug]
    # Single-token fuzzy: e.g. "fraud" already in aliases; "fraud_or" partial — avoid.
    if slug and slug in ("fraud_or", "or_scam"):
        return IssueType.FRAUD_SCAM
    logger.warning(
        "Unknown issue_type %r after normalization %r; coercing to other",
        raw,
        slug,
    )
    return IssueType.OTHER


class ClassificationResult(BaseModel):
    """Structured output produced by the classification agent."""

    product_category: ProductCategory
    issue_type: IssueType
    sub_product: Optional[str] = Field(
        None,
        description="Operational sub-product id (snake_case) from product_to_sub_product taxonomy",
    )
    sub_issue: Optional[str] = Field(
        None,
        description="Operational sub-issue id (snake_case) from issue_to_sub_issue taxonomy",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence score"
    )
    reasoning: str = Field(
        ..., description="Brief chain‑of‑thought justification"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Key phrases extracted from the narrative",
    )
    review_recommended: bool = Field(
        default=False,
        description="True if downstream QA or human review is advised",
    )
    reason_codes: list[str] = Field(
        default_factory=list,
        description="Machine-readable tags for audit (e.g. narrative_missing)",
    )
    alternate_candidates: list[dict] = Field(
        default_factory=list,
        description="Optional runner-up label hypotheses for reconciliation",
    )

    @field_validator("product_category", mode="before")
    @classmethod
    def _validate_product_category(cls, v: object) -> Any:
        return _coerce_product_category(v)

    @field_validator("issue_type", mode="before")
    @classmethod
    def _validate_issue_type(cls, v: object) -> Any:
        return _coerce_issue_type(v)

    @field_validator("reason_codes", mode="before")
    @classmethod
    def _reason_codes_as_list(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []

    @field_validator("keywords", mode="before")
    @classmethod
    def _keywords_as_list(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            for sep in (";", ","):
                if sep in s:
                    return [p.strip() for p in s.split(sep) if p.strip()]
            return [s]
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []

    @field_validator("alternate_candidates", mode="before")
    @classmethod
    def _alternate_candidates_as_list(cls, v: object) -> list[dict]:
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        return []
