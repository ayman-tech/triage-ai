from __future__ import annotations

import os
import re

# A simulated "company knowledge pack" used for this demo repository.
# In a real multi-company deployment, this pack would be replaced per company.

MOCK_COMPANY_ID = "mock_bank"


def deployment_label() -> str:
    """Stable label for observability and DB audit rows (optional ``DEPLOYMENT_ID`` env)."""
    return os.getenv("DEPLOYMENT_ID", MOCK_COMPANY_ID)
# 1. Company Profile
COMPANY_PROFILE = {
    "display_name": "Mock Bank",
    "customer_identity": "digital bank",
    "supported_products": [
        "checking accounts",
        "savings accounts",
        "debit cards",
        "credit cards",
        "money transfers",
    ],
    "intake_operator_style": (
        "Speak as Mock Bank's complaints intake team. You are the bank representative "
        "documenting and triaging complaints for internal handling. Confirm whether the "
        "customer has already reported the issue to Mock Bank before closing intake."
    ),
    "intake_routing_guidance": [
        "Ask whether the customer has already reported the issue to the bank or spoken with customer support about it.",
        "If they already contacted the bank, capture that fact before marking intake complete.",
        "If the user reports fraud, a stolen card, unauthorized transactions, or severe account access issues, treat it as urgent.",
        "Acknowledge that the bank is taking the report now instead of redirecting the customer elsewhere.",
        "You may tell the user you are recording the complaint and flagging it for the right internal team.",
    ],
    "safe_reference_guidance": (
        "If you need an account reference, ask only for a safe locator such as last 4 digits, transaction date, "
        "merchant name, or a case/reference number."
    ),
}
# 2. classification taxonomy
PRODUCT_CATEGORIES = [
    {
        "product_category": "credit_card",
        "definition": "Complaints related to credit cards, card fees, disputes, and card security holds.",
        "cues": ["credit card", "card", "charge", "purchase", "statement", "apr", "interest"],
    },
    {
        "product_category": "checking_savings",
        "definition": "Complaints related to deposit accounts, transfers, holds on funds, and access restrictions.",
        "cues": ["checking", "savings", "debit card", "funds access", "transfer", "ach", "routing number"],
    },
    {
        "product_category": "debt_collection",
        "definition": "Complaints about debt collection practices, communications, and account handling by collectors.",
        "cues": ["debt", "collector", "collection", "dunning", "harass", "communication", "calls"],
    },
    {
        "product_category": "mortgage",
        "definition": "Complaints related to mortgage servicing, escrow, modifications, and foreclosure interactions.",
        "cues": ["mortgage", "servicing", "escrow", "foreclosure", "modification", "payment", "loan"],
    },
    {
        "product_category": "credit_reporting",
        "definition": "Complaints about credit reports, inaccuracies, disputes, and reporting timelines.",
        "cues": ["credit report", "credit bureau", "dispute", "incorrect information", "report"],
    },
    {
        "product_category": "student_loan",
        "definition": "Complaints related to student loan servicing, repayment changes, and billing processing.",
        "cues": ["student loan", "servicing", "repayment", "payment", "billing"],
    },
    {
        "product_category": "vehicle_loan",
        "definition": "Complaints for auto loans and related servicing, payoff statements, or payment processing.",
        "cues": ["auto loan", "vehicle loan", "payoff", "title", "loan payment", "servicing"],
    },
    {
        "product_category": "payday_loan",
        "definition": "Complaints about payday loans, short-term borrowing terms, and related servicing.",
        "cues": ["payday", "short-term", "installment", "loan terms"],
    },
    {
        "product_category": "money_transfer",
        "definition": "Complaints related to money transfers, remittances, and payment processing failures.",
        "cues": ["wire", "transfer", "remittance", "international", "payment processing", "transaction"],
    },
    {
        "product_category": "prepaid_card",
        "definition": "Complaints regarding prepaid cards, loading issues, fees, and access restrictions.",
        "cues": ["prepaid", "card", "load", "balance", "fees"],
    },
    {
        "product_category": "other",
        "definition": "Complaints that do not clearly fall into other product categories.",
        "cues": ["other", "unknown", "miscellaneous"],
    },
]

ISSUE_TYPES = [
    {
        "issue_type": "billing_disputes",
        "definition": "Complaints involving billing errors, duplicate charges, fee disputes, or incorrect account amounts.",
        "cues": ["charged twice", "duplicate", "fee", "billing", "amount", "refund", "overcharged"],
    },
    {
        "issue_type": "payment_processing",
        "definition": "Complaints where payments are not applied correctly or processing fails, causing account/payment issues.",
        "cues": ["payment not applied", "processing", "declined", "reversed", "returned payment", "misapplied"],
    },
    {
        "issue_type": "account_management",
        "definition": "Complaints about account status changes, freezes, access restrictions, or service/record handling.",
        "cues": ["frozen", "locked", "access", "account closed", "unable to access", "restriction"],
    },
    {
        "issue_type": "fraud_or_scam",
        "definition": "Complaints alleging fraud, scam activity, or unauthorized transactions triggering holds/blocks.",
        "cues": ["fraud", "scam", "unauthorized", "imposter", "transaction alert", "hold", "security"],
    },
    {
        "issue_type": "communication_tactics",
        "definition": "Complaints about unfair or inappropriate communications, calls, or notice delivery.",
        "cues": ["harass", "threaten", "calls", "texts", "no notice", "communication", "contact"],
    },
    {
        "issue_type": "incorrect_information",
        "definition": "Complaints about inaccurate data, record errors, or missing/incorrect disclosures.",
        "cues": ["incorrect", "wrong", "inaccurate", "reporting", "disclosure", "mistake"],
    },
    {
        "issue_type": "loan_modification",
        "definition": "Complaints related to loan modification decisions, repayment plan changes, or restructuring outcomes.",
        "cues": ["modify", "modification", "repayment plan", "restructure", "denied", "trial period"],
    },
    {
        "issue_type": "disclosure_transparency",
        "definition": "Complaints about disclosures, notices, or transparency failures.",
        "cues": ["disclosure", "notice", "terms", "explained", "transparent", "statement"],
    },
    {
        "issue_type": "closing_or_cancelling",
        "definition": "Complaints involving account closure/cancellation, cancellation handling, or termination processes.",
        "cues": ["cancel", "cancellation", "closed", "closure", "termination"],
    },
    {
        "issue_type": "other",
        "definition": "Complaints that do not map cleanly to other issue types.",
        "cues": ["other", "unknown", "miscellaneous"],
    },
]

# These are not mandatory for your code to run, but they align your
# internal taxonomy more closely with the current CFPB structure.
RECOMMENDED_PRODUCT_CATEGORY_NORMALIZATION = {
    "credit_reporting": "credit_reporting_or_other_personal_consumer_reports",
    "vehicle_loan": "vehicle_loan_or_lease",
    "payday_loan": "payday_title_personal_or_advance_loan",
    "money_transfer": "money_transfer_virtual_currency_or_money_service",
}

RECOMMENDED_NEW_PRODUCT_CATEGORIES = [
    {
        "product_category": "debt_or_credit_management",
        "definition": "Complaints about third-party services used to repair credit, settle debt, avoid foreclosure, or obtain student loan debt relief.",
        "cues": [
            "credit repair",
            "debt settlement",
            "debt relief",
            "foreclosure avoidance",
            "mortgage modification company",
            "student loan forgiveness company",
        ],
    }
]

# -------------------------------------------------------------------
# NEW KEYS TO ADD TO YOUR EXISTING OPERATIONAL_TAXONOMY
# -------------------------------------------------------------------

PRODUCT_TO_SUB_PRODUCT_TAXONOMY = {
    "credit_card": [
        {
            "sup_product": "general_purpose_credit_card_or_charge_card",
            "definition": "A credit card or charge card that can be used broadly wherever cards are accepted.",
            "cues": ["visa", "mastercard", "amex", "charge card", "general purpose card"],
        },
        {
            "sup_product": "store_credit_card",
            "definition": "A retail card usable only at one store or a related chain.",
            "cues": ["store card", "retail card", "merchant card", "can only use at store"],
        },
    ],

    "checking_savings": [
        {
            "sup_product": "checking_account",
            "definition": "Standard checking account and related payment instruments.",
            "cues": ["checking account", "debit card", "atm card", "checkbook", "checking"],
        },
        {
            "sup_product": "savings_account",
            "definition": "Savings or interest-bearing deposit account.",
            "cues": ["savings account", "savings", "interest on savings"],
        },
        {
            "sup_product": "certificate_of_deposit",
            "definition": "Certificate of deposit or time deposit account.",
            "cues": ["cd", "certificate of deposit", "time deposit"],
        },
        {
            "sup_product": "other_banking_product_or_service",
            "definition": "Other deposit-adjacent banking product or service not clearly checking, savings, or CD.",
            "cues": ["banking service", "deposit product", "bank account service"],
        },
    ],

    "debt_collection": [
        {
            "sup_product": "credit_card_debt",
            "definition": "Collection on unpaid revolving or card balances.",
            "cues": ["credit card debt", "card balance", "past due card"],
        },
        {
            "sup_product": "medical_debt",
            "definition": "Collection on hospital, doctor, ambulance, or other healthcare bills.",
            "cues": ["medical debt", "hospital bill", "doctor bill", "ambulance bill"],
        },
        {
            "sup_product": "mortgage_debt",
            "definition": "Collection activity related to mortgage deficiency or mortgage-related debt.",
            "cues": ["mortgage debt", "foreclosure deficiency", "home loan debt"],
        },
        {
            "sup_product": "student_loan_debt",
            "definition": "Collection related to education debt.",
            "cues": ["student loan debt", "education debt", "school debt"],
        },
        {
            "sup_product": "auto_debt",
            "definition": "Collection on unpaid vehicle loan or lease obligations.",
            "cues": ["auto debt", "car loan debt", "vehicle debt"],
        },
        {
            "sup_product": "payday_loan_debt",
            "definition": "Collection on payday, title, installment, or similar small-dollar debt.",
            "cues": ["payday debt", "title loan debt", "small-dollar debt"],
        },
        {
            "sup_product": "rental_debt",
            "definition": "Collection related to rent or leasing obligations.",
            "cues": ["rental debt", "rent debt", "lease debt", "landlord collection"],
        },
        {
            "sup_product": "telecommunications_debt",
            "definition": "Collection related to phone, cable, internet, or telecom services.",
            "cues": ["phone bill collection", "telecom debt", "cable debt", "internet bill"],
        },
        {
            "sup_product": "other_debt",
            "definition": "Other collected debt not cleanly mapped elsewhere.",
            "cues": ["debt collector", "collection agency", "debt notice"],
        },
    ],

    "mortgage": [
        {
            "sup_product": "conventional_home_mortgage",
            "definition": "Standard non-government-backed mortgage product.",
            "cues": ["conventional mortgage", "home mortgage", "mortgage lender"],
        },
        {
            "sup_product": "fha_mortgage",
            "definition": "Federal Housing Administration-backed mortgage.",
            "cues": ["fha loan", "fha mortgage"],
        },
        {
            "sup_product": "va_mortgage",
            "definition": "Veterans Affairs-backed home loan.",
            "cues": ["va loan", "va mortgage"],
        },
        {
            "sup_product": "usda_mortgage",
            "definition": "USDA-backed mortgage product.",
            "cues": ["usda mortgage", "rural development loan"],
        },
        {
            "sup_product": "reverse_mortgage",
            "definition": "Reverse mortgage product, usually for older homeowners.",
            "cues": ["reverse mortgage", "hecm"],
        },
        {
            "sup_product": "home_equity_loan_or_line_of_credit",
            "definition": "Home equity loan or HELOC tied to residential property.",
            "cues": ["heloc", "home equity loan", "equity line"],
        },
        {
            "sup_product": "manufactured_home_loan",
            "definition": "Mortgage or secured loan tied to a manufactured home.",
            "cues": ["manufactured home loan", "mobile home mortgage"],
        },
    ],

    "credit_reporting": [
        {
            "sup_product": "credit_reporting",
            "definition": "Traditional credit reporting and bureau file disputes.",
            "cues": ["credit report", "credit bureau", "tradeline", "credit file"],
        },
        {
            "sup_product": "tenant_screening",
            "definition": "Screening report used by landlords or property managers.",
            "cues": ["tenant screening", "rental screening", "lease screening"],
        },
        {
            "sup_product": "employment_screening_or_background_check",
            "definition": "Consumer report used for hiring or employment purposes.",
            "cues": ["background check", "employment screening", "hiring report"],
        },
        {
            "sup_product": "check_writing_history",
            "definition": "Consumer reporting related to checking account or check-writing history.",
            "cues": ["chexsystems", "check writing history", "bank screening"],
        },
        {
            "sup_product": "other_personal_consumer_report",
            "definition": "Other personal consumer report outside core credit reporting.",
            "cues": ["consumer report", "screening report", "specialty report"],
        },
    ],

    "student_loan": [
        {
            "sup_product": "federal_student_loan",
            "definition": "Federally held or federally guaranteed education loan.",
            "cues": ["federal student loan", "direct loan", "department of education", "servicer"],
        },
        {
            "sup_product": "private_student_loan",
            "definition": "Privately issued student loan.",
            "cues": ["private student loan", "school lender", "private education loan"],
        },
    ],

    "vehicle_loan": [
        {
            "sup_product": "loan",
            "definition": "Vehicle financing through a loan.",
            "cues": ["auto loan", "car loan", "vehicle financing"],
        },
        {
            "sup_product": "lease",
            "definition": "Vehicle financing through a lease agreement.",
            "cues": ["car lease", "vehicle lease", "leased car"],
        },
    ],

    "payday_loan": [
        {
            "sup_product": "payday_loan",
            "definition": "Short-term payday loan product.",
            "cues": ["payday loan", "cash advance loan", "next paycheck"],
        },
        {
            "sup_product": "title_loan",
            "definition": "Loan secured by a vehicle title.",
            "cues": ["title loan", "car title loan"],
        },
        {
            "sup_product": "personal_loan",
            "definition": "Unsecured or installment-style personal loan.",
            "cues": ["personal loan", "installment loan", "signature loan"],
        },
        {
            "sup_product": "line_of_credit",
            "definition": "Open-ended small-dollar credit line.",
            "cues": ["line of credit", "credit line"],
        },
        {
            "sup_product": "advance_loan",
            "definition": "Earned wage, paycheck advance, or similar advance product.",
            "cues": ["advance loan", "earned wage access", "pay advance", "cash advance app"],
        },
        {
            "sup_product": "tax_refund_anticipation_loan_or_check",
            "definition": "Product advancing or issuing funds against a tax refund.",
            "cues": ["refund anticipation", "tax refund advance", "tax advance"],
        },
    ],

    "money_transfer": [
        {
            "sup_product": "domestic_us_money_transfer",
            "definition": "Transfer of money within the United States.",
            "cues": ["domestic transfer", "us transfer", "send money in us"],
        },
        {
            "sup_product": "international_money_transfer",
            "definition": "Cross-border remittance or money transfer.",
            "cues": ["international transfer", "remittance", "send money abroad"],
        },
        {
            "sup_product": "virtual_currency",
            "definition": "Crypto or blockchain-based money movement/storage service.",
            "cues": ["crypto", "bitcoin", "ethereum", "virtual currency", "wallet address"],
        },
        {
            "sup_product": "mobile_or_digital_wallet",
            "definition": "Wallet-based payment or stored-value service operated digitally.",
            "cues": ["digital wallet", "mobile wallet", "apple pay", "google pay", "wallet app"],
        },
        {
            "sup_product": "money_order_cashiers_check_travelers_check",
            "definition": "Money order, cashier’s check, or traveler’s check product.",
            "cues": ["money order", "cashier's check", "traveler's check"],
        },
        {
            "sup_product": "foreign_currency_exchange",
            "definition": "Currency conversion service.",
            "cues": ["exchange rate", "currency exchange", "fx conversion"],
        },
        {
            "sup_product": "check_cashing_service",
            "definition": "Service that cashes checks for a fee.",
            "cues": ["check cashing", "cash my check", "cashes check for fee"],
        },
    ],

    "prepaid_card": [
        {
            "sup_product": "general_purpose_prepaid_card",
            "definition": "Reloadable prepaid card for general purchases.",
            "cues": ["prepaid card", "reloadable card", "general purpose prepaid"],
        },
        {
            "sup_product": "gift_card",
            "definition": "Merchant or network gift card.",
            "cues": ["gift card", "gift balance"],
        },
        {
            "sup_product": "government_benefit_card",
            "definition": "Prepaid card for government benefits or disbursements.",
            "cues": ["benefit card", "ebt card", "government payment card"],
        },
        {
            "sup_product": "payroll_card",
            "definition": "Prepaid card used by an employer to pay wages.",
            "cues": ["payroll card", "wage card"],
        },
        {
            "sup_product": "student_prepaid_card",
            "definition": "Prepaid card designed for student spending or campus-linked use.",
            "cues": ["student prepaid", "campus card", "student card"],
        },
    ],

    "other": [
        {
            "sup_product": "other_financial_product",
            "definition": "Fallback sub-product for complaints that cannot yet be cleanly mapped.",
            "cues": ["other", "miscellaneous", "unknown"],
        }
    ],
}

ISSUE_TO_SUB_ISSUE_TAXONOMY = {
    "billing_disputes": [
        {
            "sub_issue": "duplicate_charge",
            "definition": "Consumer was charged more than once for the same event or transaction.",
            "applicable_products": ["credit_card", "checking_savings", "prepaid_card", "money_transfer"],
            "cues": ["charged twice", "duplicate charge", "billed twice"],
        },
        {
            "sub_issue": "wrong_amount_charged_or_received",
            "definition": "The amount charged, withdrawn, posted, or received was wrong.",
            "applicable_products": ["credit_card", "money_transfer", "vehicle_loan", "mortgage"],
            "cues": ["wrong amount", "incorrect amount", "amount charged was wrong"],
        },
        {
            "sub_issue": "problem_with_fees_charged",
            "definition": "Unexpected, excessive, undisclosed, or inaccurate fee.",
            "applicable_products": ["credit_card", "checking_savings", "payday_loan", "mortgage", "prepaid_card"],
            "cues": ["fee", "late fee", "service fee", "unexpected fee", "overdraft fee"],
        },
        {
            "sub_issue": "billing_statement_error",
            "definition": "Statement or bill contains inaccurate line items, balances, or dates.",
            "applicable_products": ["credit_card", "student_loan", "vehicle_loan", "mortgage"],
            "cues": ["statement error", "billing error", "wrong statement", "incorrect statement"],
        },
        {
            "sub_issue": "refund_not_received",
            "definition": "Consumer expected a refund, credit, or reimbursement that did not arrive.",
            "applicable_products": ["credit_card", "prepaid_card", "money_transfer"],
            "cues": ["refund not received", "credit never posted", "reimbursement missing"],
        },
        {
            "sub_issue": "payoff_amount_error",
            "definition": "Payoff quote, balance due, or deficiency amount appears incorrect.",
            "applicable_products": ["vehicle_loan", "mortgage", "student_loan"],
            "cues": ["payoff amount wrong", "incorrect payoff", "balance due incorrect"],
        },
    ],

    "payment_processing": [
        {
            "sub_issue": "payment_not_applied",
            "definition": "Consumer made a payment but it was not credited correctly.",
            "applicable_products": ["mortgage", "student_loan", "vehicle_loan", "credit_card"],
            "cues": ["payment not applied", "misapplied payment", "payment missing"],
        },
        {
            "sub_issue": "payment_posted_late",
            "definition": "Payment was received but posted too late, causing fees or delinquency.",
            "applicable_products": ["credit_card", "mortgage", "student_loan", "vehicle_loan"],
            "cues": ["posted late", "late posting", "payment credited late"],
        },
        {
            "sub_issue": "autopay_or_ach_failure",
            "definition": "Autopay or ACH setup failed, duplicated, or did not execute as intended.",
            "applicable_products": ["checking_savings", "credit_card", "student_loan", "mortgage"],
            "cues": ["autopay failed", "ach failed", "automatic payment issue"],
        },
        {
            "sub_issue": "payment_reversed_or_returned",
            "definition": "A valid payment was reversed, rejected, or returned.",
            "applicable_products": ["checking_savings", "credit_card", "payday_loan", "vehicle_loan"],
            "cues": ["reversed payment", "returned payment", "payment rejected"],
        },
        {
            "sub_issue": "funds_not_available_when_promised",
            "definition": "Transferred or deposited funds were not available in the promised time.",
            "applicable_products": ["checking_savings", "money_transfer", "prepaid_card"],
            "cues": ["funds not available", "money not available", "delay in funds"],
        },
        {
            "sub_issue": "transaction_processing_failure",
            "definition": "The transaction failed, stalled, or was cancelled incorrectly.",
            "applicable_products": ["money_transfer", "prepaid_card", "credit_card"],
            "cues": ["processing failed", "transaction failed", "declined incorrectly"],
        },
    ],

    "account_management": [
        {
            "sub_issue": "account_opening_problem",
            "definition": "Consumer could not open the account or product as expected.",
            "applicable_products": ["checking_savings", "credit_card", "student_loan", "vehicle_loan"],
            "cues": ["unable to open", "application issue", "account opening problem"],
        },
        {
            "sub_issue": "account_frozen_or_locked",
            "definition": "Account access was blocked, frozen, or locked.",
            "applicable_products": ["checking_savings", "prepaid_card", "money_transfer", "credit_card"],
            "cues": ["account frozen", "locked out", "restricted account"],
        },
        {
            "sub_issue": "account_closed_or_restricted",
            "definition": "Account was closed or limited unexpectedly or without fair explanation.",
            "applicable_products": ["checking_savings", "credit_card", "prepaid_card"],
            "cues": ["account closed", "restricted access", "closure without notice"],
        },
        {
            "sub_issue": "funds_hold_or_availability_delay",
            "definition": "Bank or provider placed a hold, reserve, or delay on funds access.",
            "applicable_products": ["checking_savings", "prepaid_card", "money_transfer"],
            "cues": ["hold on funds", "reserve", "delayed availability", "funds on hold"],
        },
        {
            "sub_issue": "card_or_credentials_not_received",
            "definition": "Card, replacement card, login credentials, or access tools were not received.",
            "applicable_products": ["credit_card", "prepaid_card", "checking_savings"],
            "cues": ["card not received", "replacement card issue", "credentials not received"],
        },
        {
            "sub_issue": "loan_transfer_or_servicer_change_problem",
            "definition": "Problems caused by account sale, transfer, or servicer migration.",
            "applicable_products": ["mortgage", "student_loan", "vehicle_loan"],
            "cues": ["loan sold", "transferred to another company", "servicer changed"],
        },
    ],

    "fraud_or_scam": [
        {
            "sub_issue": "account_opened_without_consent_or_knowledge",
            "definition": "Account or loan was opened without the consumer’s consent or knowledge.",
            "applicable_products": ["credit_card", "student_loan", "vehicle_loan", "checking_savings"],
            "cues": ["opened without my knowledge", "without my consent", "fraudulent account"],
        },
        {
            "sub_issue": "unauthorized_transaction",
            "definition": "Transaction was unauthorized, disputed, or appears fraudulent.",
            "applicable_products": ["credit_card", "checking_savings", "prepaid_card", "money_transfer"],
            "cues": ["unauthorized transaction", "unauthorized charge", "fraudulent purchase"],
        },
        {
            "sub_issue": "identity_theft_or_account_takeover",
            "definition": "Consumer identity or account credentials were used improperly.",
            "applicable_products": ["credit_card", "checking_savings", "credit_reporting", "student_loan"],
            "cues": ["identity theft", "account takeover", "stolen identity"],
        },
        {
            "sub_issue": "imposter_or_phishing_scam",
            "definition": "Consumer was targeted by impersonation, spoofing, or phishing.",
            "applicable_products": ["money_transfer", "checking_savings", "debt_collection"],
            "cues": ["phishing", "imposter", "spoofed", "pretended to be"],
        },
        {
            "sub_issue": "virtual_currency_or_wallet_fraud",
            "definition": "Fraud involving crypto assets, wallet access, or digital-wallet movement.",
            "applicable_products": ["money_transfer"],
            "cues": ["crypto scam", "wallet drained", "blockchain fraud", "token scam"],
        },
        {
            "sub_issue": "service_not_as_promised_due_to_scam_like_behavior",
            "definition": "Consumer paid for a service that appears deceptive or scam-like.",
            "applicable_products": ["payday_loan", "mortgage", "student_loan"],
            "cues": ["promised one thing", "scam service", "misled into paying"],
        },
    ],

    "communication_tactics": [
        {
            "sub_issue": "frequent_or_repeated_calls_or_messages",
            "definition": "Consumer received repeated contacts that felt excessive or harassing.",
            "applicable_products": ["debt_collection", "mortgage", "vehicle_loan"],
            "cues": ["keeps calling", "repeated messages", "constant contact"],
        },
        {
            "sub_issue": "contacted_after_stop_request",
            "definition": "Company or collector continued contact after being told to stop.",
            "applicable_products": ["debt_collection"],
            "cues": ["told them to stop", "cease communication", "still contacting me"],
        },
        {
            "sub_issue": "contacted_outside_allowed_hours",
            "definition": "Consumer was contacted at unreasonable or prohibited times.",
            "applicable_products": ["debt_collection"],
            "cues": ["before 8am", "after 9pm", "late night calls", "early morning calls"],
        },
        {
            "sub_issue": "used_threatening_or_abusive_language",
            "definition": "Communications used obscene, abusive, coercive, or threatening language.",
            "applicable_products": ["debt_collection"],
            "cues": ["threatened", "abusive language", "harassed", "obscene language"],
        },
        {
            "sub_issue": "misleading_or_false_collection_threats",
            "definition": "Collector falsely threatened arrest, lawsuit, or immediate legal action.",
            "applicable_products": ["debt_collection"],
            "cues": ["threatened arrest", "fake lawsuit threat", "legal action threat"],
        },
        {
            "sub_issue": "third_party_contact_or_privacy_violation",
            "definition": "Company disclosed debt or complaint information to unauthorized third parties.",
            "applicable_products": ["debt_collection", "credit_reporting"],
            "cues": ["called my family", "told employer", "shared private information"],
        },
    ],

    "incorrect_information": [
        {
            "sub_issue": "incorrect_credit_report_entry",
            "definition": "Credit report contains inaccurate tradeline, status, balance, or identity data.",
            "applicable_products": ["credit_reporting", "debt_collection", "credit_card", "student_loan"],
            "cues": ["wrong on credit report", "inaccurate tradeline", "incorrect bureau reporting"],
        },
        {
            "sub_issue": "incorrect_balance_or_amount_owed",
            "definition": "The stated debt, balance, or payoff amount is incorrect.",
            "applicable_products": ["debt_collection", "mortgage", "vehicle_loan", "student_loan"],
            "cues": ["wrong balance", "amount owed incorrect", "debt is not mine"],
        },
        {
            "sub_issue": "investigation_failed_to_correct_error",
            "definition": "Provider or bureau investigated but failed to fix the reported problem.",
            "applicable_products": ["credit_reporting", "checking_savings", "credit_card"],
            "cues": ["investigation did not fix", "dispute not resolved", "never corrected"],
        },
        {
            "sub_issue": "incorrect_personal_information",
            "definition": "Name, address, SSN, phone, or identity-related data is wrong.",
            "applicable_products": ["credit_reporting", "checking_savings", "student_loan"],
            "cues": ["wrong address", "incorrect personal info", "wrong identity data"],
        },
        {
            "sub_issue": "incorrect_loan_or_account_terms",
            "definition": "Company records show wrong rate, term, schedule, or product terms.",
            "applicable_products": ["mortgage", "vehicle_loan", "student_loan", "payday_loan"],
            "cues": ["wrong interest rate", "wrong terms", "term mismatch"],
        },
        {
            "sub_issue": "wrong_exchange_rate_or_tax_calculation",
            "definition": "Provider applied the wrong rate, tax, or transfer math.",
            "applicable_products": ["money_transfer"],
            "cues": ["wrong exchange rate", "taxes incorrect", "transfer math wrong"],
        },
    ],

    "loan_modification": [
        {
            "sub_issue": "application_delay_or_document_handling_problem",
            "definition": "Modification, hardship, or relief request was delayed or mishandled.",
            "applicable_products": ["mortgage", "student_loan", "vehicle_loan"],
            "cues": ["delay in modification", "lost documents", "processing delay"],
        },
        {
            "sub_issue": "denial_of_modification_or_relief",
            "definition": "Consumer was denied modification, hardship, forbearance, or lower-payment relief.",
            "applicable_products": ["mortgage", "student_loan", "vehicle_loan"],
            "cues": ["denied modification", "denied hardship", "refused lower payment"],
        },
        {
            "sub_issue": "trial_modification_or_repayment_plan_problem",
            "definition": "Trial plan or temporary payment arrangement was mismanaged.",
            "applicable_products": ["mortgage", "student_loan"],
            "cues": ["trial modification", "repayment plan issue", "temporary plan problem"],
        },
        {
            "sub_issue": "foreclosure_or_repossession_while_under_review",
            "definition": "Enforcement continued while assistance or modification was supposedly under review.",
            "applicable_products": ["mortgage", "vehicle_loan"],
            "cues": ["foreclosure while reviewing", "repossession during review"],
        },
        {
            "sub_issue": "debt_relief_service_problem",
            "definition": "Third-party debt relief or credit assistance service failed to deliver promised results.",
            "applicable_products": ["payday_loan", "student_loan", "mortgage"],
            "cues": ["debt relief company", "credit repair failed", "foreclosure rescue company"],
        },
    ],

    "disclosure_transparency": [
        {
            "sub_issue": "confusing_or_misleading_advertising",
            "definition": "Marketing, solicitation, or sales language was deceptive or unclear.",
            "applicable_products": ["credit_card", "mortgage", "vehicle_loan", "payday_loan", "money_transfer"],
            "cues": ["misleading advertising", "marketing was deceptive", "bait and switch"],
        },
        {
            "sub_issue": "missing_or_confusing_disclosures",
            "definition": "Required disclosures, fine print, or key conditions were missing or unclear.",
            "applicable_products": ["money_transfer", "mortgage", "credit_card", "debt_or_credit_management"],
            "cues": ["missing disclosure", "fine print unclear", "not properly disclosed"],
        },
        {
            "sub_issue": "terms_changed_mid_process",
            "definition": "Key terms changed during application, closing, or onboarding.",
            "applicable_products": ["mortgage", "vehicle_loan", "credit_card", "payday_loan"],
            "cues": ["terms changed", "changed after closing", "mid-deal changes"],
        },
        {
            "sub_issue": "unclear_fee_or_apr_disclosure",
            "definition": "APR, interest, fee schedule, or cost structure was unclear or misrepresented.",
            "applicable_products": ["credit_card", "payday_loan", "mortgage", "vehicle_loan"],
            "cues": ["apr not disclosed", "unclear interest", "hidden fees"],
        },
        {
            "sub_issue": "misleading_affiliation_or_forgiveness_claim",
            "definition": "Provider falsely suggested affiliation with a lender, servicer, school, or government program.",
            "applicable_products": ["student_loan", "mortgage", "debt_or_credit_management"],
            "cues": ["pretended to be government", "claimed forgiveness", "fake affiliation"],
        },
        {
            "sub_issue": "privacy_or_service_terms_unclear",
            "definition": "Important privacy, usage, or service limitations were not explained clearly.",
            "applicable_products": ["money_transfer", "checking_savings", "prepaid_card"],
            "cues": ["privacy not explained", "service limits unclear", "terms not clear"],
        },
    ],

    "closing_or_cancelling": [
        {
            "sub_issue": "unable_to_close_account",
            "definition": "Consumer tried to close or cancel but the company prevented or delayed it.",
            "applicable_products": ["checking_savings", "credit_card", "prepaid_card"],
            "cues": ["unable to close", "cannot cancel account", "won't close account"],
        },
        {
            "sub_issue": "account_closed_without_funds_returned",
            "definition": "Account was closed and the remaining funds were not returned properly.",
            "applicable_products": ["checking_savings", "prepaid_card"],
            "cues": ["funds not returned", "closed account balance missing"],
        },
        {
            "sub_issue": "cancellation_or_payoff_delay",
            "definition": "Closure, payoff, or cancellation was delayed or processed incorrectly.",
            "applicable_products": ["vehicle_loan", "mortgage", "credit_card", "student_loan"],
            "cues": ["payoff delayed", "cancellation delay", "closure processing issue"],
        },
        {
            "sub_issue": "title_or_lien_release_problem",
            "definition": "Company failed to release title, lien, or ownership documents correctly.",
            "applicable_products": ["vehicle_loan", "mortgage"],
            "cues": ["car title not received", "lien release problem", "title issue"],
        },
        {
            "sub_issue": "remaining_balance_after_repossession_or_closure",
            "definition": "Consumer disputes deficiency or residual balance after product closure or repossession.",
            "applicable_products": ["vehicle_loan", "debt_collection"],
            "cues": ["deficiency balance", "remaining balance after repo", "still owe after repossession"],
        },
        {
            "sub_issue": "voluntary_closure_penalty_or_fee",
            "definition": "Unexpected penalty or fee tied to early cancellation, payoff, or closure.",
            "applicable_products": ["vehicle_loan", "mortgage", "payday_loan"],
            "cues": ["prepayment penalty", "early closure fee", "termination fee"],
        },
    ],

    "other": [
        {
            "sub_issue": "unable_to_classify",
            "definition": "Narrative is too sparse, contradictory, or ambiguous to classify reliably.",
            "applicable_products": ["other"],
            "cues": ["unclear", "unknown", "not enough detail"],
        },
        {
            "sub_issue": "multi_issue_narrative",
            "definition": "Complaint contains multiple unrelated issues and needs decomposition.",
            "applicable_products": ["other"],
            "cues": ["multiple issues", "several problems", "different complaints in one"],
        },
    ],
}

OPERATIONAL_TAXONOMY = {
    "product_categories": PRODUCT_CATEGORIES,
    "issue_types": ISSUE_TYPES,
    "product_to_sub_product_taxonomy": PRODUCT_TO_SUB_PRODUCT_TAXONOMY,
    "issue_to_sub_issue_taxonomy": ISSUE_TO_SUB_ISSUE_TAXONOMY,
}

# 3. risk taxonomy
SEVERITY_RUBRIC = [
    {
        "level": "low",
        "description": "Informational errors or minor inconveniences with limited financial or access impact.",
        "cues": ["minor", "small", "informational", "typo"],
        "escalation": False,
    },
    {
        "level": "medium",
        "description": "Billing/payment disputes or moderate harm requiring timely resolution.",
        "cues": ["duplicate", "refund", "declined", "misapplied", "fee", "processing"],
        "escalation": False,
    },
    {
        "level": "high",
        "description": "Significant consumer impact, regulatory exposure, or repeated failures by the institution.",
        "cues": ["regulatory", "violation", "significant", "repeated", "multiple attempts", "large loss"],
        "escalation": True,
    },
    {
        "level": "critical",
        "description": "Severe harm: access to funds blocked, imminent litigation signals, fraud holds, or systemic issues.",
        "cues": ["frozen", "wage", "paycheck", "unauthorized", "fraud", "lawsuit", "imminent"],
        "escalation": True,
    },
]

# 4. compliance taxonomy
POLICY_SNIPPETS = [
    {
        "policy_id": "disclosures_and_notice",
        "description": "Ensure customers receive clear notices and disclosures for account changes, holds, and disputes.",
        "cues": ["notice", "disclosure", "terms", "explained"],
    },
    {
        "policy_id": "error_resolution",
        "description": "Follow documented error resolution steps; confirm investigations, document outcomes, and correct records when needed.",
        "cues": ["investigation", "error", "correct", "records", "dispute"],
    },
    {
        "policy_id": "access_restoration",
        "description": "When funds access is restricted, prioritize restoration timelines and provide justification if access cannot be immediately restored.",
        "cues": ["access", "restored", "released", "funds", "blocked", "hold", "frozen"],
    },
    {
        "policy_id": "collections_and_communications",
        "description": "Ensure compliant communications and appropriate handling for debt collection scenarios.",
        "cues": ["collector", "calls", "communication", "harass"],
    },
]

# 5. Routing Taxonomy
ROUTING_MATRIX = {
    # Core ownership mapping.
    # This is the "company ownership graph" simplified into rules.
    "team_by_product_category": {
        "credit_card": "credit_card_operations_team",
        "checking_savings": "fraud_and_access_ops_team",
        "debt_collection": "debt_collection_team",
        "mortgage": "mortgage_servicing_team",
        "credit_reporting": "credit_reporting_team",
        "student_loan": "student_loan_servicing_team",
        "vehicle_loan": "auto_loan_team",
        "payday_loan": "consumer_lending_team",
        "money_transfer": "payments_team",
        "prepaid_card": "payments_team",
        "other": "general_complaints_team",
    },
    "executive_team": "executive_complaints_team",
    "management_escalation_team": "management_escalation_team",
}

# 6. Root cause taxonomy
ROOT_CAUSE_CONTROLS = [
    {
        "root_cause_category": "Fraud review or card-blocking delay",
        "root_cause_code": "fraud_review_or_card_blocking_delay",
        "description": "The bank's fraud controls may not have blocked the activity quickly enough, or a fraud-related restriction was not reviewed and resolved in time.",
        "business_summary": "Likely breakdown in fraud monitoring, alert response, or card-block timing.",
        "cues": ["fraud", "unauthorized transaction", "lost card", "stolen card", "blocked card", "hold", "frozen", "access blocked"],
        "controls_to_check": ["Fraud alert decision trace", "Manual fraud review queue", "Card block and release turnaround"],
    },
    {
        "root_cause_category": "Payment posting or transaction processing error",
        "root_cause_code": "payment_posting_or_transaction_processing_error",
        "description": "A transaction, payment, reversal, or settlement may have been posted incorrectly or failed somewhere in processing.",
        "business_summary": "Likely operational issue in payment posting, settlement mapping, or reversal handling.",
        "cues": ["misapplied", "declined", "returned", "payment not applied", "processing", "posted wrong", "reversal"],
        "controls_to_check": ["Payment posting logs", "Settlement mapping checks", "Reversal workflow audit"],
    },
    {
        "root_cause_category": "Billing or fee dispute investigation gap",
        "root_cause_code": "billing_or_fee_dispute_investigation_gap",
        "description": "A duplicate charge, fee dispute, or refund request may not have been investigated or corrected through the standard dispute workflow.",
        "business_summary": "Likely weakness in dispute intake, fee review, or refund decisioning.",
        "cues": ["charged twice", "duplicate", "fee", "refund", "billing dispute", "overcharged", "late fee"],
        "controls_to_check": ["Chargeback and reversal logic", "Fee reconciliation review", "Dispute handling procedure"],
    },
    {
        "root_cause_category": "Notice or disclosure delivery failure",
        "root_cause_code": "notice_or_disclosure_delivery_failure",
        "description": "Required notices, disclosures, or account communications may have been missing, unclear, or not delivered through the right channel.",
        "business_summary": "Likely issue in customer communications, notice generation, or delivery tracking.",
        "cues": ["notice", "disclosure", "terms", "not provided", "unclear", "didn't receive", "no notice"],
        "controls_to_check": ["Template version control", "Delivery channel logs", "Customer communications review"],
    },
    {
        "root_cause_category": "Customer service follow-up failure",
        "root_cause_code": "customer_service_follow_up_failure",
        "description": "The complaint may reflect delayed response, missed callbacks, or lack of follow-up after the customer reported the issue.",
        "business_summary": "Likely service breakdown in complaint handling, callback management, or case ownership.",
        "cues": ["no response", "no one called", "no follow up", "never heard back", "customer service", "ignored"],
        "controls_to_check": ["Case ownership queue", "Callback and follow-up SLA", "Complaint response audit trail"],
    },
    {
        "root_cause_category": "Merchant or third-party dispute outside direct bank control",
        "root_cause_code": "merchant_or_third_party_dispute_outside_direct_bank_control",
        "description": "The core issue may stem from a merchant, service provider, or external party rather than an internal bank processing failure.",
        "business_summary": "Likely merchant-side fulfillment, quality, or third-party service dispute requiring the right remediation path.",
        "cues": ["merchant", "seller", "defective", "never delivered", "store", "vendor", "third party", "quality issue"],
        "controls_to_check": ["Charge dispute eligibility", "Merchant evidence requirements", "External escalation path"],
    },
]


_ROOT_CAUSE_DISPLAY_BY_KEY = {
    str(entry.get("root_cause_category") or "").strip(): entry.get("root_cause_category") or ""
    for entry in ROOT_CAUSE_CONTROLS
}
_ROOT_CAUSE_DISPLAY_BY_KEY.update(
    {
        str(entry.get("root_cause_code") or "").strip(): entry.get("root_cause_category") or ""
        for entry in ROOT_CAUSE_CONTROLS
        if entry.get("root_cause_code")
    }
)


def format_root_cause_category(value: str | None) -> str:
    """Return a business-readable root cause label for UI surfaces."""
    raw = (value or "").strip()
    if not raw:
        return ""
    if raw in _ROOT_CAUSE_DISPLAY_BY_KEY:
        return _ROOT_CAUSE_DISPLAY_BY_KEY[raw]
    slug = re.sub(r"[_\s]+", " ", raw).strip()
    return slug[:1].upper() + slug[1:]


class MockCompanyPack:
    company_id: str = MOCK_COMPANY_ID
    company_profile: dict = COMPANY_PROFILE
    operational_taxonomy: dict = OPERATIONAL_TAXONOMY
    severity_rubric: list = SEVERITY_RUBRIC
    policy_snippets: list = POLICY_SNIPPETS
    routing_matrix: dict = ROUTING_MATRIX
    root_cause_controls: list = ROOT_CAUSE_CONTROLS
