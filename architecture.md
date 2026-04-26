# Architecture: Regulatory-Grade Complaint Knowledge Base for Risk and Root-Cause Agents

## 1) Purpose

This document defines the production architecture for building a **regulatory-grade complaint knowledge base** that equips a fintech complaint-handling system with the depth required to perform **root-cause analysis** and **risk analysis** on real complaints.

This is **not** a toy RAG prototype and **not** a generic “finance chatbot” knowledge base.

The objective is to build a knowledge system that allows specialized agents to reason with:

- current federal consumer-finance regulations
- official interpretations and amendments
- supervisory and examination guidance
- real product agreements and disclosures
- company-specific policy and control mappings
- historical complaint patterns and precedent signals

The end state is an internal specialist that is **surgically equipped** before making any statement about:
- regulatory exposure
- consumer harm
- control breakdowns
- operational ownership
- remediation urgency
- escalation risk

---

## 2) Business objective

We already have a hub-and-spoke complaint system:

- **Supervisor** (hub)
- **Classification Agent**
- **Risk Agent**
- **Root-Cause Agent**
- **Routing Agent**
- **Resolve Agent**
- **Review Agent**

The **Classification Agent** already has a rich product/sub-product taxonomy.

This architecture focuses on building the knowledge base needed by:
- **Risk Agent**
- **Root-Cause Agent**

These agents must reason using both:
1. **canonical regulatory truth**
2. **historical complaint behavior and precedent**

The system must support real fintech complaint workflows, where incorrect legal or risk commentary is unacceptable.

---

## 3) Design principles

### 3.1 Canonical truth must come from official sources
The graph that defines products, laws, obligations, deadlines, and regulators must be built from official regulatory and supervisory sources, not from complaint labels or generic web pages.

### 3.2 Complaint history is a signal layer, not the law layer
CFPB complaint history is valuable, but it must not define the law graph. It is used as:
- precedent
- pattern memory
- trend signal
- company response signal
- weak supervision
- consumer-language behavior layer

### 3.3 LightRAG is an evidence and retrieval layer, not the sole source of truth
LightRAG should index and expose evidence across large text corpora, but the canonical graph should be curated and deterministic.

### 3.4 Every agent conclusion must be source-backed
Any final risk or root-cause assessment must be traceable to:
- source document
- section or clause
- effective date / version
- extraction lineage
- decision path

### 3.5 Versioning and effective dating are first-class requirements
Financial regulation changes over time. The system must preserve:
- document versions
- effective dates
- amendment lineage
- superseded sections
- complaint-time applicability where relevant

### 3.6 Internal policy must stay separate from public regulation
The system must distinguish between:
- public legal obligations
- supervisory expectations
- institution-specific policy/SOP
- product-specific disclosures/agreements
- learned precedent from historical complaints

---

## 4) What this knowledge base must do

The knowledge base must support the following question types reliably:

### 4.1 Risk Agent questions
- Which rules and regulations may govern this complaint?
- Which obligations may already be triggered?
- What facts are still missing before a legal-risk opinion can be made?
- What kind of consumer harm is alleged or implied?
- How severe is the likely risk?
- Does this complaint resemble historically elevated-risk patterns?
- Is there potential for systemic escalation or regulator scrutiny?

### 4.2 Root-Cause Agent questions
- What failure mode best explains this complaint?
- Which process/control likely failed?
- Which team/process area likely owns the breakdown?
- Is this an isolated incident or part of a repeated complaint cluster?
- What evidence should be gathered to confirm the root cause?
- What remediation or corrective action should be recommended?

### 4.3 Supervisor / Review support questions
- Is the current answer grounded in official sources?
- What confidence is justified?
- Which cited rules are binding, interpretive, supervisory, or internal?
- Has the agent mixed allegation with fact?

---

## 5) System overview

The architecture has **four major knowledge layers**:

1. **Canonical Regulatory Graph**
2. **Supervisory / Control Graph**
3. **Complaint Precedent Graph**
4. **LightRAG Evidence Layer**

These layers are connected but serve different purposes.

```text
                    ┌─────────────────────────────┐
                    │   Official Regulatory Data  │
                    │ CFR / CFPB / OCC / FDIC /   │
                    │ FFIEC / FinCEN / FR / etc.  │
                    └──────────────┬──────────────┘
                                   │
                         deterministic extraction
                                   │
                    ┌──────────────▼──────────────┐
                    │ Canonical Regulatory Graph  │
                    │ products, regs, sections,   │
                    │ obligations, deadlines,     │
                    │ regulators, evidence needs  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ Supervisory / Control Graph │
                    │ exam procedures, CMS,       │
                    │ control failures, risk      │
                    │ indicators, remediation     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ Complaint Precedent Graph   │
                    │ CFPB complaint history,     │
                    │ patterns, clusters, company │
                    │ responses, timeliness       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     LightRAG Evidence       │
                    │ manuals, interpretations,   │
                    │ narratives, agreements,     │
                    │ enforcement/guidance,       │
                    │ graph-enhanced retrieval    │
                    └──────────────┬──────────────┘
                                   │
                       queried by specialized agents
                                   │
     ┌─────────────────────────────┼─────────────────────────────┐
     │                             │                             │
┌────▼────┐                  ┌─────▼─────┐                 ┌────▼─────┐
│ Risk    │                  │ RootCause │                 │ Review    │
│ Agent   │                  │ Agent     │                 │ Agent     │
└─────────┘                  └───────────┘                 └──────────┘
```

---

## 6) Knowledge layers in detail

## 6.1 Canonical Regulatory Graph

This is the **source of truth** for:
- product definitions
- covered entities
- regulator mappings
- laws and implementing regulations
- sections and official interpretations
- obligations and deadlines
- evidence requirements
- exclusions / exceptions

This graph must be **curated and deterministic**.

### Example nodes
- `Product`
- `SubProduct`
- `IssueType`
- `Regulator`
- `Law`
- `Regulation`
- `RegulationPart`
- `RegSection`
- `OfficialInterpretation`
- `Obligation`
- `Deadline`
- `DisclosureRequirement`
- `EvidenceRequirement`
- `CoveredEntityType`
- `Exception`
- `ModelForm`

### Example edges
- `Product -> governed_by -> Regulation`
- `Regulation -> contains -> RegSection`
- `RegSection -> imposes -> Obligation`
- `Obligation -> requires -> EvidenceRequirement`
- `Obligation -> has_deadline -> Deadline`
- `RegSection -> interpreted_by -> OfficialInterpretation`
- `Obligation -> applies_to -> CoveredEntityType`
- `Obligation -> excludes -> Exception`

---

## 6.2 Supervisory / Control Graph

This graph turns regulation into operational and compliance-management context.

It captures:
- compliance management expectations
- exam procedures
- control themes
- known risk indicators
- consumer harm framing
- remediation patterns
- service-provider oversight expectations
- risk-management expectations

### Example nodes
- `ExamManual`
- `ExamProcedure`
- `ControlDomain`
- `Control`
- `FailureMode`
- `ConsumerHarmType`
- `RiskIndicator`
- `EscalationSignal`
- `RemediationAction`
- `OwningFunction`
- `ComplianceManagementElement`

### Example edges
- `ExamProcedure -> evaluates -> Control`
- `Control -> mitigates -> FailureMode`
- `FailureMode -> creates -> ConsumerHarmType`
- `FailureMode -> raises -> RiskIndicator`
- `RiskIndicator -> suggests -> EscalationSignal`
- `FailureMode -> owned_by -> OwningFunction`
- `RemediationAction -> addresses -> FailureMode`

This graph is critical for the **Root-Cause** and **Risk** agents because regulation alone rarely tells you the likely operational breakdown.

---

## 6.3 Complaint Precedent Graph

This graph is derived primarily from **CFPB complaint history** and later can be extended with internal complaint data.

It is **not** the law graph.

It captures:
- how consumers actually describe problems
- historical issue clusters
- company response patterns
- complaint channel behavior
- timeliness behavior
- geography/tags
- trend and anomaly signals
- recurring failure signatures

### Example nodes
- `Complaint`
- `ComplaintPattern`
- `NarrativeCluster`
- `WeakProductLabel`
- `WeakIssueLabel`
- `Company`
- `ResponseType`
- `TimelinessStatus`
- `Channel`
- `Tag`
- `State`
- `TimeWindow`
- `LikelyFailureMode`

### Example edges
- `Complaint -> mentions -> ComplaintPattern`
- `Complaint -> submitted_via -> Channel`
- `Complaint -> against -> Company`
- `Complaint -> weakly_labeled_as -> WeakProductLabel`
- `Complaint -> resulted_in -> ResponseType`
- `Complaint -> has_timeliness -> TimelinessStatus`
- `Complaint -> belongs_to -> NarrativeCluster`
- `NarrativeCluster -> suggests -> LikelyFailureMode`

The complaint precedent graph is used as an **observational truth layer** and a **behavioral signal layer**.

---

## 6.4 LightRAG Evidence Layer

LightRAG is used to index large document corpora and expose:
- graph-enhanced retrieval
- document indexing
- knowledge-graph exploration
- local and broader retrieval context
- passage-level evidence access

This layer is where long-form evidence lives:
- exam manuals
- official interpretations
- final rules
- complaint narratives
- product agreements
- public responses
- internal policies/SOPs
- remediation playbooks
- precedent documents

### Important boundary
LightRAG-generated graphs and extracted relationships should be treated as:
- evidence graph
- retrieval graph
- exploration graph

They should **not automatically overwrite** the curated canonical graph.

---

## 7) Source inventory: official data to ingest

This section lists the **production source inventory** from which raw files should be acquired.

---

## 7.1 CFPB complaint and taxonomy sources

### A. CFPB Consumer Complaint Database
Use for:
- complaint history
- precedent graph
- company response patterns
- timeliness signals
- narrative clustering
- trend/anomaly analysis
- weak supervision for complaint interpretation

Official source:
- https://www.consumerfinance.gov/data-research/consumer-complaints/

Why it matters:
- downloadable complaint data
- CSV / JSON export
- real complaint ecosystem behavior
- company response metadata
- complaint channels, timing, geography, tags

### B. CFPB complaint data use / field definitions
Use for:
- field semantics
- trust hierarchy
- understanding which fields are consumer-provided vs process-derived

Official sources:
- https://www.consumerfinance.gov/complaint/data-use/
- https://cfpb.github.io/api/ccdb/fields.html
- https://cfpb.github.io/api/ccdb/release-notes.html

Why it matters:
- `Product`, `Sub-product`, `Issue`, `Sub-issue` are consumer-identified
- process fields such as response/timeliness/date/company are stronger operational facts
- release notes help preserve historical interpretation of taxonomy changes

### C. CFPB complaint product coverage
Use for:
- initial complaint-product universe
- external complaint intake scope alignment

Official source:
- https://www.consumerfinance.gov/complaint/

Why it matters:
- provides the publicly accepted complaint product categories

---

## 7.2 CFPB regulations and interpretations

### A. CFPB regulations portal
Use for:
- bureau regulations under Chapter X
- version navigation
- section-level content
- official interpretations
- model forms / appendices

Official sources:
- https://www.consumerfinance.gov/rules-policy/regulations/
- https://www.consumerfinance.gov/rules-policy/final-rules/code-federal-regulations/

High-priority regulation packs for initial build:
- Regulation B (ECOA)
- Regulation E (EFTA / remittance / prepaid)
- Regulation F (FDCPA)
- Regulation P (Privacy of Consumer Financial Information)
- Regulation V (FCRA)
- Regulation X (RESPA / servicing)
- Regulation Z (TILA / credit cards / lending disclosures)
- Regulation DD (Truth in Savings)

Why they matter:
- these provide the direct complaint-handling obligations for many fintech complaint classes

### B. Section pages and official interpretation pages
Use for:
- structured section-level extraction
- official commentary / interpretation
- model clauses and forms where relevant

Examples:
- https://www.consumerfinance.gov/rules-policy/regulations/1005/
- https://www.consumerfinance.gov/rules-policy/regulations/1026/
- https://www.consumerfinance.gov/rules-policy/regulations/1022/
- https://www.consumerfinance.gov/rules-policy/regulations/1006/
- https://www.consumerfinance.gov/rules-policy/regulations/1024/
- https://www.consumerfinance.gov/rules-policy/regulations/1002/
- https://www.consumerfinance.gov/rules-policy/regulations/1030/

---

## 7.3 GovInfo and Federal Register sources

These are necessary for authoritative machine-readable regulatory ingestion and freshness tracking.

### A. GovInfo CFR bulk XML and API
Use for:
- canonical machine-readable CFR ingestion
- structured parsing
- version-controlled source snapshots
- authoritative section identifiers

Official sources:
- https://www.govinfo.gov/app/collection/cfr/
- https://www.govinfo.gov/bulkdata/
- https://www.govinfo.gov/developers
- https://www.govinfo.gov/help/cfr

Why it matters:
- provides bulk XML for the annual CFR
- gives programmatic access to metadata and packages
- should be treated as the structured canonical input for regulation text snapshots

### B. FederalRegister.gov API and XML-backed rule tracking
Use for:
- amendments
- effective-date tracking
- proposed/final rule lineage
- current change monitoring between annual CFR releases

Official sources:
- https://www.federalregister.gov/developers/documentation/api/v1
- https://www.federalregister.gov/reader-aids/developer-resources
- https://www.govinfo.gov/help/fr

Why it matters:
- the annual CFR alone is not enough for freshness
- amendments and effective-date changes must be tracked
- helps maintain time-aware regulatory applicability

---

## 7.4 Supervisory and examination sources

These are critical for **risk** and **root-cause** because they translate regulations into examination logic and control expectations.

### A. CFPB Supervision and Examination Manual
Use for:
- supervisory expectations
- product-line examination context
- compliance-management framing
- examiner logic
- consumer harm and risk assessment framing

Official sources:
- https://www.consumerfinance.gov/compliance/supervision-examinations/
- https://files.consumerfinance.gov/f/documents/cfpb_supervision-and-examination-manual.pdf

### B. CFPB Compliance Management Review Examination Procedures
Use for:
- control and CMS evaluation
- board and management oversight
- compliance program
- service-provider oversight
- violations of law and consumer harm
- examiner conclusions

Official source:
- https://www.consumerfinance.gov/compliance/supervision-examinations/compliance-management-review-examination-procedures/

### C. OCC Comptroller’s Handbook
Use for:
- bank supervision risk logic
- compliance management systems
- consumer compliance risk framing
- UDAAP examination framing

Official sources:
- https://www.occ.treas.gov/publications-and-resources/publications/comptrollers-handbook/index-comptrollers-handbook.html
- https://www.occ.treas.gov/publications-and-resources/publications/comptrollers-handbook/files/compliance-mgmt-systems/index-compliance-management-systems.html
- https://www.occ.treas.gov/publications-and-resources/publications/comptrollers-handbook/files/unfair-deceptive-act/index-udaap.html

### D. FDIC Consumer Compliance Examination Manual
Use for:
- consumer harm evaluation
- CMS expectations
- public examination guidance
- supervisory structure and updates

Official sources:
- https://www.fdic.gov/consumer-compliance-examination-manual
- https://www.fdic.gov/consumer-compliance-examination-manual/table-contents

### E. FFIEC BSA/AML Examination Manual and procedures
Use for:
- AML/CFT risk frameworks
- CIP/KYC logic
- transaction monitoring / SAR / OFAC-related procedures
- money services business and funds-transfer related risk

Official sources:
- https://bsaaml.ffiec.gov/manual
- https://bsaaml.ffiec.gov/examprocedures
- https://www.ffiec.gov/

### F. FinCEN guidance
Use for:
- BSA/AML interpretive guidance
- industry- or customer-type-specific risk
- advisories and clarifications

Official source:
- https://www.fincen.gov/resources/statutes-regulations/guidance

---

## 7.5 Product agreement and disclosure sources

These are essential because many complaints hinge on:
- fee disclosure
- authorization terms
- dispute rights language
- servicing language
- product-specific consumer expectations

### A. CFPB credit card agreement database
Use for:
- real issuer agreement text
- fee/disclosure clauses
- product terms
- issuer-specific public documents

Official sources:
- https://www.consumerfinance.gov/credit-cards/agreements/
- https://www.consumerfinance.gov/credit-cards/agreements/archive/
- https://www.consumerfinance.gov/data-research/credit-card-data/

### B. CFPB prepaid account agreements
Use for:
- prepaid product agreement ingestion
- fee and dispute terms
- product comparison
- agreement metadata and raw files

Official sources:
- https://www.consumerfinance.gov/data-research/prepaid-accounts/
- https://www.consumerfinance.gov/data-research/prepaid-accounts/download-agreements/

---

## 7.6 Internal institutional sources (company-owned)

These are not public, but must be part of the production knowledge base.

Use for:
- routing logic
- escalation thresholds
- refund / credit authority matrix
- fraud workflows
- KYC/CIP policy
- dispute operations SOPs
- complaint handling SLAs
- Jira ownership mappings
- remediation playbooks
- response templates
- audit controls
- regulator-communication policies
- product catalogs
- fee schedules
- internal glossary

These must be kept in a separate controlled domain with access restrictions.

---

## 8) Trust hierarchy of sources

The system must not treat all sources equally.

### Tier 1: Canonical / binding or primary structured source
- GovInfo CFR bulk XML / metadata
- CFPB regulation pages and official interpretation pages
- official rule/amendment sources from Federal Register and GovInfo
- official agency examination manuals and procedures

### Tier 2: Official but contextual / supervisory / interpretive
- CFPB exam manual
- OCC handbook
- FDIC manual
- FFIEC BSA/AML procedures
- FinCEN guidance
- official agreement databases

### Tier 3: Observational / precedent / market behavior
- CFPB complaint history
- internal complaint history
- public company responses
- complaint trend metrics

### Tier 4: Internal policy / institution-specific truth
- SOPs
- routing policies
- service-provider controls
- operating procedures
- remediation playbooks
- product catalog and policy mappings

The Risk and Root-Cause agents must know which tier they are citing.

---

## 9) Data stores and physical architecture

Recommended production layout:

### 9.1 Raw object store
Purpose:
- preserve immutable source files
- enable reprocessing
- audit acquisition lineage

Contents:
- PDFs
- XML
- HTML
- CSV
- JSON
- agreement files
- manual downloads

Recommended layout:
```text
data/raw/
  cfpb/
    complaints/
    regulations/
    supervision/
    agreements/
  govinfo/
    cfr/
  federal_register/
  occ/
  fdic/
  ffiec/
  fincen/
  internal/
```

### 9.2 Normalized document store
Purpose:
- consistent document schema
- sectionized text
- metadata normalization
- chunk lineage

Recommended layout:
```text
data/normalized/
  docs/
  sections/
  chunks/
  metadata/
```

### 9.3 Canonical graph database
Purpose:
- deterministic ontology
- section/obligation relationships
- policy/control mappings
- query-time reasoning

Recommended technology:
- Neo4j or PostgreSQL + graph layer
- Neo4j is preferred for relationship-heavy traversal
- PostgreSQL remains useful for metadata, audit, and analytical tables

### 9.4 Analytical store / warehouse
Purpose:
- complaint statistics
- trend tables
- cluster outputs
- feature generation
- monitoring dashboards

Recommended technology:
- PostgreSQL / DuckDB / warehouse depending on scale

### 9.5 Vector + retrieval store
Purpose:
- semantic retrieval
- passage recall
- supporting evidence fetch

Recommended options:
- LightRAG-managed storage for retrieval layer
- optional OpenSearch / pgvector / external vector store depending on deployment

### 9.6 LightRAG service
Purpose:
- document indexing
- graph-enhanced retrieval
- exploration UI
- evidence retrieval API
- subgraph inspection

---

## 10) Canonical graph schema

The canonical graph should be built around the unit:

`product -> issue context -> regulation -> obligation -> evidence -> risk/control relevance`

### 10.1 Core node types

#### Product domain
- `Product`
- `SubProduct`
- `Feature`
- `LifecycleStage`
- `FeeType`
- `DisclosureArtifact`

#### Regulatory domain
- `Regulator`
- `Law`
- `Regulation`
- `RegulationPart`
- `RegSection`
- `OfficialInterpretation`
- `ModelForm`
- `Amendment`
- `EffectivePeriod`

#### Compliance obligation domain
- `Obligation`
- `Deadline`
- `TriggerCondition`
- `EvidenceRequirement`
- `Exception`
- `ConsumerRight`
- `RequiredCommunication`
- `CoveredEntityType`

#### Supervisory domain
- `ExamManual`
- `ExamProcedure`
- `ControlDomain`
- `Control`
- `RiskIndicator`
- `ConsumerHarmType`
- `FailureMode`
- `RemediationAction`
- `OwningFunction`

#### Complaint domain
- `Complaint`
- `ComplaintPattern`
- `NarrativeCluster`
- `WeakProductLabel`
- `WeakIssueLabel`
- `ResponseType`
- `TimelinessStatus`
- `Channel`
- `Tag`
- `Geography`
- `Company`

#### Internal domain
- `InternalPolicy`
- `Playbook`
- `RoutingRule`
- `EscalationRule`
- `Team`
- `SLA`
- `CaseAction`

---

## 10.2 Core relationship types

### Product and regulation
- `(:Product)-[:HAS_SUBPRODUCT]->(:SubProduct)`
- `(:SubProduct)-[:COMMONLY_RELATES_TO]->(:IssueType)`
- `(:Product)-[:GOVERNED_BY]->(:Regulation)`
- `(:Regulation)-[:CONTAINS]->(:RegSection)`
- `(:RegSection)-[:INTERPRETED_BY]->(:OfficialInterpretation)`

### Obligations and evidence
- `(:RegSection)-[:IMPOSES]->(:Obligation)`
- `(:Obligation)-[:TRIGGERED_BY]->(:TriggerCondition)`
- `(:Obligation)-[:REQUIRES_EVIDENCE]->(:EvidenceRequirement)`
- `(:Obligation)-[:HAS_DEADLINE]->(:Deadline)`
- `(:Obligation)-[:HAS_EXCEPTION]->(:Exception)`
- `(:Obligation)-[:CREATES_RIGHT]->(:ConsumerRight)`

### Supervisory / control mapping
- `(:ExamProcedure)-[:EVALUATES]->(:Control)`
- `(:Control)-[:MITIGATES]->(:FailureMode)`
- `(:FailureMode)-[:CAUSES]->(:ConsumerHarmType)`
- `(:FailureMode)-[:RAISES]->(:RiskIndicator)`
- `(:FailureMode)-[:OWNED_BY]->(:OwningFunction)`
- `(:RemediationAction)-[:ADDRESSES]->(:FailureMode)`

### Complaint precedent mapping
- `(:Complaint)-[:AGAINST]->(:Company)`
- `(:Complaint)-[:SUBMITTED_VIA]->(:Channel)`
- `(:Complaint)-[:WEAKLY_LABELED_AS]->(:WeakProductLabel)`
- `(:Complaint)-[:HAS_RESPONSE_TYPE]->(:ResponseType)`
- `(:Complaint)-[:HAS_TIMELINESS]->(:TimelinessStatus)`
- `(:Complaint)-[:BELONGS_TO]->(:NarrativeCluster)`
- `(:NarrativeCluster)-[:SUGGESTS]->(:FailureMode)`

### Internal operational mapping
- `(:FailureMode)-[:ROUTED_TO]->(:Team)`
- `(:Team)-[:FOLLOWS]->(:Playbook)`
- `(:RoutingRule)-[:APPLIES_TO]->(:IssueType)`
- `(:EscalationRule)-[:TRIGGERED_BY]->(:RiskIndicator)`
- `(:InternalPolicy)-[:CONSTRAINS]->(:RemediationAction)`

---

## 10.3 Example graph fragment

```text
Unauthorized EFT complaint
  -> Product: Checking Account
  -> Regulation: Reg E
  -> RegSection: Error resolution / liability / authorization-related sections
  -> Obligation: open investigation / request evidence / provide response within required framework
  -> EvidenceRequirement: transaction log, authorization evidence, notice date, account statement
  -> FailureMode: unauthorized transaction handling breakdown
  -> OwningFunction: disputes/fraud operations
  -> RiskIndicator: consumer harm + timing exposure + repeat pattern
```

---

## 11) What LightRAG does in this architecture

LightRAG is valuable here, but its role must be explicit.

## 11.1 Where LightRAG fits
LightRAG should index the **text-rich corpus**:
- regulation text
- official interpretations
- exam manuals
- complaint narratives
- agreement text
- company public responses
- internal policies / SOPs
- remediation playbooks
- internal investigation notes (if allowed)

Use LightRAG for:
- evidence retrieval
- entity and relationship exploration
- graph-enhanced recall
- subgraph browsing
- user-facing exploration UI for analysts
- retrieval support for agent prompts

## 11.2 Where LightRAG does NOT fit
LightRAG should **not** be trusted by itself to define:
- canonical regulatory ontology
- final obligation logic
- exact effective-dated compliance mapping
- legal applicability without deterministic validation

The risk of relying purely on extracted graph relationships is that agent confidence may outrun source precision.

## 11.3 Recommended LightRAG integration pattern

### Step A: Normalize documents before LightRAG
Before ingesting documents into LightRAG:
- clean OCR / encoding
- preserve section boundaries
- attach metadata
- attach source tier
- attach document type
- attach regulator
- attach effective date / publication date if known

### Step B: Index only normalized documents
Each indexed record should include metadata like:
```json
{
  "doc_id": "cfpb_reg_e_2026_current",
  "source_family": "cfpb_regulation",
  "source_tier": 1,
  "regulator": "CFPB",
  "law_scope": ["EFTA"],
  "regulation_scope": ["Regulation E"],
  "document_type": "regulation",
  "effective_date": "2026-01-01",
  "version_label": "current",
  "jurisdiction": "US",
  "product_scope": ["checking account", "prepaid account", "remittance transfer"],
  "section_path": ["1005", "11"],
  "citation_anchor": "12 CFR 1005.11"
}
```

### Step C: Use LightRAG for retrieval, then validate against curated graph
Agent flow:
1. retrieve candidate evidence from LightRAG
2. map entities to canonical graph IDs
3. validate obligations and applicability using canonical graph and rules
4. compose answer with citations and confidence

## 11.4 Why this separation matters
This gives you:
- flexible exploration
- high recall
- better evidence retrieval
- graph-assisted contextual search

without letting unverified extraction directly mutate the core compliance graph.

---

## 12) Ingestion pipelines

## 12.1 Raw acquisition pipeline

Every source family must be acquired into immutable raw storage with metadata.

For each raw file capture:
- source URL
- fetch timestamp
- checksum
- document title
- regulator / owner
- publication date
- effective date if present
- retrieval method
- file type
- version label
- source tier

### Acquisition jobs by source family
- `fetch_cfpb_complaints`
- `fetch_cfpb_regulations`
- `fetch_cfpb_supervision`
- `fetch_occ_handbooks`
- `fetch_fdic_compliance_manual`
- `fetch_ffiec_bsaaml_manual`
- `fetch_fincen_guidance`
- `fetch_govinfo_cfr`
- `fetch_federal_register_updates`
- `fetch_cfpb_agreements`
- `fetch_internal_policy_docs`

---

## 12.2 Normalization pipeline

Normalize every raw file into a common schema.

### Normalized document schema
```json
{
  "doc_id": "",
  "source_family": "",
  "source_url": "",
  "source_tier": 1,
  "title": "",
  "regulator": "",
  "document_type": "",
  "publication_date": "",
  "effective_date": "",
  "version_label": "",
  "jurisdiction": "US",
  "product_scope": [],
  "law_scope": [],
  "raw_text": "",
  "sections": [],
  "citations": [],
  "checksum": "",
  "metadata": {}
}
```

### Section schema
```json
{
  "section_id": "",
  "doc_id": "",
  "section_path": [],
  "section_title": "",
  "section_text": "",
  "citation_anchor": "",
  "effective_period": {
    "start": "",
    "end": null
  }
}
```

---

## 12.3 Deterministic legal extraction pipeline

This is the most important pipeline for the canonical graph.

It should extract:

- regulation / part / section identity
- official interpretation links
- covered products
- covered entities
- obligations
- trigger conditions
- evidence requirements
- deadline language
- model forms / notices / clauses
- exceptions / exclusions
- amendment references
- effective periods

### Extraction strategy
1. rule-based parsing first
2. LLM-assisted structuring second
3. human review for uncertain items
4. write only reviewed/validated facts into canonical graph

### Why this matters
For compliance-facing systems, incorrect structure is more dangerous than missing structure.

---

## 12.4 Supervisory/control extraction pipeline

From exam manuals and procedures, extract:
- control domains
- exam objectives
- exam steps
- risk themes
- consumer harm language
- root-cause relevant control expectations
- remediation hints
- service-provider risk indicators
- management oversight expectations

These enrich the control graph.

---

## 12.5 Complaint precedent pipeline

From CFPB complaint data, build:
- complaint records
- narrative embeddings
- narrative clusters
- company response features
- timeliness features
- channel features
- geography features
- tag features
- time-series aggregates
- likely-failure-mode mappings

### Important handling rules
- treat `Product`, `Sub-product`, `Issue`, `Sub-issue` as weak labels
- do not use them as canonical truth
- preserve taxonomy version where possible
- use narratives only when consumer consent is present
- maintain clear distinction between allegation and fact

---

## 12.6 Product agreement extraction pipeline

From credit card and prepaid agreement sources, extract:
- fees
- disclosures
- dispute-related terms
- authorization terms
- limitation or communication clauses
- servicing or support contact mechanisms
- issuer identifiers
- agreement effective/version metadata

These should be linked to:
- product / sub-product
- issuer / institution
- relevant regulations
- complaint themes they often drive

---

## 13) How CFPB complaint history helps the agents

This is a critical section because complaint history is often misunderstood.

CFPB complaint history does **not** define legal truth.

It helps in six production-critical ways:

### 13.1 Real consumer language
Consumers do not describe problems using legal or internal taxonomy language. Historical complaints teach the system:
- real-world complaint phrasing
- slang / informal problem framing
- ambiguity patterns
- product confusion patterns

This improves:
- intake understanding
- retrieval
- complaint-to-failure-mode mapping

### 13.2 Pattern and cluster discovery
Historical complaints reveal recurring patterns such as:
- unauthorized transaction complaints
- payment posting issues
- dispute investigation failures
- fee disclosure mismatch complaints
- debt collection communication issues
- tradeline correction failures
- servicing communication breakdowns

This helps Root-Cause identify repeated operational failure modes.

### 13.3 Company response behavior
CFPB data includes company response fields and timeliness fields.

This helps Risk reason about:
- how similar complaints are typically answered
- whether response patterns suggest weak complaint handling
- whether certain issues trend toward explanations vs relief
- whether response delay patterns suggest operational stress

### 13.4 Trend and anomaly monitoring
Historical complaint time series can surface:
- sudden spikes
- product-specific surges
- company-specific concentration
- geography-specific anomalies
- complaint channel shifts

This is useful for systemic risk and escalation detection.

### 13.5 Weak supervision for complaint interpretation
Even though product/issue fields are not perfect truth, they still provide weak labels useful for:
- training classifiers
- complaint similarity
- query expansion
- detection of customer mental-model mismatch

### 13.6 Linking complaint patterns to failure modes
By clustering narratives and correlating with supervisory/control knowledge, we can map:
`complaint pattern -> likely failure mode -> likely owning team -> likely remediation`

That is exactly what the Root-Cause Agent needs.

---

## 14) Agent usage patterns

## 14.1 Risk Agent usage pattern

### Inputs
- classified product / sub-product
- issue hypothesis
- institution type
- consumer harm allegations
- timeline
- evidence currently available
- jurisdiction / geography if relevant

### Query path
1. query canonical graph for product-governing regulations
2. fetch applicable obligations and deadlines
3. query supervisory/control graph for risk indicators and harm themes
4. query complaint precedent graph for analogous complaint clusters and response/timeliness patterns
5. use LightRAG to retrieve supporting passages and interpretations
6. produce structured risk assessment

### Outputs
- governing regulatory candidates
- triggered obligations
- missing facts
- consumer harm level
- regulatory risk level
- operational risk level
- systemic / reputational escalation flags
- evidence citations
- confidence / uncertainty statement

---

## 14.2 Root-Cause Agent usage pattern

### Inputs
- complaint facts
- narrative
- classified taxonomy output
- current process state
- available evidence
- institution and product context

### Query path
1. query complaint precedent graph for similar complaint clusters
2. map cluster to candidate failure modes
3. query supervisory/control graph for relevant control expectations
4. retrieve evidence from manuals, policies, agreements, and prior cases via LightRAG
5. validate against canonical regulatory obligations
6. propose ranked root-cause hypotheses

### Outputs
- likely failure mode(s)
- probable broken control(s)
- likely owning function/team
- evidence needed to confirm
- recommended immediate remediation path
- confidence / alternative hypotheses

---

## 14.3 Review Agent usage pattern
Review checks:
- unsupported legal claims
- misuse of complaint labels as truth
- outdated citations
- internal policy vs public law confusion
- evidence gaps
- overconfident language

---

## 15) Time, freshness, and versioning

This system must always know:
- what version of a rule it is using
- whether an interpretation was current on the complaint date
- whether the cited policy was current at the time
- whether the agreement version aligns with the product period

### Required fields
- `publication_date`
- `effective_date`
- `replaced_by`
- `supersedes`
- `valid_from`
- `valid_to`
- `version_label`

### Operational policy
- nightly or scheduled Federal Register checks
- periodic CFPB regulation refresh
- periodic manual refresh
- agreement refresh by source cadence
- full lineage audit trail

---

## 16) Security, privacy, and regulatory hygiene

Because this is a fintech complaint system, the knowledge base must also satisfy operational hygiene.

### 16.1 PII / complaint data controls
- store only approved complaint fields in analytical systems
- isolate raw narratives if they contain sensitive information
- define access controls by role
- redact where necessary before broad retrieval exposure

### 16.2 Internal-vs-public separation
Internal policies and case notes must not be mixed into general retrieval unless access policy allows it.

### 16.3 Provenance
Every extracted fact must be traceable to:
- source document
- section
- extraction version
- reviewer / validation status

### 16.4 Auditability
Log:
- retrieval set used by agent
- graph facts used
- final citations
- confidence statement
- missing-fact warnings

---

## 17) Recommended implementation sequence

## Phase 1: Build the canonical complaint-risk spine
1. ingest CFPB complaint database and fields metadata
2. ingest high-priority CFPB regulation packs:
   - Reg E
   - Reg Z
   - Reg V
   - Reg F
   - Reg X
   - Reg B
   - Reg DD
   - Reg P
3. ingest GovInfo CFR XML for corresponding sections
4. ingest Federal Register update pipeline
5. create canonical graph nodes for:
   - product
   - regulation
   - section
   - obligation
   - evidence requirement
   - deadline
6. create complaint precedent graph from CFPB history
7. connect Risk and Root-Cause agents to read-only graph queries

## Phase 2: Add supervisory and control intelligence
1. ingest CFPB Supervision Manual
2. ingest CMS review procedures
3. ingest OCC CMS + UDAAP materials
4. ingest FDIC compliance manual
5. ingest FFIEC BSA/AML materials
6. extract control domains, failure modes, risk indicators
7. connect Root-Cause agent to failure-mode/control graph

## Phase 3: Add agreement and product-document intelligence
1. ingest credit card agreement database
2. ingest prepaid agreement data
3. extract clauses, fees, disclosures, dispute terms
4. link agreement content to products/issues/failure modes

## Phase 4: Add internal policy and operations intelligence
1. ingest SOPs, routing policies, playbooks
2. map failure modes to teams, SLAs, escalation rules
3. enable Routing, Resolve, and Review agents to consume the same backbone

## Phase 5: Add state law / specialized expansions as needed
Depending on your fintech scope:
- state debt collection rules
- state servicing rules
- state money transmitter requirements
- card network rules
- NACHA operating rules
- remittance and payments-specific rule expansions
- crypto/virtual-currency adjacent regimes if actually in scope

---

## 18) Recommended repository / folder structure

```text
knowledge_base/
  architecture.md
  configs/
    sources.yaml
    extraction_rules.yaml
    graph_schema.yaml
    lightrag.yaml
  data/
    raw/
      cfpb/
      govinfo/
      federal_register/
      occ/
      fdic/
      ffiec/
      fincen/
      internal/
    normalized/
      docs/
      sections/
      chunks/
      metadata/
    derived/
      graph_exports/
      complaint_clusters/
      obligation_tables/
      embeddings/
  ingestion/
    fetch_cfpb_complaints.py
    fetch_cfpb_regulations.py
    fetch_govinfo_cfr.py
    fetch_federal_register.py
    fetch_occ.py
    fetch_fdic.py
    fetch_ffiec.py
    fetch_fincen.py
    fetch_cfpb_agreements.py
  parsers/
    parse_cfr_xml.py
    parse_cfpb_regulations.py
    parse_exam_manuals.py
    parse_agreements.py
    parse_complaints.py
  extraction/
    extract_obligations.py
    extract_deadlines.py
    extract_controls.py
    extract_failure_modes.py
    extract_agreement_clauses.py
  graph/
    load_canonical_graph.py
    load_precedent_graph.py
    load_control_graph.py
    query_templates/
  lightrag/
    build_corpus.py
    sync_documents.py
    query_adapter.py
  agents/
    risk/
    root_cause/
    routing/
    review/
  tests/
    source_tests/
    extraction_tests/
    graph_tests/
    retrieval_tests/
    agent_eval/
```

---

## 19) Output contract for production agent use

No agent should output raw prose only.

The Risk and Root-Cause agents should emit structured outputs like:

### Risk output contract
```json
{
  "complaint_id": "",
  "product": "",
  "sub_product": "",
  "regulatory_candidates": [],
  "triggered_obligations": [],
  "missing_facts": [],
  "risk_assessment": {
    "consumer_harm": "",
    "regulatory_risk": "",
    "operational_risk": "",
    "systemic_risk": ""
  },
  "evidence": [],
  "confidence": "",
  "review_required": true
}
```

### Root-cause output contract
```json
{
  "complaint_id": "",
  "candidate_failure_modes": [],
  "candidate_broken_controls": [],
  "likely_owning_functions": [],
  "required_confirmation_evidence": [],
  "recommended_immediate_actions": [],
  "evidence": [],
  "confidence": "",
  "alternative_hypotheses": []
}
```

---

## 20) Production success criteria

This knowledge base is successful when:

- the Risk Agent cites current and applicable rules rather than generic finance text
- the Root-Cause Agent proposes plausible control failures rather than narrative summaries
- complaint history is used as precedent/signal rather than mistaken for law
- LightRAG improves evidence retrieval without becoming the only truth source
- every material statement is auditable
- updates to rules and guidance can be ingested without breaking lineage
- internal teams trust the output enough to use it in real complaint operations

---

## 21) Final architecture position

The right architecture for this project is:

- **Curated canonical graph** for products, regulations, obligations, and evidence requirements
- **Supervisory/control graph** for failure modes, risk indicators, and remediation logic
- **CFPB complaint precedent graph** for real-world complaint behavior and trend signals
- **LightRAG evidence layer** for large-scale document retrieval and graph-enhanced evidence exploration

This separation is what makes the system both:
- **trustworthy**
- **operationally useful**

and prevents a real fintech complaint system from confusing:
- allegation with fact
- retrieval with law
- similarity with applicability
- extracted graph edges with compliance truth

That separation is the foundation for building a reliable specialist for complaint risk and root-cause analysis.
