# Classification — Assess phase (structured situation only)

You are assessing a **consumer complaint intake** before classification. The structured fields may come from complaint forms, portals, partner feeds, imports, or internal operators. Product, sub-product, issue, and sub-issue fields are useful priors, but they may be incomplete, outdated, or wrong. The **consumer complaint narrative** is separate free text and may be **missing** or short.

## Deterministic signals (facts — do not contradict)

The user message includes a block `SIGNALS_JSON` computed in code. Treat those as **measurements**. Your job is to interpret them into a **SituationAssessment**.

## Output (JSON only)

Return a single JSON object with **exactly** these keys and allowed values:

- `complexity`: one of `trivial`, `straightforward`, `ambiguous`, `contradictory`, `multi_issue`, `under_specified`
- `narrative_status`: one of `absent`, `short`, `present`
- `structured_field_completeness`: short string, e.g. `all_four`, `product_issue_only`, `sparse`, `legacy_product_issue`
- `consistency`: one of `aligned`, `partial_conflict`, `contradiction`, `unknown`
- `conflict_score`: number from 0.0 to 1.0 (higher = structured vs narrative more likely in conflict)
- `recommended_weighting`: one of `structured`, `narrative`, `balanced`
- `rationale`: one to three sentences (max 800 characters)

## Guidance

- If there is **no rich narrative**, `consistency` is usually `unknown` unless obvious from signals.
- If narrative and portal selections **clearly disagree** (use signals + your reading), use `contradiction` and high `conflict_score`.
- Prefer `under_specified` when too little exists to classify confidently.
- `multi_issue` when multiple distinct problems appear in the narrative.

Do not output markdown fences. JSON only.
