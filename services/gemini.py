import asyncio
import json
import logging
import re
import time

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

RACM_SYSTEM_INSTRUCTION = """You are a SOX Compliance Auditor extracting Risk and Control Matrices from SOP documents.

DIRECTIVE 1 — COMPLETENESS
- Extract EVERY risk-control pair. Missing one = SOX audit failure.
- NEVER merge entries with different Control Owners, even if Risk Description is identical.
- Multiple risks sharing the same control (same owner+activity) → separate entries, same Control ID.

DIRECTIVE 2 — EXTRACT vs INFER

EXTRACT VERBATIM: Risk Description, Control Activity, Control description as per SOP, Risk Type (FR/OR as in SOP), Evidence/Source (include IPE refs).

EXTRACT WITH SECTION REFERENCE:
  Compliance Reference — MUST include specific SOP section number.
  Format: "{Document ID}, Section {X.Y.Z}" (e.g., "SOP-PROC-001, Section 5.3").
  If from a Key Controls subsection, reference that subsection.
  If spanning sections, list all (e.g., "SOP-PROC-001, Sections 6.4, 8.3").
  NEVER use just the document title for all entries.

EXTRACT WITH JUDGMENT:
  Control Frequency — MUST be specific and testable:
  "Per Transaction" | "Daily" | "Weekly" | "Monthly" | "Quarterly" |
  "Annually" | "Per Batch" | "Per Change" | "Continuous" | "As Needed"
  *** "Recurring" is NOT a valid frequency. NEVER output "Recurring". ***
  *** "Event based" is NOT valid. Use "Per Transaction" or "Per Change". ***
  *** "One-Time Activity" is NOT valid. Use "Annually" or "As Needed". ***
  Decision logic:
  - Triggers on each PR/PO/invoice/payment → "Per Transaction"
  - System validation at form submission → "Per Transaction"
  - Periodic report reviewed → use the review period ("Weekly", "Monthly")
  - Runs with payment batches → "Per Batch"
  - Annual budget cycle → "Annually"
  - Triggered by specific event (vendor bank change) → "Per Change"
  - Continuous monitoring tool → "Continuous"

  Control Owner — MUST be a human role:
  *** NEVER assign "ERP System", "System", "Automated", or "Banking system" as Control Owner. ***
  The Control Owner is the PERSON responsible for:
  (a) configuring the system rule, OR
  (b) monitoring the system output, OR
  (c) handling exceptions when the system flags an issue.
  Examples: "System-enforced budget check" → "Finance Controller".
  "Automated DOA routing" → "IT Administrator". "Banking dual auth" → "Finance Manager".

  Assertion Mapped — NEVER leave blank:
  For FR entries: financial statement assertions —
    "Existence" | "Completeness" | "Valuation" | "Rights & Obligations" |
    "Accuracy" | "Cutoff" | "Presentation & Disclosure"
  For OR entries: process-level assertions —
    "Authorization" | "Validity" | "Accuracy" | "Completeness" |
    "Timeliness" | "Security" | "Compliance"
  Multiple allowed, comma-separated. Old rule "BLANK for OR entries" is RETIRED.

MAY INFER: Control Type (Preventive/Detective/Corrective), Control Nature (Manual/Automated/IT-Dependent Manual), Risk Category, Control Objective, Testing Attributes, Risk Likelihood (vary — not all Medium), Risk Impact (descriptive: Financial Misstatement/Fraud/Compliance Violation/etc), Risk Rating (from Likelihood×Impact), Mitigation Effectiveness, Gaps/Weaknesses (write "None" explicitly if none found).

DIRECTIVE 3 — NULL RULES
- Unknown/not inferable → return "". Never fabricate.
- Assertion Mapped: NEVER blank. Use process assertions for non-FR entries.
- Gaps/Weaknesses: "None" if no gaps, never blank.
- Source Quote: verbatim substring for EXTRACTED entries. "" for INFERRED entries.

DIRECTIVE 4 — STRUCTURE
- Process Area/Sub-Process from document header. Risk IDs: R001, R002... Control IDs: C001, C002...
- Shared controls → different Risk IDs, same Control ID. Include IPE references.
- detailed_entries: one per risk-control pair. summary_entries: grouped by Process Area.

CONTROL TYPE vs CONTROL NATURE — DIFFERENT dimensions:
- Type = WHAT: Preventive | Detective | Corrective
- Nature = HOW: Manual | Automated | IT-Dependent Manual
- NEVER put "Manual" in Type. NEVER put "Preventive" in Nature.

KEY PATTERNS:
- FR entry: Risk Type="FR", Assertion Mapped="Existence, Valuation", Control Frequency="Per Transaction"
- OR entry: Risk Type="OR", Assertion Mapped="Authorization, Timeliness", Control Nature="Automated" if system-enforced
- Pre-approval review = Preventive/Manual. Post-event sampling = Detective. System config = Preventive/Automated.
"""

_RACM_ENTRY_PROPERTIES = {
    "Process Area": types.Schema(
        type=types.Type.STRING,
        description="The business process area from the document header or section title (e.g., Procurement, Payroll, Accounts Payable).",
    ),
    "Sub-Process": types.Schema(
        type=types.Type.STRING,
        description="The specific sub-process within the Process Area (e.g., Invoice Verification, Payment Release, Vendor Onboarding).",
    ),
    "Risk ID": types.Schema(
        type=types.Type.STRING,
        description="Sequential risk identifier: R001, R002, R003...",
    ),
    "Risk Description": types.Schema(
        type=types.Type.STRING,
        description="The exact risk text as stated in the SOP. Do not rephrase. Frame as 'Risk of...' or 'Risk that...'",
    ),
    "Risk Category": types.Schema(
        type=types.Type.STRING,
        enum=["Financial Reporting", "Operational", "Compliance", "Strategic"],
    ),
    "Risk Type": types.Schema(
        type=types.Type.STRING,
        enum=["FR", "OR"],
        description="FR = Financial Reporting risk (affects financial statements). OR = Operating Risk (operational/process risk).",
    ),
    "Control ID": types.Schema(
        type=types.Type.STRING,
        description="Sequential control identifier: C001, C002... Multiple risks sharing the SAME control (same owner+activity) get DIFFERENT Risk IDs but the SAME Control ID.",
    ),
    "Control Activity": types.Schema(
        type=types.Type.STRING,
        description="The exact control description as stated in the SOP. Do not rephrase.",
    ),
    "Control Objective": types.Schema(
        type=types.Type.STRING,
        description="What this control aims to achieve (e.g., 'Ensure accuracy of financial reporting', 'Prevent unauthorized payments', 'Detect discrepancies in reconciliation').",
    ),
    "Control Type": types.Schema(
        type=types.Type.STRING,
        enum=["Preventive", "Detective", "Corrective"],
    ),
    "Control Nature": types.Schema(
        type=types.Type.STRING,
        enum=["Manual", "Automated", "IT-Dependent Manual"],
    ),
    "Control Frequency": types.Schema(
        type=types.Type.STRING,
        enum=["Per Transaction", "Daily", "Weekly", "Monthly", "Quarterly", "Annually", "Per Batch", "Per Change", "Continuous", "As Needed"],
        description="Specific testable cadence. Never 'Recurring' or 'Event based'. Per-PR/PO/invoice → 'Per Transaction'. System validation at submit → 'Per Transaction'. Payment batches → 'Per Batch'. Vendor bank change → 'Per Change'.",
    ),
    "Control Owner": types.Schema(
        type=types.Type.STRING,
        description="Must be a human job title or role. NEVER a system name. NEVER 'ERP System', 'System', 'Automated', or 'Banking system'. For system-enforced controls, use the person who configures, monitors, or handles exceptions.",
    ),
    "Control description as per SOP": types.Schema(
        type=types.Type.STRING,
        description="The full verbatim control description paragraph from the SOP, including any conditions, thresholds, or exceptions mentioned.",
    ),
    "Testing Attributes": types.Schema(
        type=types.Type.STRING,
        description="What an auditor should test to verify this control operates effectively (e.g., 'Sample 25 transactions and verify approval signatures exist', 'Confirm reconciliation is performed within 3 business days').",
    ),
    "Evidence/Source": types.Schema(
        type=types.Type.STRING,
        description="Documents, reports, or artifacts that evidence this control's operation. Include IPE (Information Produced by Entity) references where mentioned (e.g., 'Approval email, Purchase Order, IPE: Vendor Master Report').",
    ),
    "Assertion Mapped": types.Schema(
        type=types.Type.STRING,
        description="SOX or process assertion(s). NEVER blank. For FR: Existence, Completeness, Valuation, Rights & Obligations, Accuracy, Cutoff, Presentation & Disclosure. For OR: Authorization, Validity, Accuracy, Completeness, Timeliness, Security, Compliance.",
    ),
    "Compliance Reference": types.Schema(
        type=types.Type.STRING,
        description="Must include specific section number. Format: '{Document ID}, Section {X.Y.Z}' (e.g., 'SOP-PROC-001, Section 5.2.1'). NEVER use just the document title. If spanning sections, list all.",
    ),
    "Risk Likelihood": types.Schema(
        type=types.Type.STRING,
        enum=["Low", "Medium", "High"],
    ),
    "Risk Impact": types.Schema(
        type=types.Type.STRING,
        enum=["Financial Misstatement", "Fraud/Error", "Compliance Violation", "Operational Disruption", "Reputational Damage", "Data Loss/Breach"],
        description="The primary impact if this risk materializes.",
    ),
    "Risk Rating": types.Schema(
        type=types.Type.STRING,
        enum=["Low", "Medium", "High", "Critical"],
    ),
    "Mitigation Effectiveness": types.Schema(
        type=types.Type.STRING,
        enum=["Effective", "Partially Effective", "Ineffective"],
    ),
    "Gaps/Weaknesses Identified": types.Schema(
        type=types.Type.STRING,
        description="Identified gaps or weaknesses. Write 'None' explicitly if none found. Examples: 'No segregation of duties', 'No independent review', 'Manual process prone to error'.",
    ),
    "Source Quote": types.Schema(
        type=types.Type.STRING,
        description="The EXACT verbatim text from the SOP that supports this entry. Must be a direct substring of the input. Keep under 200 characters.",
    ),
    "Extraction Confidence": types.Schema(
        type=types.Type.STRING,
        enum=["EXTRACTED", "INFERRED", "PARTIAL"],
        description="EXTRACTED = Risk and Control are verbatim from document. INFERRED = derived using professional judgment. PARTIAL = some fields extracted, others inferred.",
    ),
}

_RACM_ENTRY_REQUIRED = [
    "Process Area", "Risk Description", "Control Activity",
    "Control Owner", "Risk Type", "Control Frequency",
]

_RACM_PROPERTY_ORDERING = [
    "Process Area", "Sub-Process",
    "Risk ID", "Risk Description", "Risk Category", "Risk Type",
    "Control ID", "Control Activity", "Control Objective",
    "Control Type", "Control Nature", "Control Frequency", "Control Owner",
    "Control description as per SOP",
    "Testing Attributes", "Evidence/Source",
    "Assertion Mapped", "Compliance Reference",
    "Risk Likelihood", "Risk Impact", "Risk Rating",
    "Mitigation Effectiveness", "Gaps/Weaknesses Identified",
    "Source Quote", "Extraction Confidence",
]

RACM_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "detailed_entries": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties=_RACM_ENTRY_PROPERTIES,
                required=_RACM_ENTRY_REQUIRED,
                property_ordering=_RACM_PROPERTY_ORDERING,
            ),
        ),
        "summary_entries": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties=_RACM_ENTRY_PROPERTIES,
                required=_RACM_ENTRY_REQUIRED,
                property_ordering=_RACM_PROPERTY_ORDERING,
            ),
        ),
    },
    required=["detailed_entries"],
)


def _get_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def _parse_retry_delay(err_str: str) -> float | None:
    """Extract the retryDelay from Gemini's error response (e.g. 'retryDelay': '43s')."""
    match = re.search(r"retryDelay['\"]:\s*['\"](\d+\.?\d*)s?['\"]", err_str)
    if match:
        return float(match.group(1))
    # Also try "Please retry in Xs" pattern
    match = re.search(r"retry in (\d+\.?\d*)s", err_str)
    if match:
        return float(match.group(1))
    return None


async def _with_retry(coro_func, *args, max_retries: int = 5, base_delay: float = 5.0):
    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args)
        except Exception as e:
            if attempt == max_retries:
                raise
            err_str = str(e)
            retryable = (
                "429" in err_str or "rate" in err_str.lower() or "resource" in err_str.lower()
                or isinstance(e, json.JSONDecodeError)
            )
            if retryable:
                # Try to use the API's recommended retry delay
                api_delay = _parse_retry_delay(err_str)
                if api_delay and api_delay > 0:
                    delay = min(api_delay + 2, 120)  # respect API suggestion + 2s buffer, cap at 2min
                else:
                    delay = base_delay * (2 ** attempt)  # 5s, 10s, 20s, 40s, 80s
                logger.warning(
                    f"Retryable error (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}: {str(e)[:120]}. Retrying in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
            else:
                raise


def _extract_json(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else text


def _repair_json(raw: str) -> dict:
    """Attempt to parse JSON with progressive repair strategies."""
    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 3. Try truncating at last valid closing brace
    # Find the last complete entry by looking for the outermost closing brace
    brace_depth = 0
    last_valid_pos = -1
    for i, ch in enumerate(repaired):
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0:
                last_valid_pos = i
    if last_valid_pos > 0:
        truncated = repaired[:last_valid_pos + 1]
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

    # 4. Try closing unclosed arrays/objects
    bracket_count = raw.count('[') - raw.count(']')
    brace_count = raw.count('{') - raw.count('}')
    patched = re.sub(r",\s*$", "", raw.rstrip())
    patched += ']' * max(bracket_count, 0)
    patched += '}' * max(brace_count, 0)
    try:
        return json.loads(patched)
    except json.JSONDecodeError:
        pass

    # 5. Give up — raise with context
    raise json.JSONDecodeError(
        f"All JSON repair strategies failed (len={len(raw)})",
        raw[:200], 0
    )


async def analyze_chunk(chunk: str, user_instructions: str | None = None) -> dict:
    client = _get_client()
    prompt = (
        "AUDIT ASSIGNMENT:\n"
        "Synthesize a RACM from the following integrated evidence context.\n\n"
        "SCAN FOR THESE PATTERNS:\n"
        "- Approval/authorization flows → risk of unauthorized transactions if not performed\n"
        "- Reconciliation steps → risk of undetected errors or discrepancies\n"
        "- Segregation of duties → risk of fraud if same person performs conflicting roles\n"
        "- System access/configuration controls → risk of unauthorized access or changes\n"
        "- Review/verification steps → risk of errors passing undetected\n"
        "- Reporting deadlines/requirements → risk of late or inaccurate reporting\n"
        "- Record-keeping/documentation → risk of incomplete audit trail\n"
        "- Threshold/limit checks → risk of exceeding authorized limits\n\n"
        "RISK INDICATOR WORDS: shall, must, required, ensure, verify, approve, authorize, "
        "reconcile, review, validate, confirm, segregat, independent, escalat, monitor\n\n"
        "FOR EACH STEP: Ask 'What could go wrong if this is not done or done incorrectly?' "
        "— that is the risk. The step itself is the control.\n\n"
        "TABLE HANDLING: If the text contains tables, each row likely represents a separate "
        "control or process step. Extract each row as an independent entry.\n\n"
        "QUALITY REQUIREMENTS:\n"
        "- ALL 25 fields populated. No blank Assertion Mapped. No 'ERP System' as owner. No 'Recurring' as frequency.\n"
        "- Compliance Reference must include specific section numbers (e.g., 'SOP-PROC-001, Section 5.3').\n"
        "- Control Owner must be a human role, never a system name.\n\n"
    )
    if user_instructions:
        prompt += f"AUDITOR PREFERENCES: {user_instructions}\n\n"
    prompt += f"SOP CONTENT:\n{chunk}"

    chunk_size = len(chunk)
    logger.info(f"Gemini analyze_chunk call: input={chunk_size} chars, model={settings.gemini_model}")

    async def _call():
        t0 = time.time()
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=RACM_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=RACM_SCHEMA,
                temperature=0,
                top_p=1,
                seed=42,
            ),
        )
        api_time = time.time() - t0
        raw = _extract_json(response.text)
        result = _repair_json(raw)
        n_detailed = len(result.get("detailed_entries", []))
        n_summary = len(result.get("summary_entries", []))
        logger.info(f"Gemini response: {n_detailed} detailed + {n_summary} summary entries, {len(response.text)} chars, {api_time:.1f}s")
        return result

    return await _with_retry(_call)


async def vision_extract(image_bytes: bytes, mime_type: str) -> str:
    client = _get_client()

    async def _call():
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                (
                    "Extract ALL text from this SOP/process document image.\n\n"
                    "OUTPUT FORMAT:\n"
                    "- Preserve headings as markdown headers (# ##)\n"
                    "- Preserve tables as markdown tables (| col1 | col2 |)\n"
                    "- Preserve numbered/bulleted lists\n"
                    "- Identify roles mentioned (e.g., Manager, Officer, Analyst)\n"
                    "- Identify process steps and their sequence\n"
                    "- Include ALL fine print, footnotes, and references\n"
                    "- Do NOT summarize or paraphrase — extract verbatim"
                ),
            ],
            config=types.GenerateContentConfig(temperature=0),
        )
        return response.text

    return await _with_retry(_call)


def generate_racm_summary(detailed: list[dict], summary: list[dict], file_name: str) -> str:
    """Generate a human-readable summary of RACM results using pure Python aggregation.

    No Gemini call needed — all data is structured and can be counted/grouped directly.
    This saves ~15s per job compared to the previous Gemini-based approach.
    """
    from collections import Counter

    t0 = time.time()

    total_risks = len(detailed)
    control_ids = set(e.get("Control ID", "") for e in detailed if e.get("Control ID"))
    total_controls = len(control_ids) if control_ids else total_risks
    process_areas = sorted(set(e.get("Process Area", "") for e in detailed if e.get("Process Area")))

    # Breakdowns
    by_category = Counter(e.get("Risk Category", "Unspecified") or "Unspecified" for e in detailed)
    by_rating = Counter(e.get("Risk Rating", "Unspecified") or "Unspecified" for e in detailed)
    by_control_type = Counter(e.get("Control Type", "Unspecified") or "Unspecified" for e in detailed)
    by_confidence = Counter(e.get("Extraction Confidence", "INFERRED") or "INFERRED" for e in detailed)

    # Process area breakdown
    pa_stats = {}
    for e in detailed:
        pa = e.get("Process Area", "Other") or "Other"
        if pa not in pa_stats:
            pa_stats[pa] = {"risks": 0, "controls": set(), "top_rating": "Low"}
        pa_stats[pa]["risks"] += 1
        cid = e.get("Control ID", "")
        if cid:
            pa_stats[pa]["controls"].add(cid)
        rating = e.get("Risk Rating", "")
        rating_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        if rating_order.get(rating, 0) > rating_order.get(pa_stats[pa]["top_rating"], 0):
            pa_stats[pa]["top_rating"] = rating

    # Gaps/weaknesses
    gaps = []
    for e in detailed:
        g = (e.get("Gaps/Weaknesses Identified") or "").strip()
        if g and g.lower() != "none":
            gaps.append(f"- **{e.get('Risk ID', '?')}** ({e.get('Process Area', '?')}): {g}")

    # Confidence percentages
    conf_total = sum(by_confidence.values()) or 1
    conf_pcts = {k: round(v / conf_total * 100) for k, v in by_confidence.items()}

    # Build markdown
    md = []
    md.append(f"## RACM Analysis: {file_name}\n")

    # Overview
    md.append("### Overview\n")
    md.append(f"| Metric | Count |")
    md.append(f"|---|---|")
    md.append(f"| Total Risks | {total_risks} |")
    md.append(f"| Total Controls | {total_controls} |")
    md.append(f"| Process Areas | {len(process_areas)} |")
    md.append("")

    # Risk Rating Distribution
    md.append("### Risk Rating Distribution\n")
    for rating in ["Critical", "High", "Medium", "Low"]:
        count = by_rating.get(rating, 0)
        if count > 0:
            bar = "\u2588" * min(count * 3, 30)
            md.append(f"- **{rating}**: {bar} {count}")
    unspec = by_rating.get("Unspecified", 0)
    if unspec > 0:
        md.append(f"- **Unspecified**: {unspec}")
    md.append("")

    # Risk Category Breakdown
    md.append("### By Risk Category\n")
    md.append("| Category | Count |")
    md.append("|---|---|")
    for cat in sorted(by_category.keys()):
        if cat == "Unspecified":
            continue
        count = by_category.get(cat, 0)
        if count > 0:
            md.append(f"| {cat} | {count} |")
    unspec = by_category.get("Unspecified", 0)
    if unspec > 0:
        md.append(f"| Unspecified | {unspec} |")
    md.append("")

    # Control Type Breakdown
    md.append("### By Control Type\n")
    md.append("| Type | Count |")
    md.append("|---|---|")
    for ct in ["Preventive", "Detective", "Corrective"]:
        count = by_control_type.get(ct, 0)
        if count > 0:
            md.append(f"| {ct} | {count} |")
    md.append("")

    # By Process Area
    if pa_stats:
        md.append("### By Process Area\n")
        md.append("| Process Area | Risks | Controls | Top Risk |")
        md.append("|---|---|---|---|")
        for pa in sorted(pa_stats.keys()):
            s = pa_stats[pa]
            md.append(f"| {pa} | {s['risks']} | {len(s['controls']) or s['risks']} | {s['top_rating']} |")
        md.append("")

    # Gaps & Weaknesses
    md.append("### Key Findings & Gaps\n")
    if gaps:
        md.append(f"**{len(gaps)} gap(s)/weakness(es) identified:**\n")
        for g in gaps[:15]:  # cap at 15 to keep summary concise
            md.append(g)
        if len(gaps) > 15:
            md.append(f"\n*...and {len(gaps) - 15} more (see detailed table)*")
    else:
        md.append("No significant gaps or weaknesses identified across all entries.")
    md.append("")

    # Extraction Confidence
    md.append("### Extraction Confidence\n")
    for conf in ["EXTRACTED", "PARTIAL", "INFERRED"]:
        pct = conf_pcts.get(conf, 0)
        count = by_confidence.get(conf, 0)
        if count > 0:
            md.append(f"- **{conf}**: {pct}% ({count} entries)")
    md.append("")

    result = "\n".join(md)
    elapsed = time.time() - t0
    logger.info(f"Summary generation (Python): {len(result)} chars in {elapsed * 1000:.0f}ms")
    return result


async def consolidation_pass(
    detailed: list[dict], summary: list[dict], user_instructions: str | None = None
) -> dict:
    if len(detailed) <= 20:
        logger.info(f"Consolidation skipped: only {len(detailed)} entries (threshold: 20)")
        return {"detailed_entries": detailed, "summary_entries": summary}

    condensed = json.dumps(
        {"detailed_entries": detailed[:200], "summary_entries": summary[:50]},
        ensure_ascii=False,
    )

    # Skip consolidation if payload too large for single call
    if len(condensed) > 800_000:
        logger.warning(f"Consolidation skipped: payload too large ({len(condensed)} chars > 800K limit)")
        return {"detailed_entries": detailed, "summary_entries": summary}

    logger.info(f"Gemini consolidation call: {len(detailed)} detailed + {len(summary)} summary, payload={len(condensed)} chars")

    prompt = (
        "CONSOLIDATION TASK:\n"
        "You analyzed a document in segments. Below are the raw RACM entries from all chunks.\n"
        "Your job:\n"
        "1. MERGE entries ONLY if same Risk Description AND same Control Owner AND same Control Activity. "
        "NEVER merge entries with different Control Owners.\n"
        "2. FILL missing fields where context from other entries provides the answer.\n"
        "3. RE-SEQUENCE Risk IDs (R001, R002...) and Control IDs (C001, C002...). "
        "Two risks sharing the same control (same owner+activity) → different Risk IDs, SAME Control ID.\n"
        "4. PRODUCE summary_entries grouped by Process Area. Include all 25 fields.\n"
        "5. REMOVE only TRUE duplicates (identical risk + control + owner). When in doubt, keep both entries.\n\n"
        "QUALITY ENFORCEMENT:\n"
        "6. Control Frequency: Replace any 'Recurring', 'Event based', or 'One-Time Activity' with specific cadence from context.\n"
        "7. Control Owner: Replace any 'ERP System', 'System', 'Automated', or 'Banking system' with the appropriate human role.\n"
        "8. Assertion Mapped: Populate ALL entries. Financial assertions for FR risks, process assertions (Authorization, Validity, etc.) for OR risks. NEVER leave blank.\n"
        "9. Compliance Reference: Must have section numbers (e.g., 'SOP-PROC-001, Section 5.3'), not just the document title.\n"
        "10. Gaps/Weaknesses: 'None' explicitly if none found, never blank.\n"
        "11. Extraction Confidence: EXTRACTED entries must have a Source Quote. INFERRED entries may have empty Source Quote.\n"
        "12. Control Type vs Nature: No 'Manual' in Control Type. No 'Preventive' in Control Nature. Type=WHAT, Nature=HOW.\n\n"
    )
    if user_instructions:
        prompt += f"AUDITOR PREFERENCES: {user_instructions}\n\n"
    prompt += f"RAW ENTRIES:\n{condensed}"

    client = _get_client()

    async def _call():
        t0 = time.time()
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=RACM_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=RACM_SCHEMA,
                temperature=0,
                top_p=1,
                seed=42,
            ),
        )
        api_time = time.time() - t0
        raw = _extract_json(response.text)
        result = _repair_json(raw)
        n_d = len(result.get("detailed_entries", []))
        n_s = len(result.get("summary_entries", []))
        logger.info(f"Consolidation response: {n_d} detailed + {n_s} summary, {len(response.text)} chars, {api_time:.1f}s")
        return result

    return await _with_retry(_call)
