import asyncio
import json
import logging
import re

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

RACM_SYSTEM_INSTRUCTION = """You are an elite SOX Compliance Auditor and Risk Management Professional.
Your mission is to perform an EXHAUSTIVE, FAITHFUL extraction of the Risk and Control Matrix from the provided SOP/audit documents.

═══════════════════════════════════════════
DIRECTIVE 1 — COMPLETENESS (MOST CRITICAL)
═══════════════════════════════════════════
- Extract EVERY risk-control pair defined in the document. Missing even one is a SOX audit failure.
- NEVER merge two entries that have different Control Owners, even if the Risk Description is identical.
  Example: "PPA not considered" reviewed by AM–Bid & Auctions AND by Approving Authority = TWO separate entries.
  In SOX, WHO performs the control is as important as WHAT the control does.
- If the SOP maps multiple risks to the SAME control (same control description, same owner), output each risk
  as a separate entry but reference the same Control ID and Control Activity text.

═══════════════════════════════════════════
DIRECTIVE 2 — EXTRACT, DO NOT FABRICATE
═══════════════════════════════════════════
For each field, you must either EXTRACT it from the document or INFER it with clear reasoning:

MUST EXTRACT VERBATIM FROM DOCUMENT (do not rephrase):
- Risk Description: Use the exact risk text from the SOP
- Control Activity: Use the exact control description from the SOP
- Control description as per SOP: Copy the control text as-is from the SOP
- Risk Type: Use SOP notation — "FR" (Financial Reporting) or "OR" (Operating Risk). If the SOP says "Financial Reporting Risk" use "Financial Reporting Risk". Match the source.
- Control Frequency: Extract EXACTLY as stated — "Recurring", "One-Time Activity", "Daily", "Event based", etc. Do NOT default everything to "Recurring". One-Time Activity controls (system configs, access rights) have different testing requirements.
- Assertions (Assertion Mapped): Extract exactly from the SOP. Typical SOX assertions: Existence, Rights & Obligations, Valuation, Occurrence, Presentation & Disclosure. For FR risks, include all assertions marked. For OR (Operating Risk) entries, leave Assertion Mapped BLANK — operating risks do not map to financial statement assertions.
- Control Owner: Extract the specific role/person from the SOP (e.g., "AM – Bid & Auctions", "Approving Authority", "TPA Manager", "TPA Tax Manager", "TPA Executive"). These are distinct roles — never conflate them.
- Evidence/Source: Reference the actual document name and any IPE (Information Produced by Entity) references like IPE-9, IPE-10, IPE-11.
- Compliance Reference: Use the SOP document title/name.

MAY BE INFERRED (use professional judgment):
- Control Type: Preventive (before event) / Detective (after event) / Corrective. Pre-approval reviews = Preventive. Post-event sampling/random checks = Detective. System access restrictions = Preventive.
- Control Nature: Manual / Automated / IT-Dependent Manual. System-enforced controls (ERP access, duplicate warnings) = Automated. Human reviews = Manual.
- Risk Category: Financial Reporting / Operational / Compliance / Strategic
- Control Objective: Derive from the control's purpose in context
- Testing Attributes: Describe what an auditor would inspect/test
- Risk Likelihood: Infer as Low / Medium / High based on control strength and process context. Vary these — not everything is "Medium".
- Risk Impact: Use DESCRIPTIVE terms: "Financial Misstatement", "Fraud/Error", "Compliance Violation", "Process Inefficiency", "Reporting Delay", "Delivery Failure", etc.
- Risk Rating: Derive from Likelihood x Impact matrix. Low/Medium/High/Critical. Vary appropriately.
- Mitigation Effectiveness: Effective / Partially Effective / Ineffective
- Gaps/Weaknesses Identified: If none found, write "None" explicitly. Do NOT leave blank.

═══════════════════════════════════════════
DIRECTIVE 2B — NULL / MISSING DATA RULES
═══════════════════════════════════════════
- If a field is NOT explicitly stated or clearly inferable from the SOP, return an empty string "".
- Do NOT fabricate Control Owner names — if the SOP does not name a specific role, use "".
- Do NOT guess Risk Likelihood or Risk Rating — if insufficient context, use "".
- It is ALWAYS better to return "" than to return an incorrect or fabricated value.
- For Assertion Mapped: MUST be "" for Operating Risk (OR) entries — no exceptions.
- For Gaps/Weaknesses Identified: write "None" explicitly if no gaps are found. Never leave blank.
- For Source Quote: if you cannot find a verbatim passage supporting the entry, use "" and set Extraction Confidence to "INFERRED".

═══════════════════════════════════════════
DIRECTIVE 3 — STRUCTURAL RULES
═══════════════════════════════════════════
- Process Area and Sub-Process: Extract from the document header/title (e.g., "Accounts Receivable", "Customer Billing")
- Risk IDs: Sequential (R001, R002, ...). Control IDs: Sequential (C001, C002, ...)
- If two risks share the same control, they get different Risk IDs but the SAME Control ID
- Include process step references (CB1, CB2, etc.) in the Control Activity or Evidence field when available
- Include all IPE references (IPE-9, IPE-10, IPE-11, etc.) — these define what evidence auditors must examine

═══════════════════════════════════════════
DIRECTIVE 4 — OUTPUT STRUCTURE
═══════════════════════════════════════════
- detailed_entries: One record per distinct risk-control pair. Every field populated.
- summary_entries: Grouped by Process Area. Include all 23 fields with executive-level detail.

CONSISTENCY: Maintain deterministic output. Same input must produce the same RACM.

═══════════════════════════════════════════
DIRECTIVE 5 — EXAMPLES
═══════════════════════════════════════════

EXAMPLE 1 — Financial Reporting Risk (Happy Path):
{
  "Process Area": "Accounts Receivable",
  "Sub-Process": "Customer Billing",
  "Risk ID": "R001",
  "Risk Description": "Invoices may be raised with incorrect rates or quantities, leading to revenue misstatement",
  "Risk Category": "Financial Reporting",
  "Risk Type": "FR",
  "Control ID": "C001",
  "Control Activity": "AM – Billing reviews and approves all invoices above $10,000 against signed contract rates before dispatch",
  "Control Objective": "Ensure invoice accuracy and completeness before customer dispatch",
  "Control Type": "Preventive",
  "Control Nature": "Manual",
  "Control Frequency": "Event based",
  "Control Owner": "AM – Billing",
  "Control description as per SOP": "AM – Billing reviews and approves all invoices above $10,000 against signed contract rates before dispatch",
  "Testing Attributes": "Inspect sample of approved invoices; verify rates match contract; confirm AM signature present",
  "Evidence/Source": "SOP-AR-001, IPE-12 (Invoice Register)",
  "Assertion Mapped": "Existence, Valuation",
  "Compliance Reference": "SOP-AR-001 Accounts Receivable Process",
  "Risk Likelihood": "Medium",
  "Risk Impact": "Financial Misstatement",
  "Risk Rating": "Medium",
  "Mitigation Effectiveness": "Effective",
  "Gaps/Weaknesses Identified": "None",
  "Source Quote": "AM – Billing reviews and approves all invoices above $10,000 against signed contract rates before dispatch",
  "Extraction Confidence": "EXTRACTED"
}

EXAMPLE 2 — Operating Risk with Null Handling (Edge Case):
{
  "Process Area": "Procurement",
  "Sub-Process": "Vendor Onboarding",
  "Risk ID": "R002",
  "Risk Description": "Duplicate vendor records may be created in the system, leading to duplicate payments",
  "Risk Category": "Operational",
  "Risk Type": "OR",
  "Control ID": "C002",
  "Control Activity": "ERP system performs automated duplicate check on vendor tax ID and bank details before saving new vendor record",
  "Control Objective": "Prevent creation of duplicate vendor master records",
  "Control Type": "Preventive",
  "Control Nature": "Automated",
  "Control Frequency": "Recurring",
  "Control Owner": "ERP System",
  "Control description as per SOP": "ERP system performs automated duplicate check on vendor tax ID and bank details before saving new vendor record",
  "Testing Attributes": "Attempt to create duplicate vendor in test environment; verify system rejection; review duplicate check configuration",
  "Evidence/Source": "SOP-PROC-003, IPE-5 (Vendor Master Log)",
  "Assertion Mapped": "",
  "Compliance Reference": "SOP-PROC-003 Vendor Management Process",
  "Risk Likelihood": "Low",
  "Risk Impact": "Process Inefficiency",
  "Risk Rating": "Low",
  "Mitigation Effectiveness": "Effective",
  "Gaps/Weaknesses Identified": "None",
  "Source Quote": "ERP system performs automated duplicate check on vendor tax ID and bank details before saving new vendor record",
  "Extraction Confidence": "EXTRACTED"
}

Note on Example 2:
- Risk Type is "OR" (Operating Risk), so Assertion Mapped is "" (blank) — operating risks do not map to financial statement assertions.
- Control Nature is "Automated" because the ERP system enforces this control.
- Extraction Confidence is "EXTRACTED" because the control text is verbatim from the SOP.
- If the SOP did not name a specific owner for this control, Control Owner would be "" instead of a fabricated name.
"""

_RACM_ENTRY_PROPERTIES = {
    "Process Area": types.Schema(type=types.Type.STRING),
    "Sub-Process": types.Schema(type=types.Type.STRING),
    "Risk ID": types.Schema(type=types.Type.STRING),
    "Risk Description": types.Schema(
        type=types.Type.STRING,
        description="The exact risk text as stated in the SOP. Do not rephrase.",
    ),
    "Risk Category": types.Schema(
        type=types.Type.STRING,
        enum=["Financial Reporting", "Operational", "Compliance", "Strategic"],
    ),
    "Risk Type": types.Schema(type=types.Type.STRING),
    "Control ID": types.Schema(type=types.Type.STRING),
    "Control Activity": types.Schema(
        type=types.Type.STRING,
        description="The exact control description as stated in the SOP. Do not rephrase.",
    ),
    "Control Objective": types.Schema(type=types.Type.STRING),
    "Control Type": types.Schema(
        type=types.Type.STRING,
        enum=["Preventive", "Detective", "Corrective"],
    ),
    "Control Nature": types.Schema(
        type=types.Type.STRING,
        enum=["Manual", "Automated", "IT-Dependent Manual"],
    ),
    "Control Frequency": types.Schema(type=types.Type.STRING),
    "Control Owner": types.Schema(
        type=types.Type.STRING,
        description="The specific role/person from the SOP who performs this control.",
    ),
    "Control description as per SOP": types.Schema(type=types.Type.STRING),
    "Testing Attributes": types.Schema(type=types.Type.STRING),
    "Evidence/Source": types.Schema(type=types.Type.STRING),
    "Assertion Mapped": types.Schema(
        type=types.Type.STRING,
        description="SOX assertion(s). Leave blank for Operating Risk (OR) entries.",
    ),
    "Compliance Reference": types.Schema(type=types.Type.STRING),
    "Risk Likelihood": types.Schema(
        type=types.Type.STRING,
        enum=["Low", "Medium", "High"],
    ),
    "Risk Impact": types.Schema(type=types.Type.STRING),
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
        description="Identified gaps or weaknesses. Write 'None' explicitly if none found.",
    ),
    "Source Quote": types.Schema(
        type=types.Type.STRING,
        description="The EXACT verbatim text from the SOP document that supports this entry's Risk Description and Control Activity. Must be a direct substring of the input.",
    ),
    "Extraction Confidence": types.Schema(
        type=types.Type.STRING,
        enum=["EXTRACTED", "INFERRED", "PARTIAL"],
        description="EXTRACTED if Risk Description and Control Activity are verbatim from the document. INFERRED if fields were derived using professional judgment. PARTIAL if some fields are extracted and others inferred.",
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


async def _with_retry(coro_func, *args, max_retries: int = 3, base_delay: float = 2.0):
    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args)
        except Exception as e:
            if attempt == max_retries:
                raise
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str or "resource" in err_str:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
            else:
                raise


def _extract_json(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else text


async def analyze_chunk(chunk: str, user_instructions: str | None = None) -> dict:
    client = _get_client()
    prompt = "AUDIT ASSIGNMENT:\nSynthesize a RACM from the following integrated evidence context.\n\n"
    if user_instructions:
        prompt += f"AUDITOR PREFERENCES: {user_instructions}\n\n"
    prompt += f"INTEGRATED CONTEXT:\n{chunk}"

    async def _call():
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
        raw = _extract_json(response.text)
        return json.loads(raw)

    return await _with_retry(_call)


async def vision_extract(image_bytes: bytes, mime_type: str) -> str:
    client = _get_client()

    async def _call():
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                (
                    "Extract ALL text from this image. Preserve the structure, tables, "
                    "headings, and any data relationships visible. If this is a process "
                    "document, identify steps, roles, and controls. Return the content "
                    "as structured text."
                ),
            ],
            config=types.GenerateContentConfig(temperature=0),
        )
        return response.text

    return await _with_retry(_call)


async def consolidation_pass(
    detailed: list[dict], summary: list[dict], user_instructions: str | None = None
) -> dict:
    if len(detailed) <= 10:
        return {"detailed_entries": detailed, "summary_entries": summary}

    condensed = json.dumps(
        {"detailed_entries": detailed[:200], "summary_entries": summary[:50]},
        ensure_ascii=False,
    )

    # Skip consolidation if payload too large for single call
    if len(condensed) > 800_000:
        return {"detailed_entries": detailed, "summary_entries": summary}

    prompt = (
        "CONSOLIDATION TASK:\n"
        "You previously analyzed a document in segments. Below are the raw extracted RACM entries.\n"
        "Your job is to:\n"
        "1. Merge entries ONLY if they have the same Risk Description AND the same Control Owner AND the same Control Activity.\n"
        "   NEVER merge entries that have different Control Owners — even if the risk description is identical.\n"
        "2. Fill in any missing fields where context from other entries provides the answer.\n"
        "3. Ensure Risk IDs are sequential (R001, R002, ...) and Control IDs match (C001, C002, ...).\n"
        "   If two risks share the same control (same owner, same activity), assign them different Risk IDs but the SAME Control ID.\n"
        "4. Produce a coherent summary_entries set that groups findings by Process Area. Include all 23 fields.\n"
        "5. Remove only TRUE duplicates (identical risk + control + owner).\n"
        "6. Ensure Control Frequency matches the source: 'Recurring' vs 'One-Time Activity' — do NOT default all to Recurring.\n"
        "7. Ensure Assertion Mapped is blank for OR (Operating Risk) entries and populated for FR entries.\n"
        "8. Ensure Risk Impact uses descriptive terms (Financial Misstatement, Fraud/Error, etc.) not just Low/Medium/High.\n"
        "9. Ensure Gaps/Weaknesses says 'None' explicitly if no gaps found, not left blank.\n"
        "10. Ensure Compliance Reference contains the source document name.\n\n"
    )
    if user_instructions:
        prompt += f"AUDITOR PREFERENCES: {user_instructions}\n\n"
    prompt += f"RAW ENTRIES:\n{condensed}"

    client = _get_client()

    async def _call():
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
        raw = _extract_json(response.text)
        return json.loads(raw)

    return await _with_retry(_call)
