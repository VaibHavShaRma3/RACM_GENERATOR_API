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
EXTRACT VERBATIM: Risk Description, Control Activity, Control description as per SOP, Risk Type (FR/OR as in SOP), Control Frequency (exactly: Recurring/One-Time Activity/Daily/Event based — do NOT default to Recurring), Assertion Mapped (from SOP; BLANK for OR entries), Control Owner (exact role), Evidence/Source (include IPE refs), Compliance Reference.
MAY INFER: Control Type (Preventive/Detective/Corrective), Control Nature (Manual/Automated/IT-Dependent Manual), Risk Category, Control Objective, Testing Attributes, Risk Likelihood (vary — not all Medium), Risk Impact (descriptive: Financial Misstatement/Fraud/Compliance Violation/etc), Risk Rating (from Likelihood×Impact), Mitigation Effectiveness, Gaps/Weaknesses (write "None" explicitly if none found).

DIRECTIVE 3 — NULL RULES
- Unknown/not inferable → return "". Never fabricate.
- Assertion Mapped MUST be "" for OR entries.
- Gaps/Weaknesses: "None" if no gaps, never blank.
- Source Quote: verbatim substring or "" with Extraction Confidence="INFERRED".

DIRECTIVE 4 — STRUCTURE
- Process Area/Sub-Process from document header. Risk IDs: R001, R002... Control IDs: C001, C002...
- Shared controls → different Risk IDs, same Control ID. Include IPE references.
- detailed_entries: one per risk-control pair. summary_entries: grouped by Process Area.

KEY PATTERNS:
- FR entry: Risk Type="FR", Assertion Mapped="Existence, Valuation", Control Frequency="Event based"
- OR entry: Risk Type="OR", Assertion Mapped="" (blank!), Control Nature="Automated" if system-enforced
- Pre-approval review = Preventive/Manual. Post-event sampling = Detective. System config = Preventive/Automated.
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
    prompt = "AUDIT ASSIGNMENT:\nSynthesize a RACM from the following integrated evidence context.\n\n"
    if user_instructions:
        prompt += f"AUDITOR PREFERENCES: {user_instructions}\n\n"
    prompt += f"INTEGRATED CONTEXT:\n{chunk}"

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


async def generate_racm_summary(detailed: list[dict], summary: list[dict], file_name: str) -> str:
    """Generate a human-readable narrative summary of the RACM results."""
    # Build a lightweight payload — just field values, no source quotes
    condensed_entries = []
    for entry in detailed[:100]:
        condensed_entries.append({
            k: v for k, v in entry.items()
            if k not in ("Source Quote", "Control description as per SOP", "Testing Attributes")
        })

    payload = json.dumps(condensed_entries, ensure_ascii=False)

    prompt = (
        f"RACM SUMMARY TASK:\n"
        f"Document: {file_name}\n"
        f"Total detailed entries: {len(detailed)}\n"
        f"Total summary entries: {len(summary)}\n\n"
        f"Based on the RACM entries below, produce a structured executive summary (200-400 words) covering:\n"
        f"1. Document name and scope of the analysis\n"
        f"2. Total risks and controls identified\n"
        f"3. Breakdown by Risk Category (Financial Reporting, Operational, Compliance, Strategic)\n"
        f"4. Breakdown by Risk Rating (Critical, High, Medium, Low)\n"
        f"5. Breakdown by Control Type (Preventive, Detective, Corrective)\n"
        f"6. Key findings and notable gaps or weaknesses\n"
        f"7. Extraction confidence overview (percentage of EXTRACTED vs INFERRED vs PARTIAL)\n\n"
        f"Format as clean markdown with headers. Be concise and factual.\n\n"
        f"RACM ENTRIES:\n{payload}"
    )

    client = _get_client()

    async def _call():
        t0 = time.time()
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                top_p=1,
            ),
        )
        api_time = time.time() - t0
        logger.info(f"Summary generation: {len(response.text)} chars in {api_time:.1f}s")
        return response.text

    return await _with_retry(_call)


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
