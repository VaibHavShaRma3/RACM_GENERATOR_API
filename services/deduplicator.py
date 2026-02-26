import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def _exact_dedup(entries: list[dict]) -> list[dict]:
    """Remove entries with identical risk + control + owner combination."""
    seen: dict[str, dict] = {}
    for entry in entries:
        risk = (entry.get("Risk Description") or "").strip().lower()
        ctrl = (entry.get("Control Activity") or "").strip().lower()
        owner = (entry.get("Control Owner") or "").strip().lower()
        key = f"{risk}|{ctrl}|{owner}"
        if key not in seen:
            seen[key] = entry
    return list(seen.values())


def _count_populated(entry: dict) -> int:
    return sum(1 for v in entry.values() if v and str(v).strip())


def _fuzzy_dedup(entries: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove near-duplicate entries using sequence matching.

    Requires ALL THREE of risk description, control activity, AND control owner
    to be similar before merging. This prevents merging controls that have the
    same risk but different performers (e.g., AM vs Approving Authority).
    """
    if len(entries) <= 1:
        return entries

    keep: list[dict] = []
    for entry in entries:
        is_dup = False
        entry_risk = (entry.get("Risk Description") or "").lower()
        entry_ctrl = (entry.get("Control Activity") or "").lower()
        entry_owner = (entry.get("Control Owner") or "").lower()

        for i, kept in enumerate(keep):
            kept_risk = (kept.get("Risk Description") or "").lower()
            kept_ctrl = (kept.get("Control Activity") or "").lower()
            kept_owner = (kept.get("Control Owner") or "").lower()

            risk_sim = SequenceMatcher(None, entry_risk, kept_risk).ratio()
            ctrl_sim = SequenceMatcher(None, entry_ctrl, kept_ctrl).ratio()
            owner_sim = SequenceMatcher(None, entry_owner, kept_owner).ratio()

            # All three must be similar — different owner = different control
            if risk_sim > threshold and ctrl_sim > threshold and owner_sim > threshold:
                if _count_populated(entry) > _count_populated(kept):
                    keep[i] = entry
                is_dup = True
                break

        if not is_dup:
            keep.append(entry)

    return keep


def _reindex(entries: list[dict], risk_prefix: str = "R", ctrl_prefix: str = "C") -> list[dict]:
    """Reassign sequential Risk IDs and Control IDs.

    Preserves shared Control IDs: entries with the same Control Activity + Control Owner
    get the same Control ID (different risks can share a control).
    """
    result = []
    control_map: dict[str, str] = {}  # (activity_lower, owner_lower) → Control ID
    ctrl_counter = 0

    for idx, entry in enumerate(entries, 1):
        entry = dict(entry)  # copy
        entry["Risk ID"] = f"{risk_prefix}{str(idx).zfill(3)}"

        # Group by (Control Activity, Control Owner) to preserve shared controls
        activity = (entry.get("Control Activity") or "").strip().lower()
        owner = (entry.get("Control Owner") or "").strip().lower()
        ctrl_key = f"{activity}|{owner}"

        if ctrl_key not in control_map:
            ctrl_counter += 1
            control_map[ctrl_key] = f"{ctrl_prefix}{str(ctrl_counter).zfill(3)}"

        entry["Control ID"] = control_map[ctrl_key]
        result.append(entry)
    return result


def deduplicate_racm(racm: dict) -> dict:
    detailed = racm.get("detailed_entries", [])
    summary = racm.get("summary_entries", [])

    initial_detailed = len(detailed)
    initial_summary = len(summary)

    detailed = _exact_dedup(detailed)
    exact_removed = initial_detailed - len(detailed)
    logger.info(f"Exact dedup (detailed): {initial_detailed} → {len(detailed)} (removed {exact_removed})")

    pre_fuzzy = len(detailed)
    detailed = _fuzzy_dedup(detailed)
    fuzzy_removed = pre_fuzzy - len(detailed)
    logger.info(f"Fuzzy dedup (detailed): {pre_fuzzy} → {len(detailed)} (removed {fuzzy_removed})")

    detailed = _reindex(detailed, "R", "C")

    summary = _exact_dedup(summary)
    summary_removed = initial_summary - len(summary)
    logger.info(f"Exact dedup (summary): {initial_summary} → {len(summary)} (removed {summary_removed})")
    summary = _reindex(summary, "SR", "SC")

    logger.info(
        f"Dedup totals: detailed {initial_detailed} → {len(detailed)} "
        f"(exact -{exact_removed}, fuzzy -{fuzzy_removed}), "
        f"summary {initial_summary} → {len(summary)} (-{summary_removed})"
    )

    return {"detailed_entries": detailed, "summary_entries": summary}
