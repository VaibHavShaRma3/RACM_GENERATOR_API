from difflib import SequenceMatcher


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
    """Reassign sequential Risk IDs and Control IDs."""
    result = []
    for idx, entry in enumerate(entries, 1):
        entry = dict(entry)  # copy
        entry["Risk ID"] = f"{risk_prefix}{str(idx).zfill(3)}"
        entry["Control ID"] = f"{ctrl_prefix}{str(idx).zfill(3)}"
        result.append(entry)
    return result


def deduplicate_racm(racm: dict) -> dict:
    detailed = racm.get("detailed_entries", [])
    summary = racm.get("summary_entries", [])

    detailed = _exact_dedup(detailed)
    detailed = _fuzzy_dedup(detailed)
    detailed = _reindex(detailed, "R", "C")

    summary = _exact_dedup(summary)
    summary = _reindex(summary, "SR", "SC")

    return {"detailed_entries": detailed, "summary_entries": summary}
