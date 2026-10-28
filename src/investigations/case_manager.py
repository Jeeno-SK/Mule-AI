"""Alert status workflow, analyst notes and audit logging. Terminology: alerts are unverified indicators."""
from __future__ import annotations

from src.database.repository import Repository
from src.investigations.audit_log import log_action
from src.utils.exceptions import CaseError

STATUSES = ("open", "under_review", "resolved", "escalated")
MAX_NOTE = 2000
MAX_REASON = 1000


def set_alert_status(repo: Repository, dataset_id: int, alert_ids: list[str], status: str, reason: str | None, actor: str) -> int:
    """Set the status of one or more alerts. 'resolved' requires a resolution reason."""
    if status not in STATUSES:
        raise CaseError(f"Invalid status '{status}'.")
    reason = (reason or "").strip()
    if status == "resolved" and not reason:
        raise CaseError("A resolution reason is required to mark an alert as resolved.")
    if len(reason) > MAX_REASON:
        raise CaseError(f"Resolution reason is limited to {MAX_REASON} characters.")
    ids = list(dict.fromkeys(alert_ids))
    if not ids:
        raise CaseError("No alerts selected.")
    repo.upsert_alert_status(dataset_id, ids, status, reason or None, actor)
    log_action(repo, actor, "alert_status_changed", dataset_id, ids[0] if len(ids) == 1 else f"{len(ids)} alerts",
               {"status": status, "reason": reason, "alert_ids": ids[:50]})
    return len(ids)


def set_status_for_account(repo: Repository, dataset_id: int, account: str, status: str, reason: str | None, actor: str) -> int:
    ids = repo.alert_ids_for_account(dataset_id, account)
    if not ids:
        raise CaseError("This account has no alerts.")
    return set_alert_status(repo, dataset_id, ids, status, reason, actor)


def add_note(repo: Repository, dataset_id: int, account: str, note: str, actor: str, alert_id: str | None = None) -> None:
    note = (note or "").strip()
    if not note:
        raise CaseError("A note cannot be empty.")
    if len(note) > MAX_NOTE:
        raise CaseError(f"Notes are limited to {MAX_NOTE} characters.")
    repo.insert_note(dataset_id, account, alert_id, note, actor)
    log_action(repo, actor, "note_added", dataset_id, account, {"alert_id": alert_id, "length": len(note)})
