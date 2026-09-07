"""In-memory job registry.

This is the ONLY place job state lives in this phase -- no database, no
Redis, no disk-backed index. The surface is deliberately narrow
(create/get/mark_*/list) so that a later phase can swap the storage backend
(e.g. Redis or a database, for a multi-process/EC2 deployment) without
touching the routers or Pydantic schemas that consume `JobRecord` objects:
those layers never see how a record is stored, only its fields.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .enums import JobStatus, JobType


@dataclass
class JobRecord:
    job_id: str
    job_type: JobType
    status: JobStatus
    created_at: float
    source_description: str = ""
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    message: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    # Internal working file for this job (e.g. an uploaded video). Never
    # serialized into an API response -- it's a filesystem detail, not part
    # of the public job contract.
    temp_path: Optional[Path] = None


class JobStore:
    """Thread-safe in-memory job registry with TTL-based eviction.

    Locking: a single `threading.Lock` guards all mutation. Job volume/
    concurrency in this phase is low (interactive uploads, not a high-rate
    stream), so a single coarse lock is simpler and safer than fine-grained
    per-job locks, and every method holds it only briefly.
    """

    def __init__(self, ttl_seconds: float = 3600.0):
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds

    def create(
        self,
        job_type: JobType,
        source_description: str = "",
        temp_path: Optional[Path] = None,
        message: Optional[str] = None,
    ) -> JobRecord:
        with self._lock:
            self._purge_expired_locked()
            record = JobRecord(
                job_id=uuid.uuid4().hex,
                job_type=job_type,
                status=JobStatus.PENDING,
                created_at=time.time(),
                source_description=source_description,
                temp_path=temp_path,
                message=message,
            )
            self._jobs[record.job_id] = record
            return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            self._purge_expired_locked()
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.RUNNING
            record.started_at = time.time()

    def update_progress(self, job_id: str, progress: float, message: Optional[str] = None) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.progress = max(0.0, min(1.0, progress))
            if message is not None:
                record.message = message

    def mark_completed(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.COMPLETED
            record.completed_at = time.time()
            record.progress = 1.0
            record.result = result

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.FAILED
            record.completed_at = time.time()
            record.error = error

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            self._purge_expired_locked()
            return list(self._jobs.values())

    def _purge_expired_locked(self) -> None:
        """Evict jobs (and delete their temp working file, if any) older
        than the TTL, regardless of status. This matters even for jobs that
        never leave PENDING -- e.g. an accepted-but-not-yet-processed video
        job in this phase -- so an uploaded file can't sit on disk forever
        just because nothing ever moved its job to a terminal state."""
        if self._ttl_seconds <= 0:
            return
        now = time.time()
        expired_ids = [jid for jid, rec in self._jobs.items() if (now - rec.created_at) > self._ttl_seconds]
        for jid in expired_ids:
            record = self._jobs.pop(jid)
            if record.temp_path is not None:
                try:
                    record.temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
