"""APScheduler with SQLite job store — persistent daily and weekly routines.

Replaces session-only CRON (from Alpaca Integration) with durable scheduling.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from config import DB_DIR, DB_PATH

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("US/Eastern")
UTC = timezone.utc


def get_ny_time() -> datetime:
    """Current NY time."""
    return datetime.now(EASTERN)


def get_ny_time_utc(hour: int, minute: int = 0) -> datetime:
    """Return next occurrence of a given NY time, converted to UTC for APScheduler.

    If the time has passed today, returns tomorrow.
    """
    now_ny = get_ny_time()
    target = now_ny.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now_ny:
        from datetime import timedelta
        target += timedelta(days=1)
    return target.astimezone(UTC)


class BrainScheduler:
    """Wraps APScheduler with SQLite job store for durable scheduling."""

    def __init__(self):
        DB_DIR.mkdir(parents=True, exist_ok=True)
        jobstore_path = str(DB_PATH)

        self._jobstores = {
            "default": SQLAlchemyJobStore(url=f"sqlite:///{jobstore_path}")
        }
        self._executors = {
            "default": ThreadPoolExecutor(max_workers=10)
        }
        self._scheduler = BackgroundScheduler(
            jobstores=self._jobstores,
            executors=self._executors,
            timezone=UTC,
        )
        self._job_ids: set[str] = set()

    def add_daily(self, job_id: str, func, hour: int, minute: int = 7):
        """Schedule a daily job at NY time (hour, minute)."""
        # APScheduler cron: minute hour * * *
        self._scheduler.add_job(
            func,
            trigger="cron",
            hour=hour,
            minute=minute,
            timezone=EASTERN,
            id=job_id,
            replace_existing=True,
            misfire_grace_time=600,  # 10 min grace
        )
        self._job_ids.add(job_id)
        logger.info(f"Scheduled '{job_id}' daily at {hour:02d}:{minute:02d} ET")

    def add_weekly(self, job_id: str, func, day_of_week: str, hour: int, minute: int = 7):
        """Schedule a weekly job. day_of_week: 'mon','tue','wed','thu','fri','sat','sun'."""
        self._scheduler.add_job(
            func,
            trigger="cron",
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            timezone=EASTERN,
            id=job_id,
            replace_existing=True,
            misfire_grace_time=3600,  # 1h grace for weekly
        )
        self._job_ids.add(job_id)
        logger.info(f"Scheduled '{job_id}' weekly on {day_of_week} at {hour:02d}:{minute:02d} ET")

    def add_interval(self, job_id: str, func, minutes: int):
        """Schedule a recurring interval job."""
        self._scheduler.add_job(
            func,
            trigger="interval",
            minutes=minutes,
            id=job_id,
            replace_existing=True,
            misfire_grace_time=300,
        )
        self._job_ids.add(job_id)
        logger.info(f"Scheduled '{job_id}' every {minutes} min")

    def start(self):
        self._scheduler.start()
        logger.info(f"Scheduler started with {len(self._job_ids)} jobs")

    def stop(self):
        self._scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped")

    @property
    def jobs(self):
        return self._scheduler.get_jobs()
