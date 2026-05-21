from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.reminder_notifier import ReminderPopupDaemon
from app.storage import JSONReminderStorage


def test_popup_daemon_notifies_within_half_hour_window_and_marks_notified(tmp_path):
    tz = timezone(timedelta(hours=8))
    now = datetime(2026, 5, 19, 17, 35, 0, tzinfo=tz)

    storage = JSONReminderStorage(tmp_path / "reminders.json")
    storage.save(
        [
            {
                "reminder_id": "rem_0001",
                "task": "play basketball",
                "scheduled_time": "2026-05-19T18:00:00+08:00",
                "time_text": "today 4pm",
                "target": "self",
                "status": "active",
                "created_at": "2026-05-19T15:00:00+08:00",
                "updated_at": "2026-05-19T15:00:00+08:00",
            },
            {
                "reminder_id": "rem_0002",
                "task": "drink water",
                "scheduled_time": "2026-05-19T18:20:00+08:00",
                "time_text": "today 6pm",
                "target": "self",
                "status": "active",
                "created_at": "2026-05-19T15:00:00+08:00",
                "updated_at": "2026-05-19T15:00:00+08:00",
            },
        ]
    )

    popped: list[str] = []

    def fake_notifier(reminder):
        popped.append(reminder["reminder_id"])

    daemon = ReminderPopupDaemon(
        storage=storage,
        notifier=fake_notifier,
        poll_interval_seconds=0.1,
        lead_time_minutes=30,
    )
    notified = daemon.poll_once(now=now)
    assert notified == ["rem_0001"]
    assert popped == ["rem_0001"]

    saved = storage.load()
    due = next(r for r in saved if r["reminder_id"] == "rem_0001")
    future = next(r for r in saved if r["reminder_id"] == "rem_0002")
    assert due.get("notified_at") == "2026-05-19T17:35:00+08:00"
    assert future.get("notified_at") is None

    notified_again = daemon.poll_once(now=now)
    assert notified_again == []
    assert popped == ["rem_0001"]


def test_popup_daemon_ignores_stale_notified_at_from_old_schedule(tmp_path):
    tz = timezone(timedelta(hours=8))
    now = datetime(2026, 5, 19, 17, 22, 0, tzinfo=tz)

    storage = JSONReminderStorage(tmp_path / "reminders.json")
    storage.save(
        [
            {
                "reminder_id": "rem_0001",
                "task": "play basketball",
                "scheduled_time": "2026-05-19T17:51:48+08:00",
                "time_text": "tomorrow at 4:00pm",
                "target": "self",
                "status": "active",
                "created_at": "2026-05-14T16:56:48+08:00",
                "updated_at": "2026-05-19T17:09:56+08:00",
                "notified_at": "2026-05-19T17:09:56+08:00",
            }
        ]
    )

    popped: list[str] = []

    def fake_notifier(reminder):
        popped.append(reminder["reminder_id"])

    daemon = ReminderPopupDaemon(storage=storage, notifier=fake_notifier, lead_time_minutes=30)
    notified = daemon.poll_once(now=now)
    assert notified == ["rem_0001"]
    assert popped == ["rem_0001"]
