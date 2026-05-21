from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Callable

from app.storage import JSONReminderStorage


Notifier = Callable[[dict[str, Any]], None]


def _default_popup_notifier(reminder: dict[str, Any]) -> None:
    title = "Reminder"
    task = reminder.get("task", "(no task)")
    time_text = reminder.get("time_text", "")
    rid = reminder.get("reminder_id", "")
    body = f"{task}\n\nTime: {time_text}\nID: {rid}"
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showinfo(title, body)
        root.destroy()
    except Exception:
        # Fallback for environments where tkinter or GUI is unavailable.
        print(f"[REMINDER] {body}")


class ReminderPopupDaemon:
    def __init__(
        self,
        storage: JSONReminderStorage,
        notifier: Notifier | None = None,
        poll_interval_seconds: float = 5.0,
        lead_time_minutes: int = 30,
    ) -> None:
        self.storage = storage
        self.notifier = notifier or _default_popup_notifier
        self.poll_interval_seconds = poll_interval_seconds
        self.lead_time_minutes = max(0, lead_time_minutes)

    def _parse_scheduled_time(self, value: str) -> datetime | None:
        try:
            dt = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
        if dt.tzinfo is None:
            return dt.astimezone()
        return dt

    def _is_due(self, reminder: dict[str, Any], now: datetime) -> bool:
        if reminder.get("status") != "active":
            return False
        scheduled = self._parse_scheduled_time(reminder.get("scheduled_time", ""))
        if scheduled is None:
            return False
        window_start = scheduled - timedelta(minutes=self.lead_time_minutes)
        notified_at = self._parse_scheduled_time(reminder.get("notified_at", ""))
        # If a previous notification timestamp is already inside this window,
        # treat it as already notified; otherwise allow notification again.
        if notified_at is not None and notified_at >= window_start:
            return False
        return window_start <= now <= scheduled

    def poll_once(self, now: datetime | None = None) -> list[str]:
        current = now or datetime.now().astimezone()
        reminders = self.storage.load()
        notified_ids: list[str] = []
        changed = False

        for reminder in reminders:
            if not self._is_due(reminder, current):
                continue
            self.notifier(reminder)
            reminder["notified_at"] = current.isoformat(timespec="seconds")
            reminder["updated_at"] = current.isoformat(timespec="seconds")
            notified_ids.append(reminder.get("reminder_id", ""))
            changed = True

        if changed:
            self.storage.save(reminders)
        return notified_ids

    def run_forever(self) -> None:
        while True:
            self.poll_once()
            time.sleep(self.poll_interval_seconds)
