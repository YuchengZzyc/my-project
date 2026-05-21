from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reminder_notifier import ReminderPopupDaemon
from app.storage import JSONReminderStorage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run popup reminder watcher.")
    parser.add_argument("--storage-path", default="data/reminders.json", help="Path to reminders JSON file.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Polling interval in seconds.")
    parser.add_argument("--lead-minutes", type=int, default=30, help="Notify when reminder enters this many minutes before scheduled_time.")
    parser.add_argument("--once", action="store_true", help="Run a single poll and exit.")
    args = parser.parse_args()

    daemon = ReminderPopupDaemon(
        storage=JSONReminderStorage(args.storage_path),
        poll_interval_seconds=args.poll_interval,
        lead_time_minutes=args.lead_minutes,
    )

    if args.once:
        notified = daemon.poll_once()
        print({"notified_count": len(notified), "notified_ids": notified})
        return

    print(
        {
            "status": "watching",
            "storage_path": args.storage_path,
            "poll_interval_seconds": args.poll_interval,
            "lead_minutes": args.lead_minutes,
        }
    )
    daemon.run_forever()


if __name__ == "__main__":
    main()
