import json
from datetime import datetime
from pathlib import Path

# Logs directory at project root
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def log_event(filename: str, payload: dict):
    """
    Append a structured JSON log entry to a JSONL file.
    """
    payload["timestamp"] = datetime.utcnow().isoformat()

    path = LOG_DIR / filename
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
