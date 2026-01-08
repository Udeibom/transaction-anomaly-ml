from pathlib import Path
import pandas as pd

# -----------------------------
# Resolve project root safely
# -----------------------------
BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()


def log_metrics(metrics: dict, relative_path: str):
    """
    Append metrics to a CSV file in a path-safe way.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics to log
    relative_path : str
        Path relative to project root (e.g. 'data/monitoring/metrics.csv')
    """

    # Resolve full path relative to project root
    path = BASE_DIR / relative_path

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame([metrics])

    # Append or create
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
