"""Central configuration. Override paths/URLs via environment variables."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_URL: str = os.getenv(
    "SPAM_DATASET_URL",
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
)
DATA_DIR: Path = Path(os.getenv("SPAM_DATA_DIR", PROJECT_ROOT / "data" / "sms_spam_collection"))
DATASET_FILE: Path = DATA_DIR / "SMSSpamCollection"

MODEL_DIR: Path = Path(os.getenv("SPAM_MODEL_DIR", PROJECT_ROOT / "models"))
MODEL_PATH: Path = MODEL_DIR / "spam_detection_model.joblib"

UPLOAD_URL: str = os.getenv("SPAM_UPLOAD_URL", "http://localhost:8000/api/upload")

# Hyperparameter search space for MultinomialNB smoothing
ALPHA_GRID: list[float] = [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
CV_FOLDS: int = 5
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
HTTP_TIMEOUT: int = 30
