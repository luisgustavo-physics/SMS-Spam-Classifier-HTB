"""Dataset acquisition, safe extraction and loading."""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract a zip while blocking path traversal (Zip Slip)."""
    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not target.is_relative_to(dest):
            raise ValueError(f"Unsafe path in archive: {member.filename}")
    zf.extractall(dest)


def download_dataset(url: str = config.DATASET_URL, dest: Path = config.DATA_DIR,
                     force: bool = False) -> Path:
    """Download and extract the UCI SMS Spam Collection. Returns the data file path."""
    data_file = dest / "SMSSpamCollection"
    if data_file.exists() and not force:
        log.info("Dataset already present at %s", data_file)
        return data_file

    log.info("Downloading dataset from %s", url)
    response = requests.get(url, timeout=config.HTTP_TIMEOUT)
    response.raise_for_status()

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        _safe_extract(zf, dest)
    log.info("Extracted: %s", sorted(p.name for p in dest.iterdir()))
    return data_file


def load_dataset(path: Path = config.DATASET_FILE) -> pd.DataFrame:
    """Load TSV into a DataFrame, drop nulls/duplicates, add binary target column."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"],
                     quoting=3, encoding="utf-8")
    before = len(df)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    log.info("Loaded %d rows (%d removed as null/duplicate)", len(df), before - len(df))
    df["target"] = (df["label"] == "spam").astype(int)
    log.info("Class balance: %s", df["label"].value_counts().to_dict())
    return df
