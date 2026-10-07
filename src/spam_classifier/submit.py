"""Upload a trained model to the HTB evaluation endpoint."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from . import config

log = logging.getLogger(__name__)


def upload_model(path: Path = config.MODEL_PATH, url: str = config.UPLOAD_URL) -> dict:
    with open(path, "rb") as fh:
        response = requests.post(url, files={"model": fh}, timeout=config.HTTP_TIMEOUT)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text}
    log.info("Server response:\n%s", json.dumps(payload, indent=4))
    return payload
