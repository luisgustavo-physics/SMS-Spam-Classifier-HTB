import io
import zipfile

import pytest

from spam_classifier.data import _safe_extract
from spam_classifier.model import build_pipeline, load_model, save_model
from spam_classifier.preprocessing import preprocess_message


def test_preprocess_keeps_signal_symbols():
    out = preprocess_message("WIN $1000 NOW!!! Call 0800")
    assert "$" in out and "!" in out
    assert not any(ch.isdigit() for ch in out)


def test_preprocess_removes_stopwords_and_stems():
    assert preprocess_message("The winners are winning") == "winner win"


def test_zip_slip_blocked(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with zipfile.ZipFile(buf) as zf, pytest.raises(ValueError):
        _safe_extract(zf, tmp_path)


def test_tampered_model_rejected(tmp_path):
    model = build_pipeline().fit(["free prize win", "see you lunch"], [1, 0])
    path = tmp_path / "m.joblib"
    save_model(model, path)
    assert load_model(path).predict(["free prize"])[0] == 1
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        load_model(path)
