"""Pipeline construction, hyperparameter search, evaluation and persistence."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from . import config

log = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    """Bag-of-words (unigrams+bigrams) -> Multinomial Naive Bayes."""
    return Pipeline([
        ("vectorizer", CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))),
        ("classifier", MultinomialNB()),
    ])


def train(messages: list[str], y, holdout: bool = True) -> tuple[Pipeline, dict]:
    """Grid-search alpha with F1 scoring.

    With ``holdout=True`` a stratified test split is kept aside for an honest
    evaluation, then the final model is refit on all data with the best alpha.
    """
    metrics: dict = {}
    X_train, y_train = messages, y
    if holdout:
        X_train, X_test, y_train, y_test = train_test_split(
            messages, y, test_size=config.TEST_SIZE, stratify=y,
            random_state=config.RANDOM_STATE)

    search = GridSearchCV(build_pipeline(), {"classifier__alpha": config.ALPHA_GRID},
                          cv=config.CV_FOLDS, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    metrics["best_params"] = search.best_params_
    metrics["cv_f1"] = float(search.best_score_)
    log.info("Best params %s | CV F1 %.4f", search.best_params_, search.best_score_)

    if holdout:
        y_pred = search.best_estimator_.predict(X_test)
        metrics["report"] = classification_report(
            y_test, y_pred, target_names=["ham", "spam"], digits=4)
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        log.info("Hold-out evaluation:\n%s", metrics["report"])
        final = build_pipeline().set_params(**search.best_params_).fit(messages, y)
    else:
        final = search.best_estimator_
    return final, metrics


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(model: Pipeline, path: Path = config.MODEL_PATH) -> str:
    """Persist model and write a sidecar .sha256 for integrity checking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    digest = sha256_of(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    log.info("Model saved to %s (sha256=%s)", path, digest)
    return digest


def load_model(path: Path = config.MODEL_PATH, verify: bool = True) -> Pipeline:
    """Load a joblib model. joblib uses pickle => arbitrary code execution on load.

    Only load artifacts you produced; the SHA-256 sidecar check guards against tampering.
    """
    if verify:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.exists():
            raise FileNotFoundError(f"Missing integrity file {sidecar}; refusing to unpickle.")
        expected = sidecar.read_text().split()[0]
        if sha256_of(path) != expected:
            raise ValueError("Model hash mismatch — artifact may be tampered with.")
    return joblib.load(path)
