"""Command-line entry point: download | train | predict | submit."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config
from .data import download_dataset, load_dataset
from .model import load_model, save_model, train
from .predict import SAMPLE_MESSAGES, predict
from .preprocessing import ensure_nltk_resources, preprocess_corpus
from .submit import upload_model


def cmd_download(a: argparse.Namespace) -> None:
    download_dataset(force=a.force)


def cmd_train(a: argparse.Namespace) -> None:
    ensure_nltk_resources()
    df = load_dataset(download_dataset())
    model, _ = train(preprocess_corpus(df["message"]), df["target"].values,
                     holdout=not a.no_holdout)
    save_model(model, a.model)
    for p in predict(model, SAMPLE_MESSAGES):
        print(p)


def cmd_predict(a: argparse.Namespace) -> None:
    model = load_model(a.model, verify=not a.skip_verify)
    for p in predict(model, a.messages or SAMPLE_MESSAGES):
        print(p)


def cmd_submit(a: argparse.Namespace) -> None:
    upload_model(a.model, a.url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spam-classifier", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="Fetch the UCI SMS Spam Collection")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_download)

    t = sub.add_parser("train", help="Preprocess, grid-search, evaluate, save")
    t.add_argument("--model", type=Path, default=config.MODEL_PATH)
    t.add_argument("--no-holdout", action="store_true",
                   help="Fit on full data only (original notebook behaviour)")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="Classify messages")
    p.add_argument("messages", nargs="*")
    p.add_argument("--model", type=Path, default=config.MODEL_PATH)
    p.add_argument("--skip-verify", action="store_true", help="Skip SHA-256 check (unsafe)")
    p.set_defaults(func=cmd_predict)

    s = sub.add_parser("submit", help="Upload model to evaluation API")
    s.add_argument("--model", type=Path, default=config.MODEL_PATH)
    s.add_argument("--url", default=config.UPLOAD_URL)
    s.set_defaults(func=cmd_submit)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001 — top-level CLI guard
        logging.getLogger("spam_classifier").error("%s", exc, exc_info=args.verbose)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
