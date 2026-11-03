"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from pagrus_age.config import load_config
from pagrus_age.data import download, ingest
from pagrus_age.predict import drift_report, load_bundle, load_input, predict_file
from pagrus_age.train import run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pagrus-age", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="fetch the raw CSV and verify its checksum")
    commands.add_parser("ingest", help="validate the contract, apply QC and write parquet")
    train = commands.add_parser("train", help="select, evaluate, gate and save a model")
    train.add_argument("--no-test", action="store_true", help="stop before touching the test set")
    predict = commands.add_parser("predict", help="batch inference")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--unaged-only", action="store_true")
    predict.add_argument("--run-id", help="use this run even if it was not promoted")
    drift = commands.add_parser("drift", help="compare a batch with the training reference")
    drift.add_argument("--input", required=True)
    drift.add_argument("--run-id")
    serve = commands.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "download":
        print(download(cfg))
    elif args.command == "ingest":
        print(json.dumps(ingest(cfg)[1], indent=2))
    elif args.command == "train":
        print(json.dumps(run(args.config, no_test=args.no_test), indent=2, default=str))
    elif args.command == "predict":
        bundle = load_bundle(cfg.paths.models_dir, args.run_id)
        result = predict_file(bundle, args.input, args.output, cfg.qc, args.unaged_only)
        print(f"{len(result)} predictions written to {args.output}")
    elif args.command == "drift":
        bundle = load_bundle(cfg.paths.models_dir, args.run_id)
        report = drift_report(bundle, load_input(args.input, cfg.qc))
        print(json.dumps(report, indent=2))
        return 1 if report["alerts"] else 0
    elif args.command == "serve":
        import uvicorn

        from pagrus_age.serve import create_app

        app = create_app(str(cfg.paths.models_dir))
        uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
