"""Run the complete training pipeline or inference with its saved Spark model."""
import argparse
import sys
from pathlib import Path

from pipeline.pipeline import predict, train

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Diabetes prediction using Spark MLlib")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Inspect, clean, split, train, evaluate and save results")
    training.add_argument("--input", type=Path, default=ROOT / "data/raw/03_Dataset.csv")
    training.add_argument("--header", action="store_true", help="Input uses the documented column names")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--demo-patients", type=int, default=10)
    training.add_argument("--ui-hold-seconds", type=int, default=0)
    inference = commands.add_parser("predict", help="Predict supplied patient CSV using the saved best model")
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--model", type=Path, default=ROOT / "outputs/models/best_model")
    for command in (training, inference):
        command.add_argument("--output", type=Path, default=ROOT / "outputs")
        command.add_argument("--master", default="local[2]", help="Spark master, such as local[2] or spark://host:7077")
        command.add_argument("--partitions", type=int, default=2)
    args = parser.parse_args()
    if args.partitions < 1:
        parser.error("--partitions must be positive")
    if args.command == "train" and (args.demo_patients < 1 or args.ui_hold_seconds < 0):
        parser.error("--demo-patients must be positive and --ui-hold-seconds must be nonnegative")
    try:
        if args.command == "train":
            train(args)
        else:
            predict(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
