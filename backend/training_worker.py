import argparse
import logging
from pathlib import Path

from training import run_training_worker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--candidate-path", type=Path, required=True)
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    return run_training_worker(
        arguments.dataset_id,
        arguments.epochs,
        arguments.run_directory,
        arguments.status_path,
        arguments.candidate_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
