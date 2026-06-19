"""Terminal entry point for claim evidence review pipeline."""

import argparse
import csv
import logging
import sys
from pathlib import Path

# Ensure code directory is on path
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from config import DEFAULT_OUTPUT, DATASET_DIR, SAMPLE_CLAIMS, TEST_CLAIMS
from data_loader import load_csv
from pipeline.orchestrator import process_claims
from schemas import OUTPUT_COLUMNS

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def write_output(results, output_path: Path):
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for r in results:
      writer.writerow(r.to_csv_row())
  logging.info("Wrote %d rows to %s", len(results), output_path)


def main():
  parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review System")
  parser.add_argument(
    "--input",
    type=Path,
    default=TEST_CLAIMS,
    help="Input claims CSV path",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=DEFAULT_OUTPUT,
    help="Output CSV path",
  )
  parser.add_argument(
    "--sample",
    action="store_true",
    help="Run on sample_claims.csv instead",
  )
  args = parser.parse_args()

  input_path = SAMPLE_CLAIMS if args.sample else args.input
  rows = load_csv(input_path)
  logging.info("Loaded %d claims from %s", len(rows), input_path)

  results = process_claims(rows, repo_root=DATASET_DIR.parent)
  write_output(results, args.output)


if __name__ == "__main__":
  main()
