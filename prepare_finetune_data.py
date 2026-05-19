#!/usr/bin/env python3
"""Prepare instruction-style JSONL data from defects_data.csv for LLM fine-tuning."""

import argparse
import csv
import json
import random
from pathlib import Path


FEATURE_COLUMNS = [
    "product_id",
    "defect_type",
    "defect_date",
    "defect_location",
    "inspection_method",
]

TARGET_COLUMNS = ["severity", "repair_cost"]


def row_to_example(row: dict) -> dict:
    features = {k: row[k] for k in FEATURE_COLUMNS}
    target = {
        "severity": row["severity"],
        "repair_cost": round(float(row["repair_cost"]), 2),
    }

    prompt = (
        "Given manufacturing defect details, predict the expected severity label and repair cost. "
        "Return only valid JSON with keys severity and repair_cost."
    )

    return {
        "instruction": prompt,
        "input": features,
        "output": target,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create train/val JSONL for fine-tuning")
    parser.add_argument("--csv", default="defects_data.csv", help="Input CSV path")
    parser.add_argument("--out_dir", default="training_data", help="Output directory")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV has no rows")

    missing = [c for c in FEATURE_COLUMNS + TARGET_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    examples = [row_to_example(r) for r in rows]

    random.seed(args.seed)
    random.shuffle(examples)

    val_size = max(1, int(len(examples) * args.val_ratio))
    val_records = examples[:val_size]
    train_records = examples[val_size:]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    print(f"Wrote {len(train_records)} training rows to {train_path}")
    print(f"Wrote {len(val_records)} validation rows to {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
