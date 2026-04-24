"""
Reads all benchmark_result*.json files under data/benchmark_results/
and outputs a CSV summary with mean, std, and count for key metrics.

Optional fields (discovery_internal_retries, schema_summary_retries, query_builder_internal_retries)
are included when present in a file; otherwise reported as N/A.
"""

import csv
import glob
import json
import math
import os
import sys

from recalculate_discovery_metrics import _load_results


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCHMARK_RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "benchmark_results")
SUMMARY_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "benchmark_summary.csv")


METRICS = [
    "execution_accuracy",
    "discovery_accuracy",
    "discovery_f1",
    "f1_score",
    "refinement_attempts",
    "discovery_internal_retries",
    "schema_summary_retries",
    "query_builder_internal_retries",
]


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def summarize_file(path):
    data, results = _load_results(path)

    # Support {"config": ..., "results": [...]} and flat list formats
    if isinstance(data, dict):
        config = data.get("config", {})
    else:
        config = {}

    # Derive a human-readable run label from path and config
    rel = os.path.relpath(path, BENCHMARK_RESULTS_DIR)
    # e.g. azure/gpt-5-mini/benchmark_result_5_retries.json
    parts = rel.replace("\\", "/").split("/")
    provider = parts[0] if len(parts) > 0 else ""
    model = parts[1] if len(parts) > 1 else ""
    filename = os.path.splitext(parts[-1])[0] if len(parts) > 2 else ""

    llm_model = config.get("llm_model", f"{provider}/{model}")
    max_retries = config.get("max_retries", "?")
    timestamp = config.get("timestamp", "")

    row = {
        "file": rel,
        "llm_model": llm_model,
        "max_retries": max_retries,
        "timestamp": timestamp,
        "n_questions": len(results),
    }

    for metric in METRICS:
        values = [r[metric] for r in results if metric in r]
        if not values:
            row[f"{metric}_mean"] = ""
            row[f"{metric}_std"] = ""
            row[f"{metric}_n"] = 0
        else:
            row[f"{metric}_mean"] = round(mean(values), 4)
            row[f"{metric}_std"] = round(std(values), 4)
            row[f"{metric}_n"] = len(values)

    return row


def main():
    pattern = os.path.join(BENCHMARK_RESULTS_DIR, "**", "*.json")
    paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        print(f"No JSON files found under {BENCHMARK_RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    rows = [summarize_file(p) for p in paths]

    # Build CSV columns
    base_cols = ["file", "llm_model", "max_retries", "timestamp", "n_questions"]
    metric_cols = []
    for metric in METRICS:
        metric_cols += [f"{metric}_mean", f"{metric}_std", f"{metric}_n"]

    fieldnames = base_cols + metric_cols

    with open(SUMMARY_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary written to {SUMMARY_CSV_PATH}")
    print(f"Processed {len(rows)} file(s):\n")

    # Pretty-print a quick overview to stdout
    for row in rows:
        print(f"  {row['file']}  (n={row['n_questions']}, model={row['llm_model']}, max_retries={row['max_retries']})")
        for metric in METRICS:
            m = row.get(f"{metric}_mean", "")
            s = row.get(f"{metric}_std", "")
            n = row.get(f"{metric}_n", 0)
            if m == "" or n == 0:
                print(f"    {metric:40s}: N/A")
            else:
                print(f"    {metric:40s}: mean={m:.4f}  std={s:.4f}  (n={n})")
        print()


if __name__ == "__main__":
    main()
