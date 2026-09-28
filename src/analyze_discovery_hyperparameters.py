"""Find the best Discovery hyperparameters from a CSV or JSON results file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


HYPERPARAMETER_FIELDS = [
    "bm25_k1",
    "bm25_b",
    "bm25_epsilon",
    "batch_size",
    "discovery_retry_limit",
    "rrf_k",
    "label_weight",
    "classes_weight",
    "properties_weight",
    "description_weight",
    "examples_weight",
]

DEFAULT_COLUMNS = [
    "rank",
    "run",
    "discovery_accuracy_mean",
    "discovery_f1_mean",
    "n_examples",
    "avg_predicted_endpoints",
    "avg_discovery_internal_retries",
    *HYPERPARAMETER_FIELDS,
]


def coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped == "":
        return None

    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if any(char in stripped for char in [".", "e", "E"]):
            return float(stripped)
        return int(stripped)
    except ValueError:
        return value


def coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: coerce_scalar(value) for key, value in row.items()}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def flatten_dense_weights(row: dict[str, Any], dense_weights: dict[str, Any] | None) -> None:
    if not dense_weights:
        return

    for field in ["label", "classes", "properties", "description", "examples"]:
        key = f"{field}_weight"
        if key not in row and field in dense_weights:
            row[key] = dense_weights[field]


def flatten_config(row: dict[str, Any], config: dict[str, Any] | None) -> None:
    if not config:
        return

    for key, value in config.items():
        if key == "dense_weights":
            flatten_dense_weights(row, value)
        elif key not in row:
            row[key] = value


def summarize_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy_values = [
        float(row["discovery_accuracy"])
        for row in examples
        if row.get("discovery_accuracy") is not None
    ]
    f1_values = [
        float(row["discovery_f1"])
        for row in examples
        if row.get("discovery_f1") is not None
    ]
    predicted_sizes = [
        float(row["predicted_size"])
        for row in examples
        if row.get("predicted_size") is not None
    ]
    gold_sizes = [
        float(row["gold_size"])
        for row in examples
        if row.get("gold_size") is not None
    ]
    retry_values = [
        float(row["discovery_internal_retries"])
        for row in examples
        if row.get("discovery_internal_retries") is not None
    ]

    return {
        "n_examples": len(examples),
        "discovery_accuracy_mean": mean(accuracy_values),
        "discovery_f1_mean": mean(f1_values),
        "avg_predicted_endpoints": mean(predicted_sizes),
        "avg_gold_endpoints": mean(gold_sizes),
        "avg_discovery_internal_retries": mean(retry_values),
    }


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as f:
        return [coerce_row(row) for row in csv.DictReader(f)]


def rows_from_checkpoint(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    runs = payload.get("runs", {})

    for run_key in sorted(runs, key=lambda value: int(value) if str(value).isdigit() else str(value)):
        run_state = runs[run_key]
        summary = run_state.get("summary") or {}
        examples = run_state.get("examples") or []
        row = dict(summary)

        if not row and examples:
            row.update(summarize_examples(examples))

        if not row:
            continue

        row["run"] = coerce_scalar(row.get("run", run_key))
        row["status"] = run_state.get("status")
        flatten_config(row, run_state.get("config"))
        rows.append(coerce_row(row))

    return rows


def rows_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("runs"), dict):
        return rows_from_checkpoint(payload)

    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        row = summarize_examples(payload["results"])
        flatten_config(row, payload.get("config"))
        return [coerce_row(row)]

    if isinstance(payload, list):
        if all(isinstance(row, dict) for row in payload):
            if payload and "discovery_accuracy_mean" not in payload[0] and "discovery_accuracy" in payload[0]:
                return [coerce_row(summarize_examples(payload))]
            return [coerce_row(row) for row in payload]

    if isinstance(payload, dict):
        return [coerce_row(payload)]

    raise ValueError("Unsupported JSON shape. Expected a checkpoint object, result object, list, or row object.")


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = load_csv(path)
    elif suffix == ".json":
        with path.open("r") as f:
            rows = rows_from_json(json.load(f))
    else:
        raise ValueError(f"Unsupported file extension `{path.suffix}`. Use .csv or .json.")

    if not rows:
        raise ValueError(f"No analyzable rows found in {path}.")
    return rows


def metric_value(row: dict[str, Any], metric: str, default: float) -> float:
    value = row.get(metric)
    if value is None:
        return default

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(numeric):
        return default
    return numeric


def ranked_rows(
    rows: list[dict[str, Any]],
    primary: str,
    secondary: str,
    minimize: list[str],
) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        maximize_keys = [
            metric_value(row, primary, float("-inf")),
            metric_value(row, secondary, float("-inf")),
            metric_value(row, "n_examples", float("-inf")),
        ]
        minimize_keys = [
            -metric_value(row, metric, float("inf"))
            for metric in minimize
        ]
        return tuple(maximize_keys + minimize_keys)

    return sorted(rows, key=sort_key, reverse=True)


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    rendered = [
        {column: format_value(row.get(column)) for column in columns}
        for row in rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rendered))
        for column in columns
    }

    header = "  ".join(column.ljust(widths[column]) for column in columns)
    separator = "  ".join("-" * widths[column] for column in columns)
    print(header)
    print(separator)
    for row in rendered:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def best_config(row: dict[str, Any]) -> dict[str, Any]:
    config = {
        field: row[field]
        for field in HYPERPARAMETER_FIELDS
        if row.get(field) is not None
    }
    dense_weights = {
        field.replace("_weight", ""): config.pop(field)
        for field in [
            "label_weight",
            "classes_weight",
            "properties_weight",
            "description_weight",
            "examples_weight",
        ]
        if field in config
    }
    if dense_weights:
        config["dense_weights"] = dense_weights
    return config


def print_recommended_command(row: dict[str, Any]) -> None:
    flags = []
    flag_names = {
        "bm25_k1": "--bm25-k1",
        "bm25_b": "--bm25-b",
        "bm25_epsilon": "--bm25-epsilon",
        "batch_size": "--batch-size",
        "discovery_retry_limit": "--discovery-retry-limit",
        "rrf_k": "--rrf-k",
        "label_weight": "--label-weight",
        "classes_weight": "--classes-weight",
        "properties_weight": "--properties-weight",
        "description_weight": "--description-weight",
        "examples_weight": "--examples-weight",
    }
    for key in HYPERPARAMETER_FIELDS:
        if row.get(key) is not None:
            flags.extend([flag_names[key], str(row[key])])

    print("\nRecommended ablation command for this config:")
    print("python3 src/ablate_discovery.py " + " ".join(flags))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank Discovery hyperparameter results from a CSV or JSON file."
    )
    parser.add_argument("input", help="CSV summary or JSON checkpoint/results file.")
    parser.add_argument("--top", type=int, default=5, help="Number of ranked rows to print.")
    parser.add_argument(
        "--primary",
        default="discovery_accuracy_mean",
        help="Primary metric to maximize.",
    )
    parser.add_argument(
        "--secondary",
        default="discovery_f1_mean",
        help="Secondary metric to maximize.",
    )
    parser.add_argument(
        "--minimize",
        nargs="*",
        default=["avg_predicted_endpoints", "avg_discovery_internal_retries"],
        help="Tie-breaker metrics to minimize after primary/secondary/n_examples.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the best hyperparameter config as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    rows = ranked_rows(
        load_rows(input_path),
        primary=args.primary,
        secondary=args.secondary,
        minimize=args.minimize,
    )

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    best = rows[0]
    top_rows = rows[: max(args.top, 1)]

    print(f"Loaded {len(rows)} run(s) from {input_path}")
    print(f"Ranking: maximize {args.primary}, then {args.secondary}; minimize {', '.join(args.minimize)}")
    print()
    print_table(top_rows, [column for column in DEFAULT_COLUMNS if any(column in row for row in top_rows)])

    print("\nBest hyperparameters:")
    print(json.dumps(best_config(best), indent=2))
    print_recommended_command(best)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(best_config(best), f, indent=2)
            f.write("\n")
        print(f"\nSaved best config to {output_path}")


if __name__ == "__main__":
    main()
