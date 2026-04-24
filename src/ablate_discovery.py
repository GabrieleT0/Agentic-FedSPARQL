"""Evaluate the discovery module in isolation and run simple ablation sweeps.

Examples:
  python3 src/ablate_discovery.py
  python3 src/ablate_discovery.py --split dev --limit 50 --label-weight 0.2 --properties-weight 0.35
  python3 src/ablate_discovery.py --grid-config data/discovery_grid.json --output data/discovery_ablation.csv
"""

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
from typing import Any

from config import BENCHMARK_DATA_PATH
from metrics import discovery_accuracy
from modules.discovery2 import DEFAULT_DENSE_WEIGHTS, Discovery2
from recalculate_discovery_metrics import extract_service_endpoints


def discovery_f1(predicted_endpoints: list[str], gold_endpoints: list[str]) -> float:
    predicted_set = set(predicted_endpoints or [])
    gold_set = set(gold_endpoints or [])

    if not predicted_set and not gold_set:
        return 1.0
    if not predicted_set or not gold_set:
        return 0.0

    tp = len(predicted_set & gold_set)
    precision = tp / len(predicted_set)
    recall = tp / len(gold_set)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_examples(benchmark_path: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    with open(benchmark_path, "r") as f:
        data = json.load(f)

    examples = [
        ex for ex in data
        if ex.get("validation", {}).get("valid")
        and ex.get("validation", {}).get("original_count", 0) > 0
        and (split == "all" or ex.get("split") == split)
    ]

    if limit is not None:
        examples = examples[:limit]
    return examples


def build_dense_weights(args: argparse.Namespace) -> dict[str, float]:
    weights = DEFAULT_DENSE_WEIGHTS.copy()
    cli_overrides = {
        "label": args.label_weight,
        "classes": args.classes_weight,
        "properties": args.properties_weight,
        "description": args.description_weight,
        "examples": args.examples_weight,
    }
    for field, value in cli_overrides.items():
        if value is not None:
            weights[field] = value
    return weights


def gold_endpoints_for_example(example: dict[str, Any], use_service_endpoints: bool) -> list[str]:
    if use_service_endpoints:
        return extract_service_endpoints(example.get("federated_sparql", ""))
    return [ep["url"] for ep in example.get("endpoints", [])]


def evaluate_config(
    examples: list[dict[str, Any]],
    *,
    model_name: str,
    dense_weights: dict[str, float],
    bm25_k1: float,
    bm25_b: float,
    bm25_epsilon: float,
    batch_size: int,
    discovery_retry_limit: int,
    rrf_k: int,
    use_service_endpoints: bool,
) -> dict[str, Any]:
    discovery = Discovery2(
        model_name=model_name,
        dense_weights=dense_weights,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        bm25_epsilon=bm25_epsilon,
        batch_size=batch_size,
        discovery_retry_limit=discovery_retry_limit,
        rrf_k=rrf_k,
    )

    accuracy_values = []
    f1_values = []
    predicted_sizes = []
    gold_sizes = []
    retry_values = []

    for idx, example in enumerate(examples, start=1):
        gold_endpoints = gold_endpoints_for_example(example, use_service_endpoints=use_service_endpoints)
        prediction = discovery(question=example["question"])
        predicted_endpoints = prediction.candidate_endpoints or []

        accuracy_values.append(discovery_accuracy(predicted_endpoints, gold_endpoints))
        f1_values.append(discovery_f1(predicted_endpoints, gold_endpoints))
        predicted_sizes.append(len(set(predicted_endpoints)))
        gold_sizes.append(len(set(gold_endpoints)))
        retry_values.append(prediction.internal_retries or 0)

        print(
            f"[{idx}/{len(examples)}] "
            f"acc={accuracy_values[-1]:.0f} f1={f1_values[-1]:.4f} "
            f"pred={len(set(predicted_endpoints))} gold={len(set(gold_endpoints))}"
        )

    normalized_weights = Discovery2._resolve_dense_weights(dense_weights)
    return {
        "n_examples": len(examples),
        "discovery_accuracy_mean": round(mean(accuracy_values), 4),
        "discovery_f1_mean": round(mean(f1_values), 4),
        "avg_predicted_endpoints": round(mean(predicted_sizes), 4),
        "avg_gold_endpoints": round(mean(gold_sizes), 4),
        "avg_discovery_internal_retries": round(mean(retry_values), 4),
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "bm25_epsilon": bm25_epsilon,
        "batch_size": batch_size,
        "discovery_retry_limit": discovery_retry_limit,
        "rrf_k": rrf_k,
        "label_weight": round(normalized_weights["label"], 6),
        "classes_weight": round(normalized_weights["classes"], 6),
        "properties_weight": round(normalized_weights["properties"], 6),
        "description_weight": round(normalized_weights["description"], 6),
        "examples_weight": round(normalized_weights["examples"], 6),
    }


def load_grid_configs(path: str, base_config: dict[str, Any]) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        search_space = json.load(f)

    dense_weight_configs = search_space.pop("dense_weights", [base_config["dense_weights"]])
    scalar_keys = sorted(search_space.keys())
    scalar_value_lists = [search_space[key] for key in scalar_keys]

    configs = []
    for dense_weights in dense_weight_configs:
        for values in itertools.product(*scalar_value_lists) if scalar_keys else [()]:
            config = base_config.copy()
            config["dense_weights"] = dense_weights
            for key, value in zip(scalar_keys, values):
                config[key] = value
            configs.append(config)
    return configs


def write_results(rows: list[dict[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the discovery module only.")
    parser.add_argument("--benchmark", default=BENCHMARK_DATA_PATH, help="Path to the benchmark JSON.")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "all"], help="Benchmark split to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of examples.")
    parser.add_argument("--model-name", default="nomic-ai/nomic-embed-text-v1.5", help="SentenceTransformer model.")
    parser.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 k1.")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b.")
    parser.add_argument("--bm25-epsilon", type=float, default=0.25, help="BM25 epsilon floor.")
    parser.add_argument("--batch-size", type=int, default=100, help="Discovery batch size.")
    parser.add_argument("--discovery-retry-limit", type=int, default=3, help="Max discovery batches and per-batch retries.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF k parameter.")
    parser.add_argument("--label-weight", type=float, default=None, help="Dense weight for endpoint labels.")
    parser.add_argument("--classes-weight", type=float, default=None, help="Dense weight for classes.")
    parser.add_argument("--properties-weight", type=float, default=None, help="Dense weight for properties.")
    parser.add_argument("--description-weight", type=float, default=None, help="Dense weight for descriptions.")
    parser.add_argument("--examples-weight", type=float, default=None, help="Dense weight for example queries.")
    parser.add_argument(
        "--use-service-endpoints",
        action="store_true",
        help="Use SERVICE endpoints extracted from federated_sparql as gold endpoints.",
    )
    parser.add_argument(
        "--grid-config",
        default=None,
        help="Optional JSON file defining a grid search. Scalars should be arrays; dense_weights should be a list of weight dicts.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(BENCHMARK_DATA_PATH), "..", "discovery_ablation_results.csv"),
        help="CSV file for ablation results.",
    )
    args = parser.parse_args()

    examples = load_examples(args.benchmark, split=args.split, limit=args.limit)
    if not examples:
        raise ValueError("No benchmark examples matched the requested filters.")

    base_config = {
        "model_name": args.model_name,
        "dense_weights": build_dense_weights(args),
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "bm25_epsilon": args.bm25_epsilon,
        "batch_size": args.batch_size,
        "discovery_retry_limit": args.discovery_retry_limit,
        "rrf_k": args.rrf_k,
    }

    if args.grid_config:
        configs = load_grid_configs(args.grid_config, base_config=base_config)
    else:
        configs = [base_config]

    rows = []
    for run_idx, config in enumerate(configs, start=1):
        print(f"\n=== Discovery Ablation Run {run_idx}/{len(configs)} ===")
        print(json.dumps(config, indent=2))
        row = evaluate_config(
            examples,
            model_name=config["model_name"],
            dense_weights=config["dense_weights"],
            bm25_k1=float(config["bm25_k1"]),
            bm25_b=float(config["bm25_b"]),
            bm25_epsilon=float(config["bm25_epsilon"]),
            batch_size=int(config["batch_size"]),
            discovery_retry_limit=int(config["discovery_retry_limit"]),
            rrf_k=int(config["rrf_k"]),
            use_service_endpoints=args.use_service_endpoints,
        )
        row["run"] = run_idx
        row["split"] = args.split
        row["service_gold"] = args.use_service_endpoints
        rows.append(row)
        print(
            f"Run {run_idx}: accuracy={row['discovery_accuracy_mean']:.4f}, "
            f"f1={row['discovery_f1_mean']:.4f}"
        )

    rows.sort(key=lambda row: (row["discovery_accuracy_mean"], row["discovery_f1_mean"]), reverse=True)
    write_results(rows, args.output)

    print(f"\nSaved {len(rows)} run(s) to {args.output}")
    print("Top configurations:")
    for row in rows[:5]:
        print(
            f"  run={row['run']} acc={row['discovery_accuracy_mean']:.4f} "
            f"f1={row['discovery_f1_mean']:.4f} "
            f"bm25(k1={row['bm25_k1']}, b={row['bm25_b']}, eps={row['bm25_epsilon']}) "
            f"weights=({row['label_weight']}, {row['classes_weight']}, {row['properties_weight']}, "
            f"{row['description_weight']}, {row['examples_weight']})"
        )


if __name__ == "__main__":
    main()
