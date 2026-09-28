"""Evaluate the discovery module in isolation and run simple ablation sweeps.

Examples:
  python3 src/ablate_discovery.py
  python3 src/ablate_discovery.py --split dev --limit 50 --label-weight 0.2 --properties-weight 0.35
  python3 src/ablate_discovery.py --grid-config data/discovery_grid.json --output data/discovery_ablation.csv
  python3 src/ablate_discovery.py --grid-config data/discovery_grid.json --search-method random --max-configs 50
  python3 src/ablate_discovery.py --output data/discovery_ablation.csv  # resumes from data/discovery_ablation.checkpoint.json
  python3 ablate_discovery.py --grid-config small_grid.json --split dev --limit 50 --use-service-endpoints --output ../data/discovery_hyperparameters.csv
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable

from config import BENCHMARK_DATA_PATH
from metrics import discovery_accuracy
from modules.discovery import DEFAULT_DENSE_WEIGHTS, Discovery
from recalculate_discovery_metrics import extract_service_endpoints


SEARCH_CONTROL_KEYS = {"search_method", "max_configs", "seed", "search_space"}
DEFAULT_RANDOM_SEED = 20240624


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


def atomic_write_json(output_path: str | Path, payload: dict[str, Any]) -> None:
    """Persist JSON atomically so an interruption cannot corrupt the checkpoint."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_path = tmp.name

    os.replace(tmp_path, output)


def default_checkpoint_path(output_path: str) -> str:
    output = Path(output_path)
    return str(output.with_name(f"{output.stem}.checkpoint.json"))


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def evaluate_example(
    discovery: Discovery,
    example: dict[str, Any],
    example_index: int,
    use_service_endpoints: bool,
) -> dict[str, Any]:
    gold_endpoints = gold_endpoints_for_example(example, use_service_endpoints=use_service_endpoints)
    prediction = discovery(question=example["question"])
    predicted_endpoints = prediction.candidate_endpoints or []
    accuracy = discovery_accuracy(predicted_endpoints, gold_endpoints)
    f1 = discovery_f1(predicted_endpoints, gold_endpoints)

    return {
        "example_index": example_index,
        "question": example["question"],
        "gold_endpoints": gold_endpoints,
        "predicted_endpoints": predicted_endpoints,
        "discovery_accuracy": accuracy,
        "discovery_f1": f1,
        "predicted_size": len(set(predicted_endpoints)),
        "gold_size": len(set(gold_endpoints)),
        "discovery_internal_retries": prediction.internal_retries or 0,
    }


def summarize_example_results(
    example_results: list[dict[str, Any]],
    *,
    dense_weights: dict[str, float],
    bm25_k1: float,
    bm25_b: float,
    bm25_epsilon: float,
    batch_size: int,
    discovery_retry_limit: int,
    rrf_k: int,
) -> dict[str, Any]:
    normalized_weights = Discovery._resolve_dense_weights(dense_weights)
    return {
        "n_examples": len(example_results),
        "discovery_accuracy_mean": round(mean([r["discovery_accuracy"] for r in example_results]), 4),
        "discovery_f1_mean": round(mean([r["discovery_f1"] for r in example_results]), 4),
        "avg_predicted_endpoints": round(mean([r["predicted_size"] for r in example_results]), 4),
        "avg_gold_endpoints": round(mean([r["gold_size"] for r in example_results]), 4),
        "avg_discovery_internal_retries": round(mean([r["discovery_internal_retries"] for r in example_results]), 4),
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
    saved_example_results: list[dict[str, Any]] | None = None,
    on_example_saved: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    example_results = sorted(saved_example_results or [], key=lambda row: row["example_index"])
    completed_indexes = {row["example_index"] for row in example_results}

    if completed_indexes:
        print(f"Resuming run: {len(completed_indexes)}/{len(examples)} example(s) already saved.")

    if len(completed_indexes) == len(examples):
        return summarize_example_results(
            example_results,
            dense_weights=dense_weights,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            bm25_epsilon=bm25_epsilon,
            batch_size=batch_size,
            discovery_retry_limit=discovery_retry_limit,
            rrf_k=rrf_k,
        )

    discovery = Discovery(
        model_name=model_name,
        dense_weights=dense_weights,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        bm25_epsilon=bm25_epsilon,
        batch_size=batch_size,
        discovery_retry_limit=discovery_retry_limit,
        rrf_k=rrf_k,
    )

    for idx, example in enumerate(examples, start=1):
        example_index = idx - 1
        if example_index in completed_indexes:
            continue

        example_result = evaluate_example(
            discovery,
            example,
            example_index,
            use_service_endpoints=use_service_endpoints,
        )
        example_results.append(example_result)
        completed_indexes.add(example_index)

        if on_example_saved is not None:
            on_example_saved(example_result)

        print(
            f"[{idx}/{len(examples)}] "
            f"acc={example_result['discovery_accuracy']:.0f} f1={example_result['discovery_f1']:.4f} "
            f"pred={example_result['predicted_size']} gold={example_result['gold_size']} saved"
        )

    return summarize_example_results(
        example_results,
        dense_weights=dense_weights,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        bm25_epsilon=bm25_epsilon,
        batch_size=batch_size,
        discovery_retry_limit=discovery_retry_limit,
        rrf_k=rrf_k,
    )


def normalize_search_definition(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Support either a raw search space or a JSON object with search controls."""
    if "search_space" in raw:
        unknown_keys = set(raw) - SEARCH_CONTROL_KEYS
        if unknown_keys:
            raise ValueError(
                "Unknown top-level search config field(s): "
                + ", ".join(sorted(unknown_keys))
            )
        search_space = raw["search_space"]
    else:
        search_space = {
            key: value
            for key, value in raw.items()
            if key not in SEARCH_CONTROL_KEYS
        }

    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("Grid config must define a non-empty search space.")

    search_options = {
        key: raw[key]
        for key in ["search_method", "max_configs", "seed"]
        if key in raw and raw[key] is not None
    }
    return search_space, search_options


def load_grid_configs(path: str, base_config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with open(path, "r") as f:
        raw_search_definition = json.load(f)

    search_space, search_options = normalize_search_definition(raw_search_definition)
    search_space = search_space.copy()
    dense_weight_configs = search_space.pop("dense_weights", [base_config["dense_weights"]])
    if isinstance(dense_weight_configs, dict):
        dense_weight_configs = [dense_weight_configs]
    if not isinstance(dense_weight_configs, list) or not dense_weight_configs:
        raise ValueError("Grid field `dense_weights` must be a dict or non-empty list.")

    scalar_keys = sorted(search_space.keys())
    scalar_value_lists = []
    for key in scalar_keys:
        values = search_space[key]
        if not isinstance(values, list):
            values = [values]
        if not values:
            raise ValueError(f"Grid field `{key}` must be non-empty.")
        scalar_value_lists.append(values)

    configs = []
    for dense_weights in dense_weight_configs:
        for values in itertools.product(*scalar_value_lists) if scalar_keys else [()]:
            config = base_config.copy()
            config["dense_weights"] = dense_weights
            for key, value in zip(scalar_keys, values):
                config[key] = value
            configs.append(config)
    return configs, search_options


def choose_configs(
    configs: list[dict[str, Any]],
    *,
    method: str,
    max_configs: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if method not in {"grid", "random"}:
        raise ValueError("--search-method must be either `grid` or `random`.")
    if max_configs is not None and max_configs < 1:
        raise ValueError("--max-configs must be positive when provided.")
    if max_configs is None or max_configs >= len(configs):
        return configs
    if method == "grid":
        return configs[:max_configs]

    rng = random.Random(seed)
    indexed_configs = list(enumerate(configs))
    rng.shuffle(indexed_configs)
    selected = sorted(indexed_configs[:max_configs], key=lambda item: item[0])
    return [config for _, config in selected]


def write_results(rows: list[dict[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sorted_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (row["discovery_accuracy_mean"], row["discovery_f1_mean"]),
        reverse=True,
    )


def checkpoint_fingerprint(
    args: argparse.Namespace,
    configs: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> str:
    example_signature = [
        {
            "question": example.get("question"),
            "gold_endpoints": gold_endpoints_for_example(
                example,
                use_service_endpoints=args.use_service_endpoints,
            ),
        }
        for example in examples
    ]
    return stable_hash(
        {
            "benchmark": str(Path(args.benchmark).resolve()),
            "split": args.split,
            "limit": args.limit,
            "use_service_endpoints": args.use_service_endpoints,
            "configs": configs,
            "examples": example_signature,
        }
    )


def load_checkpoint(checkpoint_path: str, fingerprint: str, fresh: bool) -> dict[str, Any] | None:
    if fresh:
        return None

    path = Path(checkpoint_path)
    if not path.exists():
        return None

    try:
        with open(path, "r") as f:
            checkpoint = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Warning: could not parse checkpoint {checkpoint_path}: {exc}. Starting fresh.")
        return None

    if checkpoint.get("schema_version") != 1:
        print(f"Warning: ignoring checkpoint {checkpoint_path} with unsupported schema.")
        return None

    if checkpoint.get("fingerprint") != fingerprint:
        print(f"Warning: checkpoint {checkpoint_path} does not match this run. Starting fresh.")
        return None

    return checkpoint


def completed_rows_from_checkpoint(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    runs = checkpoint.get("runs", {})
    rows = []
    for run_key in sorted(runs, key=lambda value: int(value)):
        run_state = runs[run_key]
        if run_state.get("status") == "completed" and run_state.get("summary"):
            rows.append(run_state["summary"])
    return rows


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
        "--search-method",
        default=None,
        choices=["grid", "random"],
        help="How to select configs from the JSON search space. Overrides search_method in the JSON.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Evaluate at most this many configs. Overrides max_configs in the JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible random config selection. Overrides seed in the JSON.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(BENCHMARK_DATA_PATH), "..", "discovery_ablation_results.csv"),
        help="CSV file for ablation results.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="JSON checkpoint for resumable progress. Defaults to <output-stem>.checkpoint.json.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing checkpoint and start this ablation from scratch.",
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

    search_options: dict[str, Any] = {}
    if args.grid_config:
        all_configs, search_options = load_grid_configs(args.grid_config, base_config=base_config)
    else:
        all_configs = [base_config]

    search_method = args.search_method or search_options.get("search_method", "grid")
    max_configs = args.max_configs
    if max_configs is None:
        max_configs = search_options.get("max_configs")
    if max_configs is not None:
        max_configs = int(max_configs)
    seed = args.seed
    if seed is None:
        seed = int(search_options.get("seed", DEFAULT_RANDOM_SEED))

    configs = choose_configs(
        all_configs,
        method=search_method,
        max_configs=max_configs,
        seed=seed,
    )

    checkpoint_path = args.checkpoint or default_checkpoint_path(args.output)
    fingerprint = checkpoint_fingerprint(args, configs, examples)
    checkpoint = load_checkpoint(checkpoint_path, fingerprint=fingerprint, fresh=args.fresh)
    if checkpoint is None:
        checkpoint = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "benchmark": str(Path(args.benchmark).resolve()),
            "split": args.split,
            "limit": args.limit,
            "use_service_endpoints": args.use_service_endpoints,
            "n_examples": len(examples),
            "search_method": search_method,
            "seed": seed,
            "max_configs": max_configs,
            "n_total_configs": len(all_configs),
            "n_runs": len(configs),
            "runs": {},
        }
        atomic_write_json(checkpoint_path, checkpoint)
        print(f"Checkpointing progress to {checkpoint_path}")
    else:
        print(f"Resuming from checkpoint {checkpoint_path}")

    existing_rows = completed_rows_from_checkpoint(checkpoint)
    if existing_rows:
        write_results(sorted_result_rows(existing_rows), args.output)

    print(
        f"Selected {len(configs)}/{len(all_configs)} discovery config(s) "
        f"with search_method={search_method}, seed={seed}, max_configs={max_configs}"
    )

    for run_idx, config in enumerate(configs, start=1):
        run_key = str(run_idx)
        run_state = checkpoint["runs"].setdefault(
            run_key,
            {
                "config": config,
                "status": "pending",
                "examples": [],
                "summary": None,
            },
        )
        run_state["config"] = config

        if run_state.get("status") == "completed" and run_state.get("summary"):
            row = run_state["summary"]
            print(
                f"\n=== Discovery Ablation Run {run_idx}/{len(configs)} already completed ==="
            )
            print(
                f"Run {run_idx}: accuracy={row['discovery_accuracy_mean']:.4f}, "
                f"f1={row['discovery_f1_mean']:.4f}"
            )
            continue

        print(f"\n=== Discovery Ablation Run {run_idx}/{len(configs)} ===")
        print(json.dumps(config, indent=2))

        run_state["status"] = "in_progress"
        atomic_write_json(checkpoint_path, checkpoint)

        def save_example_checkpoint(example_result: dict[str, Any]) -> None:
            results_by_index = {
                row["example_index"]: row
                for row in run_state.get("examples", [])
            }
            results_by_index[example_result["example_index"]] = example_result
            run_state["examples"] = [
                results_by_index[index]
                for index in sorted(results_by_index)
            ]
            checkpoint["last_position"] = {
                "run": run_idx,
                "example_index": example_result["example_index"],
                "completed_examples_in_run": len(run_state["examples"]),
            }
            atomic_write_json(checkpoint_path, checkpoint)

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
            saved_example_results=run_state.get("examples", []),
            on_example_saved=save_example_checkpoint,
        )
        row["run"] = run_idx
        row["split"] = args.split
        row["service_gold"] = args.use_service_endpoints

        run_state["summary"] = row
        run_state["status"] = "completed"
        checkpoint["last_position"] = {
            "run": run_idx,
            "example_index": len(examples) - 1,
            "completed_examples_in_run": len(run_state.get("examples", [])),
        }
        atomic_write_json(checkpoint_path, checkpoint)

        completed_rows = completed_rows_from_checkpoint(checkpoint)
        write_results(sorted_result_rows(completed_rows), args.output)

        print(
            f"Run {run_idx}: accuracy={row['discovery_accuracy_mean']:.4f}, "
            f"f1={row['discovery_f1_mean']:.4f}"
        )

    rows = sorted_result_rows(completed_rows_from_checkpoint(checkpoint))
    if rows:
        write_results(rows, args.output)

    print(f"\nSaved {len(rows)} run(s) to {args.output}")
    print(f"Checkpoint saved at {checkpoint_path}")
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
