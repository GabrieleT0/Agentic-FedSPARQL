"""Search full-pipeline hyperparameters with a reproducible protocol.

This script is intended for paper-quality model selection:
1. Define a fixed hyperparameter grid before running experiments.
2. Select a deterministic development subset with a fixed random seed.
3. Run every configuration on the same examples.
4. Save raw benchmark JSON, logs, a resumable checkpoint, and a ranked CSV.
5. Rank by answer metrics while reporting cost/retry metrics and bootstrap CIs.

Examples:
  python3 src/search_pipeline_hyperparameters.py --split dev --limit 100
  python3 src/search_pipeline_hyperparameters.py --grid-config data/pipeline_hyperparameter_grid.json
  OLLAMA_MODEL=llama3.1 python3 src/search_pipeline_hyperparameters.py --limit 25 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from metrics import discovery_accuracy, execution_accuracy, f1_score
from recalculate_discovery_metrics import _load_results, discovery_f1, extract_service_endpoints


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "SPIDER4FedSPARQL" / "class-sharding" / "SPIDER4FedSPARQL_benchmark.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "pipeline_hyperparameter_search"

HYPERPARAMETER_FIELDS = [
    "max_retries",
    "discovery_retry_limit",
    "query_builder_retries",
    "schema_summary_retry_limit",
    "batch_size",
]

DEFAULT_HYPERPARAMETERS = {
    "max_retries": 3,
    "discovery_retry_limit": 2,
    "query_builder_retries": 3,
    "schema_summary_retry_limit": 2,
    "batch_size": 50,
}

DEFAULT_SEARCH_SPACE = {
    "max_retries": [1, 3, 5],
    "discovery_retry_limit": [1, 2, 3],
    "query_builder_retries": [1, 2, 3],
    "schema_summary_retry_limit": [1, 2, 3],
    "batch_size": [25, 50, 100],
}

ALIASES = {
    "MAX_RETRIES": "max_retries",
    "max_retry": "max_retries",
    "DISCOVERY_RETRY": "discovery_retry_limit",
    "DISCOVERY_RETYRU": "discovery_retry_limit",
    "DISCOVERY_RETRY_LIMIT": "discovery_retry_limit",
    "DISCOVERY_RETYRU_LIMIT": "discovery_retry_limit",
    "discovery_retry": "discovery_retry_limit",
    "discovery_retyru": "discovery_retry_limit",
    "QUERY_BUILDER_RETRIES": "query_builder_retries",
    "QUERY_BUILDER_RETRY": "query_builder_retries",
    "query_builder_retry": "query_builder_retries",
    "SCHEMA_SUMMARY_RETRY_LIMIT": "schema_summary_retry_limit",
    "SCHEMA_SUMMARY_RETRY": "schema_summary_retry_limit",
    "schema_retry_limit": "schema_summary_retry_limit",
    "BATCH_SIZE": "batch_size",
}

ANSWER_METRICS = {"execution_accuracy", "f1_score"}
SUMMARY_METRICS = [
    "execution_accuracy",
    "f1_score",
    "discovery_accuracy",
    "discovery_f1",
    "refinement_attempts",
    "discovery_internal_retries",
    "schema_summary_retries",
    "query_builder_internal_retries",
]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_key(key: str) -> str:
    return ALIASES.get(key, key)


def normalize_config(config: dict[str, Any]) -> dict[str, int]:
    normalized = {}
    for key, value in config.items():
        normalized_key = normalize_key(key)
        if normalized_key not in HYPERPARAMETER_FIELDS:
            raise ValueError(
                f"Unknown hyperparameter `{key}`. Supported fields: {', '.join(HYPERPARAMETER_FIELDS)}"
            )
        normalized[normalized_key] = int(value)

    for field in HYPERPARAMETER_FIELDS:
        normalized.setdefault(field, DEFAULT_HYPERPARAMETERS[field])

    return normalized


def load_search_space(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        raw = DEFAULT_SEARCH_SPACE
    else:
        with path.open("r") as f:
            raw = json.load(f)

    search_space: dict[str, list[int]] = {}
    for key, values in raw.items():
        normalized_key = normalize_key(key)
        if normalized_key not in HYPERPARAMETER_FIELDS:
            raise ValueError(
                f"Unknown grid field `{key}`. Supported fields: {', '.join(HYPERPARAMETER_FIELDS)}"
            )
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid field `{key}` must be a non-empty list.")
        search_space[normalized_key] = [int(value) for value in values]

    for field in HYPERPARAMETER_FIELDS:
        search_space.setdefault(field, [DEFAULT_HYPERPARAMETERS[field]])

    return search_space


def build_grid(search_space: dict[str, list[int]]) -> list[dict[str, int]]:
    value_lists = [search_space[field] for field in HYPERPARAMETER_FIELDS]
    return [
        dict(zip(HYPERPARAMETER_FIELDS, values))
        for values in itertools.product(*value_lists)
    ]


def choose_configs(
    configs: list[dict[str, int]],
    *,
    method: str,
    max_configs: int | None,
    seed: int,
) -> list[dict[str, int]]:
    if max_configs is None or max_configs >= len(configs):
        return configs

    if method == "grid":
        return configs[:max_configs]

    rng = random.Random(seed)
    indexed = list(enumerate(configs))
    rng.shuffle(indexed)
    selected = sorted(indexed[:max_configs], key=lambda item: item[0])
    return [config for _, config in selected]


def parse_run_ids(raw: str | None) -> set[int] | None:
    if raw is None:
        return None

    selected: set[int] = set()
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue

        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if start > end:
                raise ValueError(f"Invalid run range `{token}`: start must be <= end.")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))

    if any(run_idx < 1 for run_idx in selected):
        raise ValueError("Run ids are 1-based and must be positive.")

    return selected


def select_indexed_configs(
    configs: list[dict[str, int]],
    *,
    run_start: int | None,
    run_end: int | None,
    run_ids: str | None,
) -> list[tuple[int, dict[str, int]]]:
    if run_start is not None and run_start < 1:
        raise ValueError("--run-start must be >= 1.")
    if run_end is not None and run_end < 1:
        raise ValueError("--run-end must be >= 1.")
    if run_start is not None and run_end is not None and run_start > run_end:
        raise ValueError("--run-start must be <= --run-end.")

    explicit_ids = parse_run_ids(run_ids)
    indexed_configs = list(enumerate(configs, start=1))

    selected = []
    for run_idx, config in indexed_configs:
        if run_start is not None and run_idx < run_start:
            continue
        if run_end is not None and run_idx > run_end:
            continue
        if explicit_ids is not None and run_idx not in explicit_ids:
            continue
        selected.append((run_idx, config))

    if not selected:
        raise ValueError("No configurations matched the requested run selection.")

    return selected


def load_examples(
    benchmark_path: Path,
    *,
    split: str,
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    with benchmark_path.open("r") as f:
        data = json.load(f)

    examples = [
        example
        for example in data
        if example.get("validation", {}).get("valid")
        and example.get("validation", {}).get("original_count", 0) > 0
        and (split == "all" or example.get("split") == split)
    ]

    if not examples:
        raise ValueError("No examples matched the requested split and validation filters.")

    if limit is not None:
        rng = random.Random(seed)
        examples = list(examples)
        rng.shuffle(examples)
        examples = examples[:limit]

    return examples


def example_signature(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "question": example.get("question"),
            "split": example.get("split"),
            "gold_sparql_hash": stable_hash(example.get("federated_sparql", "")),
        }
        for example in examples
    ]


def write_selected_benchmark(path: Path, examples: list[dict[str, Any]]) -> None:
    atomic_write_json(path, examples)


def has_empty_gold_answer_placeholder(result: dict[str, Any]) -> bool:
    return result.get("gold_answers") == [{}]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = mean(values)
    assert m is not None
    variance = sum((value - m) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int,
    n_bootstrap: int,
    confidence: float,
) -> tuple[float | None, float | None]:
    if not values or n_bootstrap <= 0:
        return None, None
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    sample_size = len(values)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(sample_size)] for _ in range(sample_size)]
        bootstrap_means.append(sum(sample) / sample_size)

    bootstrap_means.sort()
    alpha = 1.0 - confidence
    return (
        percentile(bootstrap_means, alpha / 2),
        percentile(bootstrap_means, 1 - alpha / 2),
    )


def numeric_values(
    results: list[dict[str, Any]],
    metric: str,
    *,
    recompute_stored_metrics: bool,
) -> list[float]:
    values = []
    for result in results:
        if metric in ANSWER_METRICS and has_empty_gold_answer_placeholder(result):
            continue

        if (
            recompute_stored_metrics
            and
            metric in ANSWER_METRICS
            and "predicted_answers" in result
            and "gold_answers" in result
        ):
            if metric == "execution_accuracy":
                value = execution_accuracy(result["predicted_answers"], result["gold_answers"])
            else:
                value = f1_score(result["predicted_answers"], result["gold_answers"])
        elif (
            recompute_stored_metrics
            and
            metric in {"discovery_accuracy", "discovery_f1"}
            and "predicted_endpoints" in result
            and ("gold_sparql" in result or "gold:sparql" in result or "gold_endpoints" in result)
        ):
            gold_endpoints = (
                extract_service_endpoints(result.get("gold_sparql") or result.get("gold:sparql") or "")
                or result.get("gold_endpoints", [])
            )
            if metric == "discovery_accuracy":
                value = discovery_accuracy(result["predicted_endpoints"], gold_endpoints)
            else:
                value = discovery_f1(result["predicted_endpoints"], gold_endpoints)
        else:
            if metric not in result:
                continue
            value = result[metric]

        if value is None:
            continue
        values.append(float(value))
    return values


def summarize_run(
    result_path: Path,
    *,
    config: dict[str, int],
    run: int,
    bootstrap_seed: int,
    n_bootstrap: int,
    confidence: float,
    recompute_stored_metrics: bool = True,
) -> dict[str, Any]:
    _, results = _load_results(str(result_path))
    summary: dict[str, Any] = {
        "run": run,
        **config,
        "n_questions": len(results),
        "n_answer_evaluable": sum(
            1 for result in results
            if not has_empty_gold_answer_placeholder(result)
            and "execution_accuracy" in result
        ),
        "n_empty_gold_answer_placeholders": sum(
            1 for result in results
            if has_empty_gold_answer_placeholder(result)
        ),
    }

    for metric in SUMMARY_METRICS:
        values = numeric_values(
            results,
            metric,
            recompute_stored_metrics=recompute_stored_metrics,
        )
        metric_mean = mean(values)
        metric_std = std(values)
        summary[f"{metric}_mean"] = round(metric_mean, 4) if metric_mean is not None else ""
        summary[f"{metric}_std"] = round(metric_std, 4) if metric_std is not None else ""
        summary[f"{metric}_n"] = len(values)

        if metric in ANSWER_METRICS:
            low, high = bootstrap_mean_ci(
                values,
                seed=bootstrap_seed + run * 1009 + stable_int(metric),
                n_bootstrap=n_bootstrap,
                confidence=confidence,
            )
            summary[f"{metric}_ci_low"] = round(low, 4) if low is not None else ""
            summary[f"{metric}_ci_high"] = round(high, 4) if high is not None else ""

    retry_cost = sum(
        float(summary.get(f"{metric}_mean") or 0.0)
        for metric in [
            "refinement_attempts",
            "discovery_internal_retries",
            "schema_summary_retries",
            "query_builder_internal_retries",
        ]
    )
    summary["retry_cost_proxy"] = round(retry_cost, 4)
    return summary


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def value(row: dict[str, Any], key: str, default: float) -> float:
        raw = row.get(key)
        if raw == "" or raw is None:
            return default
        return float(raw)

    return sorted(
        rows,
        key=lambda row: (
            value(row, "execution_accuracy_mean", float("-inf")),
            value(row, "f1_score_mean", float("-inf")),
            value(row, "discovery_f1_mean", float("-inf")),
            -value(row, "retry_cost_proxy", float("inf")),
            -int(row["max_retries"]),
            -int(row["discovery_retry_limit"]),
            -int(row["query_builder_retries"]),
            -int(row["schema_summary_retry_limit"]),
            -int(row["batch_size"]),
        ),
        reverse=True,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "run",
        *HYPERPARAMETER_FIELDS,
        "n_questions",
        "n_answer_evaluable",
        "n_empty_gold_answer_placeholders",
        "execution_accuracy_mean",
        "execution_accuracy_std",
        "execution_accuracy_ci_low",
        "execution_accuracy_ci_high",
        "execution_accuracy_n",
        "f1_score_mean",
        "f1_score_std",
        "f1_score_ci_low",
        "f1_score_ci_high",
        "f1_score_n",
        "discovery_accuracy_mean",
        "discovery_accuracy_std",
        "discovery_accuracy_n",
        "discovery_f1_mean",
        "discovery_f1_std",
        "discovery_f1_n",
        "refinement_attempts_mean",
        "discovery_internal_retries_mean",
        "schema_summary_retries_mean",
        "query_builder_internal_retries_mean",
        "retry_cost_proxy",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(sort_rows(rows), start=1):
            writer.writerow({"rank": rank, **row})


RUN_FILE_PATTERN = re.compile(r"run_(\d+)\.json$")


def run_index_from_path(path: Path) -> int | None:
    match = RUN_FILE_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def load_protocol_configs(path: Path) -> dict[int, dict[str, int]]:
    if not path.exists():
        return {}

    with path.open("r") as f:
        protocol = json.load(f)

    configs = protocol.get("configs", [])
    return {
        idx: normalize_config(config)
        for idx, config in enumerate(configs, start=1)
    }


def load_checkpoint_unchecked(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "runs": {},
        }

    with path.open("r") as f:
        checkpoint = json.load(f)

    if checkpoint.get("schema_version") != 1:
        raise ValueError(f"Unsupported checkpoint schema in {path}.")

    checkpoint.setdefault("runs", {})
    return checkpoint


def config_from_result_file(path: Path) -> dict[str, int] | None:
    envelope, _ = _load_results(str(path))
    config = envelope.get("config") if isinstance(envelope, dict) else None
    if not config:
        return None

    return normalize_config(
        {
            field: config[field]
            for field in HYPERPARAMETER_FIELDS
            if field in config
        }
    )


def config_for_existing_run(
    *,
    run_idx: int,
    run_state: dict[str, Any],
    result_path: Path,
    protocol_configs: dict[int, dict[str, int]],
) -> dict[str, int]:
    if run_state.get("config"):
        return normalize_config(run_state["config"])

    if run_idx in protocol_configs:
        return protocol_configs[run_idx]

    config = config_from_result_file(result_path)
    if config is not None:
        return config

    raise ValueError(f"Could not determine hyperparameter config for run {run_idx}.")


def existing_run_paths(output_dir: Path, checkpoint: dict[str, Any]) -> dict[int, Path]:
    paths: dict[int, Path] = {}

    for run_key, run_state in checkpoint.get("runs", {}).items():
        run_idx = int(run_key)
        state_path = run_state.get("result_path")
        if state_path:
            path = Path(state_path)
            if path.exists():
                paths[run_idx] = path

    for path in sorted((output_dir / "runs").glob("run_*.json")):
        run_idx = run_index_from_path(path)
        if run_idx is not None:
            paths.setdefault(run_idx, path)

    return dict(sorted(paths.items()))


def recompute_summaries(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    results_csv_path: Path,
    protocol_path: Path,
    bootstrap_seed: int,
    n_bootstrap: int,
    confidence: float,
    recompute_stored_metrics: bool,
) -> list[dict[str, Any]]:
    checkpoint = load_checkpoint_unchecked(checkpoint_path)
    protocol_configs = load_protocol_configs(protocol_path)
    run_paths = existing_run_paths(output_dir, checkpoint)

    if not run_paths:
        raise ValueError(f"No existing run JSON files found under {output_dir / 'runs'}.")

    rows = []
    for run_idx, result_path in run_paths.items():
        run_key = str(run_idx)
        run_state = checkpoint["runs"].setdefault(run_key, {})
        run_state.setdefault("result_path", str(result_path))
        run_state.setdefault("log_path", str(output_dir / "logs" / f"run_{run_idx:04d}.log"))

        try:
            config = config_for_existing_run(
                run_idx=run_idx,
                run_state=run_state,
                result_path=result_path,
                protocol_configs=protocol_configs,
            )
            summary = summarize_run(
                result_path,
                config=config,
                run=run_idx,
                bootstrap_seed=bootstrap_seed,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                recompute_stored_metrics=recompute_stored_metrics,
            )
        except Exception as exc:
            run_state["status"] = "failed_to_summarize"
            run_state["summary_error"] = str(exc)
            print(f"Run {run_idx} could not be summarized: {exc}")
            continue

        run_state["config"] = config
        run_state["status"] = "completed"
        run_state["summary"] = summary
        run_state.pop("summary_error", None)
        rows.append(summary)
        print(
            f"Run {run_idx}: execution_accuracy={summary['execution_accuracy_mean']} "
            f"f1={summary['f1_score_mean']} retry_cost={summary['retry_cost_proxy']}"
        )

    checkpoint["n_completed_summaries"] = len(rows)
    atomic_write_json(checkpoint_path, checkpoint)
    write_csv(results_csv_path, rows)
    return sort_rows(rows)


def load_checkpoint(path: Path, fingerprint: str, fresh: bool) -> dict[str, Any] | None:
    if fresh or not path.exists():
        return None

    with path.open("r") as f:
        checkpoint = json.load(f)

    if checkpoint.get("schema_version") != 1:
        print(f"Warning: ignoring unsupported checkpoint schema in {path}.")
        return None

    if checkpoint.get("fingerprint") != fingerprint:
        print(f"Warning: checkpoint {path} does not match this search protocol. Starting fresh.")
        return None

    return checkpoint


def run_configuration(
    *,
    config: dict[str, int],
    selected_benchmark_path: Path,
    result_path: Path,
    log_path: Path,
) -> int:
    env = os.environ.copy()
    env.update(
        {
            "MODE": "benchmark",
            "BENCHMARK_DATA_PATH": str(selected_benchmark_path),
            "BENCHMARK_RESULT_PATH": str(result_path),
            "MAX_RETRIES": str(config["max_retries"]),
            "DISCOVERY_RETRY_LIMIT": str(config["discovery_retry_limit"]),
            "QUERY_BUILDER_RETRIES": str(config["query_builder_retries"]),
            "SCHEMA_SUMMARY_RETRY_LIMIT": str(config["schema_summary_retry_limit"]),
            "BATCH_SIZE": str(config["batch_size"]),
            "PYTHONUNBUFFERED": "1",
        }
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        process = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "main.py")],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode


def protocol_text(args: argparse.Namespace, n_configs: int, n_examples: int) -> str:
    return (
        "Hyperparameters were selected by a pre-specified deterministic search over "
        f"{n_configs} configuration(s) on the {args.split} split "
        f"({n_examples} validated examples"
        f"{', sampled with seed ' + str(args.seed) if args.limit is not None else ''}). "
        "Each configuration was evaluated on the same examples. The primary selection "
        "criterion was mean execution accuracy, with mean answer F1 and discovery F1 "
        "as tie-breakers, followed by a lower retry-cost proxy. Rows whose "
        "gold_answers field was the placeholder [{}] were excluded from answer-metric "
        "denominators. Bootstrap confidence intervals were computed over examples "
        f"using {args.bootstrap_samples} resamples."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible full-pipeline hyperparameter search."
    )
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH), help="Benchmark JSON path.")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "all"], help="Split used for selection.")
    parser.add_argument("--limit", type=int, default=None, help="Optional deterministic sample size.")
    parser.add_argument("--seed", type=int, default=20240624, help="Seed for sampling and randomized search.")
    parser.add_argument("--grid-config", default=None, help="JSON file with pre-specified hyperparameter values.")
    parser.add_argument("--search-method", default="grid", choices=["grid", "random"], help="Grid order or random subset.")
    parser.add_argument("--max-configs", type=int, default=None, help="Evaluate at most this many configurations.")
    parser.add_argument("--run-start", type=int, default=None, help="First 1-based run index to execute.")
    parser.add_argument("--run-end", type=int, default=None, help="Last 1-based run index to execute.")
    parser.add_argument(
        "--run-ids",
        default=None,
        help="Comma-separated 1-based run ids or ranges to execute, e.g. `1,4,7-10`.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for raw runs and summaries.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint JSON path. Defaults to <output-dir>/checkpoint.json.")
    parser.add_argument("--results-csv", default=None, help="Ranking CSV path. Defaults to <output-dir>/results.csv.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap resamples for answer-metric CIs.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Bootstrap CI confidence level.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoint metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Write protocol files and print configs without running.")
    parser.add_argument(
        "--recompute-summaries",
        action="store_true",
        help="Rebuild checkpoint summaries and results CSV from existing run JSON files, then exit without rerunning.",
    )
    parser.add_argument(
        "--use-stored-metrics",
        action="store_true",
        help="When recomputing summaries, trust metric fields already stored in run JSON files instead of recalculating them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else output_dir / "checkpoint.json"
    results_csv_path = Path(args.results_csv).resolve() if args.results_csv else output_dir / "results.csv"
    selected_benchmark_path = output_dir / "selected_benchmark.json"
    protocol_path = output_dir / "protocol.json"

    if args.recompute_summaries:
        ranked = recompute_summaries(
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            results_csv_path=results_csv_path,
            protocol_path=protocol_path,
            bootstrap_seed=args.seed,
            n_bootstrap=args.bootstrap_samples,
            confidence=args.confidence,
            recompute_stored_metrics=not args.use_stored_metrics,
        )
        print(f"\nRecomputed {len(ranked)} summarized run(s) to {results_csv_path}")
        if ranked:
            best = ranked[0]
            print("Best configuration:")
            print(json.dumps({field: best[field] for field in HYPERPARAMETER_FIELDS}, indent=2))
            print(
                f"execution_accuracy={best['execution_accuracy_mean']} "
                f"95% CI=[{best['execution_accuracy_ci_low']}, {best['execution_accuracy_ci_high']}], "
                f"f1={best['f1_score_mean']}, retry_cost={best['retry_cost_proxy']}"
            )
        return

    search_space = load_search_space(Path(args.grid_config).resolve() if args.grid_config else None)
    all_configs = build_grid(search_space)
    configs = choose_configs(
        all_configs,
        method=args.search_method,
        max_configs=args.max_configs,
        seed=args.seed,
    )
    indexed_configs = select_indexed_configs(
        configs,
        run_start=args.run_start,
        run_end=args.run_end,
        run_ids=args.run_ids,
    )
    examples = load_examples(benchmark_path, split=args.split, limit=args.limit, seed=args.seed)
    write_selected_benchmark(selected_benchmark_path, examples)

    fingerprint = stable_hash(
        {
            "benchmark_path": str(benchmark_path),
            "split": args.split,
            "limit": args.limit,
            "seed": args.seed,
            "search_method": args.search_method,
            "search_space": search_space,
            "configs": configs,
            "examples": example_signature(examples),
        }
    )

    checkpoint = load_checkpoint(checkpoint_path, fingerprint=fingerprint, fresh=args.fresh)
    if checkpoint is None:
        checkpoint = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "benchmark_path": str(benchmark_path),
            "selected_benchmark_path": str(selected_benchmark_path),
            "split": args.split,
            "limit": args.limit,
            "seed": args.seed,
            "search_method": args.search_method,
        "search_space": search_space,
        "n_examples": len(examples),
        "n_configs": len(configs),
        "selected_run_indexes": [run_idx for run_idx, _ in indexed_configs],
        "runs": {},
    }
        atomic_write_json(checkpoint_path, checkpoint)
        print(f"Created checkpoint: {checkpoint_path}")
    else:
        print(f"Resuming checkpoint: {checkpoint_path}")

    protocol = {
        "protocol": protocol_text(args, n_configs=len(configs), n_examples=len(examples)),
        "benchmark_path": str(benchmark_path),
        "selected_benchmark_path": str(selected_benchmark_path),
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed,
        "search_method": args.search_method,
        "search_space": search_space,
        "configs": configs,
        "primary_metric": "execution_accuracy_mean",
        "tie_breakers": ["f1_score_mean", "discovery_f1_mean", "retry_cost_proxy (lower is better)"],
        "placeholder_rule": "Exclude rows with gold_answers == [{}] from answer metrics.",
        "bootstrap_samples": args.bootstrap_samples,
        "confidence": args.confidence,
    }
    atomic_write_json(protocol_path, protocol)

    print(protocol["protocol"])
    print(f"Selected benchmark written to {selected_benchmark_path}")
    print(f"Protocol written to {protocol_path}")
    print(f"Results CSV will be written to {results_csv_path}")
    print(
        f"This invocation will execute {len(indexed_configs)}/{len(configs)} config(s): "
        f"{', '.join(str(run_idx) for run_idx, _ in indexed_configs)}"
    )

    if args.dry_run:
        print("\nDry run configurations:")
        for run_idx, config in indexed_configs:
            print(f"  run={run_idx}: {config}")
        return

    rows: list[dict[str, Any]] = []
    for run_idx, config in indexed_configs:
        run_key = str(run_idx)
        result_path = output_dir / "runs" / f"run_{run_idx:04d}.json"
        log_path = output_dir / "logs" / f"run_{run_idx:04d}.log"
        run_state = checkpoint["runs"].setdefault(
            run_key,
            {
                "config": config,
                "status": "pending",
                "result_path": str(result_path),
                "log_path": str(log_path),
            },
        )
        run_state["config"] = config
        run_state["result_path"] = str(result_path)
        run_state["log_path"] = str(log_path)

        if run_state.get("status") == "completed" and result_path.exists():
            print(f"\n=== Run {run_idx}/{len(configs)} already completed ===")
        else:
            print(f"\n=== Run {run_idx}/{len(configs)} ===")
            print(json.dumps(config, indent=2))
            run_state["status"] = "in_progress"
            atomic_write_json(checkpoint_path, checkpoint)

            returncode = run_configuration(
                config=config,
                selected_benchmark_path=selected_benchmark_path,
                result_path=result_path,
                log_path=log_path,
            )
            run_state["returncode"] = returncode

            if returncode != 0:
                run_state["status"] = "failed"
                atomic_write_json(checkpoint_path, checkpoint)
                print(f"Run {run_idx} failed with exit code {returncode}. See {log_path}")
                continue

            run_state["status"] = "completed"
            atomic_write_json(checkpoint_path, checkpoint)

        try:
            summary = summarize_run(
                result_path,
                config=config,
                run=run_idx,
                bootstrap_seed=args.seed,
                n_bootstrap=args.bootstrap_samples,
                confidence=args.confidence,
            )
        except Exception as exc:
            run_state["status"] = "failed_to_summarize"
            run_state["summary_error"] = str(exc)
            atomic_write_json(checkpoint_path, checkpoint)
            print(f"Run {run_idx} could not be summarized: {exc}")
            continue

        run_state["summary"] = summary
        checkpoint["runs"][run_key] = run_state
        atomic_write_json(checkpoint_path, checkpoint)
        rows.append(summary)
        write_csv(results_csv_path, rows)

        print(
            f"Run {run_idx}: execution_accuracy={summary['execution_accuracy_mean']} "
            f"f1={summary['f1_score_mean']} retry_cost={summary['retry_cost_proxy']}"
        )

    completed_summaries = [
        state["summary"]
        for state in checkpoint.get("runs", {}).values()
        if state.get("status") == "completed" and state.get("summary")
    ]
    write_csv(results_csv_path, completed_summaries)

    ranked = sort_rows(completed_summaries)
    print(f"\nSaved {len(ranked)} completed run(s) to {results_csv_path}")
    if ranked:
        best = ranked[0]
        print("Best configuration:")
        print(json.dumps({field: best[field] for field in HYPERPARAMETER_FIELDS}, indent=2))
        print(
            f"execution_accuracy={best['execution_accuracy_mean']} "
            f"95% CI=[{best['execution_accuracy_ci_low']}, {best['execution_accuracy_ci_high']}], "
            f"f1={best['f1_score_mean']}, retry_cost={best['retry_cost_proxy']}"
        )


if __name__ == "__main__":
    main()
