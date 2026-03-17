import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

from federation_query_tools import analyze_query_classes, rewrite_as_federated
from kg_sharding_tools import analyze_kg, shard_by_class
from real_federated_tools import LocalFederationServer, validate_rewrite_real_federated


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_ROOT_CANDIDATES = [
    ROOT_DIR / "data" / "original_SPIDER4SPARQL",
    ROOT_DIR / "data" / "original_SPIEDER4SPARQL",
]
OUTPUT_ROOT = ROOT_DIR / "data" / "federated_output"
SPLITS = ("dev", "train")
MAX_EXAMPLES_PER_SPLIT = None
KG_NAME_ALIASES = {
    "musical": "music_1",
}
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 18080


def _resolve_dataset_root() -> Path:
    for path in DATASET_ROOT_CANDIDATES:
        if path.exists():
            return path
    candidates = ", ".join(str(p) for p in DATASET_ROOT_CANDIDATES)
    raise FileNotFoundError(f"Could not find SPIDER4SPARQL root. Checked: {candidates}")


def _to_jsonable_shards(shards: dict) -> dict:
    return {str(k): v for k, v in shards.items()}


def _load_split_examples(dataset_root: Path, split: str) -> list:
    split_path = dataset_root / split / f"{split}.json"
    if not split_path.exists():
        return []
    data = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {split_path}")
    return data


def _kg_path_for_example(dataset_root: Path, split: str, kg_name: str) -> Path | None:
    lookup_names = [kg_name]
    if kg_name in KG_NAME_ALIASES:
        lookup_names.append(KG_NAME_ALIASES[kg_name])

    for name in lookup_names:
        preferred = dataset_root / f"materialized_triples_{split}" / f"{name}.ttl"
        if preferred.exists():
            return preferred

        # Fallback across split folders for robustness.
        for candidate_split in SPLITS:
            candidate = dataset_root / f"materialized_triples_{candidate_split}" / f"{name}.ttl"
            if candidate.exists():
                return candidate

    return None


def run_all(
    dataset_root: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
    splits: tuple = SPLITS,
    max_examples_per_split: int | None = MAX_EXAMPLES_PER_SPLIT,
    max_kg_size_mb: int | None = None,
):
    dataset_root = dataset_root or _resolve_dataset_root()
    output_root.mkdir(parents=True, exist_ok=True)

    benchmark = {
        "metadata": {
            "name": "FedSpider-real-federated",
            "description": "Benchmark built from SPIDER4SPARQL with real federated SERVICE validation on local HTTP shard endpoints.",
            "version": "1.0",
            "source_root": str(dataset_root),
            "splits": list(splits),
            "num_domains": 0,
            "num_examples": 0,
            "num_federated_queries": 0,
            "num_single_queries": 0,
            "num_validated_equal": 0,
            "num_failed_examples": 0,
            "num_failed_domains": 0,
            "num_missing_kg_domains": 0,
            "num_skipped_large_kg_domains": 0,
        },
        "domains": [],
        "examples": [],
    }

    seen_domains = set()
    federation_server = LocalFederationServer(host=SERVER_HOST, port=SERVER_PORT)

    try:
        for split in splits:
            split_examples = _load_split_examples(dataset_root, split)
            if max_examples_per_split is not None:
                split_examples = split_examples[:max_examples_per_split]

            if not split_examples:
                continue

            print(f"Processing split={split}, examples={len(split_examples)}")
            grouped = defaultdict(list)
            for ex in split_examples:
                grouped[ex["kg_name"]].append(ex)

            for kg_name, domain_examples in grouped.items():
                kg_path = _kg_path_for_example(dataset_root, split, kg_name)
                if kg_path is None:
                    benchmark["metadata"]["num_missing_kg_domains"] += 1
                    benchmark["metadata"]["num_failed_domains"] += 1
                    benchmark["domains"].append(
                        {
                            "split": split,
                            "name": kg_name,
                            "kg_path": None,
                            "error": f"KG not found for domain {kg_name}",
                        }
                    )
                    continue

                if max_kg_size_mb is not None:
                    kg_size_mb = kg_path.stat().st_size / (1024 * 1024)
                    if kg_size_mb > max_kg_size_mb:
                        benchmark["metadata"]["num_skipped_large_kg_domains"] += 1
                        benchmark["domains"].append(
                            {
                                "split": split,
                                "name": kg_name,
                                "kg_path": str(kg_path),
                                "error": f"Skipped large KG ({kg_size_mb:.1f} MB > limit {max_kg_size_mb} MB)",
                            }
                        )
                        continue

                shards_dir = output_root / "shards" / split / kg_name

                try:
                    analysis = analyze_kg(str(kg_path))
                    shards = shard_by_class(str(kg_path), str(shards_dir), analysis=analysis)
                    shards_jsonable = _to_jsonable_shards(shards)

                    metadata_path = output_root / "shards" / split / kg_name / "shards_metadata.json"
                    metadata_path.write_text(json.dumps(shards_jsonable, indent=2), encoding="utf-8")

                    deployed_shards = federation_server.set_shards(
                        shards_jsonable,
                        dataset_prefix=f"{split}/{kg_name}",
                    )
                except Exception as exc:
                    benchmark["metadata"]["num_failed_domains"] += 1
                    benchmark["domains"].append(
                        {
                            "split": split,
                            "name": kg_name,
                            "kg_path": str(kg_path),
                            "error": str(exc),
                        }
                    )
                    continue

                domain_key = (split, kg_name)
                if domain_key not in seen_domains:
                    benchmark["domains"].append(
                        {
                            "split": split,
                            "name": kg_name,
                            "kg_path": str(kg_path),
                            "shards_metadata": str(metadata_path),
                            "num_shards": len(deployed_shards),
                            "shards": [
                                {
                                    "class": str(cls),
                                    "endpoint": info.get("endpoint_url"),
                                    "triple_count": info.get("triple_count"),
                                    "path": info.get("path"),
                                }
                                for cls, info in deployed_shards.items()
                            ],
                        }
                    )
                    seen_domains.add(domain_key)

                for i, ex in enumerate(domain_examples):
                    question = ex.get("question", "")
                    original_query = ex["query"].replace("\\#", "#")
                    error_message = None

                    try:
                        query_analysis = analyze_query_classes(
                            original_query,
                            analysis["subject_to_class"],
                        )

                        rewritten_query = rewrite_as_federated(
                            original_query,
                            query_analysis,
                            deployed_shards,
                        )

                        if rewritten_query:
                            validation = validate_rewrite_real_federated(
                                original_sparql=original_query,
                                federated_sparql=rewritten_query,
                                full_graph=analysis.get("graph"),
                            )
                            is_valid = bool(validation.get("equivalent", False))
                            benchmark["metadata"]["num_federated_queries"] += 1
                            if is_valid:
                                benchmark["metadata"]["num_validated_equal"] += 1
                        else:
                            validation = {"equivalent": True}
                            is_valid = True
                            benchmark["metadata"]["num_single_queries"] += 1
                    except Exception as exc:
                        query_analysis = {"classes": set()}
                        rewritten_query = None
                        validation = {"equivalent": False}
                        is_valid = False
                        error_message = str(exc)
                        benchmark["metadata"]["num_failed_examples"] += 1

                    query_out = output_root / "queries" / split / kg_name / f"{i:04d}.rq"
                    query_out.parent.mkdir(parents=True, exist_ok=True)
                    if rewritten_query:
                        query_out.write_text(rewritten_query, encoding="utf-8")

                    benchmark["examples"].append(
                        {
                            "id": f"{split}_{kg_name}_{i}",
                            "split": split,
                            "domain": kg_name,
                            "question": question,
                            "original_sparql": original_query,
                            "federated_sparql": rewritten_query,
                            "classes_involved": sorted(query_analysis.get("classes", [])),
                            "num_endpoints": len(query_analysis.get("classes", [])) or 1,
                            "type": "federated" if rewritten_query else "single",
                            "is_valid": is_valid,
                            "federated_query_path": str(query_out) if rewritten_query else None,
                            "error": error_message,
                        }
                    )
                    benchmark["metadata"]["num_examples"] += 1

                # Release domain-level objects as soon as possible.
                del deployed_shards
                del shards
                del analysis
                gc.collect()
    finally:
        federation_server.close()

    benchmark["metadata"]["num_domains"] = len(benchmark["domains"])
    benchmark_out = output_root / "benchmark_spider4sparql_real_federated.json"
    benchmark_out.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")

    print("Completed SPIDER4SPARQL pipeline")
    print(f"Dataset root: {dataset_root}")
    print(f"Output benchmark: {benchmark_out}")
    print(f"Examples: {benchmark['metadata']['num_examples']}")
    print(f"Federated: {benchmark['metadata']['num_federated_queries']}")
    print(f"Validated equal: {benchmark['metadata']['num_validated_equal']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SPIDER4SPARQL real federated pipeline."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to original SPIDER4SPARQL root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Path where outputs are written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=MAX_EXAMPLES_PER_SPLIT,
        help="Limit examples processed per split.",
    )
    parser.add_argument(
        "--max-kg-size-mb",
        type=int,
        default=None,
        help="Skip domains whose KG file size exceeds this threshold (MB).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick validation on dev split with a few examples.",
    )

    args = parser.parse_args()

    splits = tuple(args.splits)
    max_examples = args.max_examples_per_split
    if args.smoke_test:
        splits = ("dev",)
        if max_examples is None:
            max_examples = 3

    run_all(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        splits=splits,
        max_examples_per_split=max_examples,
        max_kg_size_mb=args.max_kg_size_mb,
    )
