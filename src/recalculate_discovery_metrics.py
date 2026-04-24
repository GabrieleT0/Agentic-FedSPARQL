"""Recompute discovery metrics using only endpoints referenced in gold SERVICE clauses.

This script:
1. Extracts endpoint URLs from SERVICE clauses in each gold SPARQL query.
2. Overrides ``gold_endpoints`` with the extracted endpoints.
3. Recomputes ``discovery_accuracy``.
4. Adds ``discovery_f1`` (endpoint-level F1) for discovery.
"""

import argparse
import json
import os
import re
import sys
from typing import Any


SERVICE_PATTERN = re.compile(r"SERVICE\s*(?:SILENT\s*)?<([^>]+)>", re.IGNORECASE)
VALID_JSON_LITERALS = {"true", "false", "null"}


def extract_service_endpoints(sparql_query: str) -> list[str]:
    """Extract unique SERVICE endpoint URLs from a SPARQL query, preserving order."""
    if not sparql_query:
        return []

    seen = set()
    endpoints = []
    for url in SERVICE_PATTERN.findall(sparql_query):
        normalized = url.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            endpoints.append(normalized)
    return endpoints


def discovery_accuracy(predicted_endpoints: list[str], gold_endpoints: list[str]) -> float:
    """Exact-match discovery accuracy between predicted and gold endpoints."""
    predicted_set = set(predicted_endpoints or [])
    gold_set = set(gold_endpoints or [])
    return 1.0 if predicted_set == gold_set else 0.0


def discovery_f1(predicted_endpoints: list[str], gold_endpoints: list[str]) -> float:
    """Endpoint-level F1 between predicted and gold endpoints."""
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _is_escaped(text: str, idx: int) -> bool:
    backslashes = 0
    cursor = idx - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _previous_non_whitespace(text: str, idx: int) -> str | None:
    cursor = idx - 1
    while cursor >= 0 and text[cursor].isspace():
        cursor -= 1
    return text[cursor] if cursor >= 0 else None


def _next_non_whitespace(text: str, idx: int) -> str | None:
    cursor = idx
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return text[cursor] if cursor < len(text) else None


def _line_col(text: str, idx: int) -> tuple[int, int]:
    line = text.count("\n", 0, idx) + 1
    column = idx - text.rfind("\n", 0, idx)
    return line, column


def _repair_invalid_json_literals(text: str) -> tuple[str, list[dict[str, Any]]]:
    in_string = False
    idx = 0
    repairs: list[dict[str, Any]] = []
    chunks: list[str] = []
    last_idx = 0

    while idx < len(text):
        char = text[idx]
        if char == '"' and not _is_escaped(text, idx):
            in_string = not in_string
            idx += 1
            continue

        if in_string or not char.isalpha():
            idx += 1
            continue

        start = idx
        while idx < len(text) and text[idx].isalpha():
            idx += 1
        token = text[start:idx]

        previous = _previous_non_whitespace(text, start)
        following = _next_non_whitespace(text, idx)
        looks_like_value = previous in {":", "[", ","} and following != ":"
        if looks_like_value and token not in VALID_JSON_LITERALS:
            line, column = _line_col(text, start)
            chunks.append(text[last_idx:start])
            chunks.append("null")
            last_idx = idx
            repairs.append({"token": token, "line": line, "column": column})

    if not repairs:
        return text, repairs

    chunks.append(text[last_idx:])
    return "".join(chunks), repairs


def _load_results(input_path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(input_path, "r") as f:
        raw_text = f.read()

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        repaired_text, repairs = _repair_invalid_json_literals(raw_text)
        if not repairs:
            raise ValueError(
                f"Failed to parse JSON in {input_path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc

        try:
            payload = json.loads(repaired_text)
        except json.JSONDecodeError as repair_exc:
            raise ValueError(
                f"Failed to parse JSON in {input_path} at line {repair_exc.lineno}, "
                f"column {repair_exc.colno} even after repairing {len(repairs)} invalid literal(s)."
            ) from repair_exc

        print(
            f"Warning: repaired {len(repairs)} invalid JSON literal(s) while loading {input_path}.",
            file=sys.stderr,
        )
        for repair in repairs:
            print(
                f"  line {repair['line']}, column {repair['column']}: replaced `{repair['token']}` with `null`",
                file=sys.stderr,
            )

    if isinstance(payload, dict):
        return payload, payload.get("results", [])
    return {"results": payload}, payload


def _write_results(output_path: str, envelope: dict[str, Any], results: list[dict[str, Any]]) -> None:
    if "results" in envelope:
        envelope["results"] = results
        to_write = envelope
    else:
        to_write = results

    with open(output_path, "w") as f:
        json.dump(to_write, f, indent=2)


def process_results(results: list[dict[str, Any]], keep_original: bool) -> dict[str, Any]:
    n_total = len(results)
    n_service_found = 0
    n_gold_changed = 0
    n_discovery_acc_changed = 0

    for row in results:
        sparql = row.get("gold:sparql") or row.get("gold_sparql") or ""
        extracted = extract_service_endpoints(sparql)
        original_gold = row.get("gold_endpoints", [])

        if extracted:
            n_service_found += 1
            if original_gold != extracted:
                n_gold_changed += 1

            if keep_original and "gold_endpoints_original" not in row:
                row["gold_endpoints_original"] = original_gold

            row["gold_endpoints"] = extracted

        recomputed_gold = row.get("gold_endpoints", [])
        predicted = row.get("predicted_endpoints", [])

        old_acc = row.get("discovery_accuracy")
        new_acc = discovery_accuracy(predicted, recomputed_gold)
        row["discovery_accuracy"] = new_acc
        row["discovery_f1"] = discovery_f1(predicted, recomputed_gold)

        if old_acc is not None and old_acc != new_acc:
            n_discovery_acc_changed += 1

    accuracy_values = [float(r.get("discovery_accuracy", 0.0)) for r in results]
    f1_values = [float(r.get("discovery_f1", 0.0)) for r in results]

    return {
        "n_total": n_total,
        "n_service_found": n_service_found,
        "n_gold_changed": n_gold_changed,
        "n_discovery_acc_changed": n_discovery_acc_changed,
        "discovery_accuracy_mean": _mean(accuracy_values),
        "discovery_f1_mean": _mean(f1_values),
    }


def build_default_output_path(input_path: str) -> str:
    root, ext = os.path.splitext(input_path)
    return f"{root}_service_recalc{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Override gold_endpoints from gold SERVICE clauses and recompute discovery metrics."
    )
    parser.add_argument("--input", required=True, help="Path to benchmark result JSON.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. If omitted, writes <input>_service_recalc.json.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file in place.",
    )
    parser.add_argument(
        "--keep-original-gold",
        action="store_true",
        help="Store previous gold_endpoints in gold_endpoints_original before overriding.",
    )
    args = parser.parse_args()

    if args.in_place and args.output:
        raise ValueError("Use either --in-place or --output, not both.")

    output_path = args.input if args.in_place else (args.output or build_default_output_path(args.input))

    envelope, results = _load_results(args.input)
    summary = process_results(results, keep_original=args.keep_original_gold)
    _write_results(output_path, envelope, results)

    print(f"Input:  {args.input}")
    print(f"Output: {output_path}")
    print(f"Rows processed:                       {summary['n_total']}")
    print(f"Rows with SERVICE endpoints found:    {summary['n_service_found']}")
    print(f"Rows with changed gold_endpoints:     {summary['n_gold_changed']}")
    print(f"Rows with changed discovery_accuracy: {summary['n_discovery_acc_changed']}")
    print(f"Mean discovery_accuracy:              {summary['discovery_accuracy_mean']:.4f}")
    print(f"Mean discovery_f1:                    {summary['discovery_f1_mean']:.4f}")


if __name__ == "__main__":
    main()
