import json
import re
from pathlib import Path

from rdflib import Graph


def _term_to_text(term):
    if term is None:
        return ""
    if hasattr(term, "n3"):
        return term.n3()
    return str(term)


def _normalize_select_result(query_result):
    vars_ = [str(v) for v in getattr(query_result, "vars", [])]
    rows = []
    for binding in query_result.bindings:
        row = tuple(_term_to_text(binding.get(v)) for v in query_result.vars)
        rows.append(row)
    rows.sort()
    return {"type": "SELECT", "vars": vars_, "rows": rows}


def execute_query_local(graph: Graph, sparql: str):
    result = graph.query(sparql)
    result_type = getattr(result, "type", "SELECT")

    if result_type == "ASK":
        return {"type": "ASK", "value": bool(result.askAnswer)}
    if result_type == "SELECT":
        return _normalize_select_result(result)

    # For CONSTRUCT/DESCRIBE, compare canonical N-Triples strings.
    g = Graph()
    for triple in result.graph:
        g.add(triple)
    serialized = g.serialize(format="nt")
    return {"type": str(result_type), "graph_nt": serialized}


def _remove_service_clauses(federated_query: str) -> str:
    """Convert SERVICE blocks into plain group patterns for local execution."""
    query = federated_query

    while True:
        m = re.search(r"\bSERVICE\b\s*<[^>]+>\s*\{", query, flags=re.IGNORECASE)
        if not m:
            break

        open_brace_index = query.find("{", m.start())
        if open_brace_index == -1:
            break

        i = open_brace_index
        depth = 0
        close_brace_index = -1
        while i < len(query):
            ch = query[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    close_brace_index = i
                    break
            i += 1

        if close_brace_index == -1:
            break

        inner = query[open_brace_index + 1 : close_brace_index]
        replacement = "{\n" + inner.strip() + "\n}"
        query = query[: m.start()] + replacement + query[close_brace_index + 1 :]

    return query


def load_union_graph_from_shards(shards_info: dict) -> Graph:
    union_graph = Graph()
    for info in shards_info.values():
        shard_path = info.get("path")
        if shard_path:
            union_graph.parse(shard_path)
    return union_graph


def validate_rewrite_local(
    original_sparql: str,
    federated_sparql: str,
    full_kg_path: str,
    shards_info: dict,
) -> dict:
    """Validate rewrite without endpoints by local in-process SPARQL execution."""
    full_graph = Graph()
    full_graph.parse(full_kg_path)

    union_graph = load_union_graph_from_shards(shards_info)
    de_federated = _remove_service_clauses(federated_sparql)

    original_result = execute_query_local(full_graph, original_sparql)
    rewritten_result = execute_query_local(union_graph, de_federated)

    equivalent = original_result == rewritten_result
    return {
        "equivalent": equivalent,
        "original_result": original_result,
        "rewritten_result": rewritten_result,
        "executed_rewritten_query": de_federated,
    }


def build_single_query_benchmark(
    output_path: str,
    domain_name: str,
    question: str,
    original_sparql: str,
    query_analysis: dict,
    federated_sparql: str,
    shards_info: dict,
    validation: dict,
):
    benchmark = {
        "metadata": {
            "name": "FedSpider-lightweight",
            "description": "Single-query federated benchmark packaged in lightweight local mode.",
            "version": "1.0",
            "num_domains": 1,
            "num_federated_queries": 1 if federated_sparql else 0,
            "num_single_queries": 0 if federated_sparql else 1,
        },
        "domains": [
            {
                "name": domain_name,
                "shards": [
                    {
                        "class": str(cls),
                        "endpoint": info.get("endpoint_url"),
                        "triple_count": info.get("triple_count"),
                        "path": info.get("path"),
                    }
                    for cls, info in shards_info.items()
                ],
            }
        ],
        "examples": [
            {
                "id": f"{domain_name}_0",
                "domain": domain_name,
                "question": question,
                "original_sparql": original_sparql,
                "federated_sparql": federated_sparql,
                "classes_involved": sorted(query_analysis.get("classes", [])),
                "num_endpoints": len(query_analysis.get("classes", [])) or 1,
                "type": "federated" if federated_sparql else "single",
                "is_valid": validation.get("equivalent", False),
            }
        ],
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    return benchmark
