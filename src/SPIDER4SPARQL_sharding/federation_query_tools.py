from collections import defaultdict
import re
from rdflib import RDF, URIRef
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery


RDF_TYPE_TOKENS = {
    "a",
    "rdf:type",
    "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
}


def _term_to_sparql_str(term) -> str:
    if term is None:
        return ""
    if term.__class__.__name__ == "Variable":
        return f"?{term}"
    if isinstance(term, URIRef):
        return f"<{str(term)}>"
    if hasattr(term, "n3"):
        return term.n3()
    return str(term)


def _collect_bgp_triples(node) -> list:
    triples = []

    if node is None:
        return triples

    node_name = getattr(node, "name", None)
    if node_name == "BGP" and hasattr(node, "triples"):
        triples.extend(list(node.triples))

    if isinstance(node, dict):
        for value in node.values():
            triples.extend(_collect_bgp_triples(value))
    elif isinstance(node, (list, tuple, set)):
        for item in node:
            triples.extend(_collect_bgp_triples(item))
    else:
        keys = getattr(node, "keys", None)
        if callable(keys):
            for key in node.keys():
                try:
                    triples.extend(_collect_bgp_triples(node[key]))
                except Exception:
                    continue

    return triples


def analyze_query_classes(sparql: str, subject_to_class: dict) -> dict:
    classes_involved = set()
    triple_patterns = []

    try:
        algebra = translateQuery(parseQuery(sparql)).algebra
        raw_triples = _collect_bgp_triples(algebra)

        for s, p, o in raw_triples:
            s_str = _term_to_sparql_str(s)
            p_str = _term_to_sparql_str(p)
            o_str = _term_to_sparql_str(o)
            triple_patterns.append((s_str, p_str, o_str))

            if p == RDF.type or p_str in ("a", "rdf:type", f"<{str(RDF.type)}>"):
                if isinstance(o, URIRef):
                    classes_involved.add(str(o))
                elif not o_str.startswith("?"):
                    classes_involved.add(o_str.strip("<>"))

            if isinstance(s, URIRef) and s in subject_to_class:
                classes_involved.add(str(subject_to_class[s]))

    except Exception:
        triple_patterns = re.findall(
            r'(\?\w+|\<[^>]+\>)\s+(\?\w+|\<[^>]+\>|[\w:]+)\s+(\?\w+|\<[^>]+\>|[\w:]+|"[^"]*")',
            sparql,
        )
        for _s, p, o in triple_patterns:
            if p in ["rdf:type", "a"] and not o.startswith("?"):
                classes_involved.add(o.strip("<>"))

    return {
        "classes": classes_involved,
        "is_federated": len(classes_involved) > 1,
        "triple_patterns": triple_patterns,
    }


def _strip_uri_brackets(value: str) -> str:
    if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
        return value[1:-1]
    return value


def _local_name(uri_like: str) -> str:
    token = _strip_uri_brackets(uri_like)
    if "#" in token:
        return token.rsplit("#", 1)[-1]
    if "/" in token:
        return token.rsplit("/", 1)[-1]
    if ":" in token:
        return token.rsplit(":", 1)[-1]
    return token


def _class_key_index(shard_info: dict) -> dict:
    idx = {}
    for k in shard_info.keys():
        k_str = str(k)
        idx[k_str] = k
        idx[_strip_uri_brackets(k_str)] = k
        idx[_local_name(k_str)] = k
    return idx


def _resolve_class_key(class_token: str, shard_info: dict):
    if class_token is None:
        return None
    idx = _class_key_index(shard_info)
    candidates = [
        class_token,
        _strip_uri_brackets(class_token),
        _local_name(class_token),
    ]
    for candidate in candidates:
        if candidate in idx:
            return idx[candidate]
    return None


def _split_query_sections(query: str):
    m = re.search(r"\bWHERE\b", query, flags=re.IGNORECASE)
    if not m:
        return query.strip(), "", ""

    head = query[: m.start()].strip()
    rest = query[m.end() :]

    brace_start = rest.find("{")
    if brace_start == -1:
        return head, "", rest.strip()

    i = brace_start
    depth = 0
    end_pos = -1
    while i < len(rest):
        c = rest[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end_pos = i
                break
        i += 1

    if end_pos == -1:
        return head, rest[brace_start + 1 :].strip(), ""

    where_body = rest[brace_start + 1 : end_pos].strip()
    tail = rest[end_pos + 1 :].strip()
    return head, where_body, tail


def group_patterns_by_class(triple_patterns: list, shard_info: dict) -> tuple:
    subject_class = {}
    grouped = defaultdict(list)
    unresolved = []

    for s, p, o in triple_patterns:
        if p in RDF_TYPE_TOKENS and not str(o).startswith("?"):
            subject_class[s] = o

    for s, p, o in triple_patterns:
        cls_token = None
        if p in RDF_TYPE_TOKENS and not str(o).startswith("?"):
            cls_token = o
        elif s in subject_class:
            cls_token = subject_class[s]
        elif isinstance(o, str) and o in subject_class:
            cls_token = subject_class[o]

        if cls_token is None:
            unresolved.append((s, p, o))
            continue

        cls_key = _resolve_class_key(cls_token, shard_info)
        if cls_key is None:
            unresolved.append((s, p, o))
        else:
            grouped[cls_key].append((s, p, o))

    return grouped, unresolved


def rewrite_as_federated(original_sparql: str, query_analysis: dict, shard_info: dict) -> str:
    if not query_analysis.get("is_federated"):
        return None

    triple_patterns = query_analysis.get("triple_patterns", [])
    if not triple_patterns:
        return None

    pattern_groups, unresolved = group_patterns_by_class(triple_patterns, shard_info)
    if not pattern_groups:
        return None

    head, _where_body, tail = _split_query_sections(original_sparql)

    service_blocks = []
    for cls_key, patterns in pattern_groups.items():
        endpoint = shard_info[cls_key]["endpoint_url"]
        lines = [f"    {s} {p} {o} ." for s, p, o in patterns]
        block = [f"  SERVICE <{endpoint}> {{"] + lines + ["  }"]
        service_blocks.append("\n".join(block))

    unresolved_lines = [f"  {s} {p} {o} ." for s, p, o in unresolved]

    where_lines = []
    where_lines.extend(service_blocks)
    if unresolved_lines:
        where_lines.extend(unresolved_lines)

    rewritten = f"{head} WHERE {{\n" + "\n".join(where_lines) + "\n}"
    if tail:
        rewritten += f"\n{tail}"

    return rewritten
