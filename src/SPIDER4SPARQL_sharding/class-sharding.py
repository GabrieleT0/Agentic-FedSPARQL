# Spider4SPARQL — Class Sharding Pipeline
#
# Confirmed structure:
#   /data/original_SPIDER4SPARQL/
#   ├── train/train.json          keys: kg_name, question, query
#   ├── dev/dev.json              keys: kg_name, question, query
#   ├── materialized_triples_train/<kg_name>.ttl
#   └── materialized_triples_dev/<kg_name>.ttl
#
# Output: /data/SPIDER4FedSPARQL/class-sharding/
#   ├── shards/<kg_name>/<ClassName>.ttl
#   ├── fkgqa_benchmark.json
#   ├── fkgqa_train.json
#   └── fkgqa_dev.json

# pip install rdflib requests

import re
import json
import requests
from pathlib import Path
from collections import defaultdict

from rdflib import Graph, RDF, OWL, URIRef
from rdflib.namespace import RDFS

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path("../../data/SPIDER4SPARQL")
OUTPUT_DIR = Path("../../data/SPIDER4FedSPARQL/class-sharding")
FUSEKI_URL = "http://host.docker.internal:3030"
FUSEKI_ADMIN_USER = "admin"
FUSEKI_ADMIN_PASS = "NG0sz3FE5L4Yk0t"

TRAIN_DIR = BASE_DIR / "train"
DEV_DIR   = BASE_DIR / "dev"
TTL_TRAIN = BASE_DIR / "materialized_triples_train"
TTL_DEV   = BASE_DIR / "materialized_triples_dev"

# KGs with TTL file larger than this (MB) use the streaming path
STREAMING_THRESHOLD_MB = 50

# Dedicated empty Fuseki dataset used as the SPARQL executor for federated
# queries — all patterns are wrapped in SERVICE, so the dataset itself needs
# no data; Fuseki dispatches every SERVICE call outward.
FEDERATION_DATASET = "SPIDER4FedSPARQL"

_META_NS = (
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2001/XMLSchema#",
)


def _extract_base_ns(g, ttl_path=None) -> str | None:
    """Return the base namespace for a KG using multiple fallback strategies.

    1. rdflib namespace with empty prefix  (PREFIX : <...> in the TTL)
    2. @base / BASE directive parsed from the raw TTL source
    3. Most common URI prefix (up to last # or /) across all class URIs
    """
    from rdflib import URIRef as _URIRef
    # Strategy 1
    for prefix, ns in g.namespaces():
        if prefix == "":
            candidate = str(ns)
            if not any(candidate.startswith(m) for m in _META_NS):
                return candidate
    # Strategy 2
    if ttl_path is not None:
        try:
            raw = ttl_path.read_text(encoding="utf-8", errors="replace")
            m = re.search(r'(?:@base|BASE)\s+<([^>]+)>', raw, re.IGNORECASE)
            if m:
                base = m.group(1)
                if not any(base.startswith(meta) for meta in _META_NS):
                    return base
        except Exception:
            pass
    # Strategy 3 — derive from typed-subject URIs only (most reliable signal).
    from rdflib import RDF as _RDF
    prefix_counts: dict = {}
    for s, _p, _o in g.triples((None, _RDF.type, None)):
        if not isinstance(s, _URIRef):
            continue
        uri = str(s)
        if any(uri.startswith(m) for m in _META_NS):
            continue
        cut = max(uri.rfind("#"), uri.rfind("/"))
        if cut > 0:
            ns_candidate = uri[:cut + 1]
            prefix_counts[ns_candidate] = prefix_counts.get(ns_candidate, 0) + 1
    if prefix_counts:
        candidates = list(prefix_counts.keys())
        candidates.sort(key=len)
        for pivot in candidates:
            if all(c.startswith(pivot) for c in candidates):
                return pivot
        common = candidates[0]
        for c in candidates[1:]:
            while not c.startswith(common):
                common = common[:-1]
        cut = max(common.rfind("#"), common.rfind("/"))
        if cut > 0:
            return common[:cut + 1]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load NL/SPARQL pairs
# ─────────────────────────────────────────────────────────────────────────────

def load_pairs(split_dir: Path, split_name: str) -> list:
    pairs = []
    json_files = sorted(split_dir.glob("**/*.json"))
    print(f"[{split_name}] {len(json_files)} file(s) in {split_dir}")
    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            sparql   = entry.get("query") or entry.get("sparql") or entry.get("SPARQL", "")
            question = entry.get("question") or entry.get("nl", "")
            kg_name  = entry.get("kg_name") or entry.get("db_id") or json_path.stem
            if sparql.strip() and question.strip() and kg_name:
                pairs.append({
                    "question": question.strip(),
                    "sparql":   sparql.strip(),
                    "kg_name":  kg_name,
                    "split":    split_name,
                })
    print(f"[{split_name}] loaded {len(pairs)} pairs")
    return pairs


def find_ttl(kg_name: str) -> Path | None:
    for ttl_dir in [TTL_TRAIN, TTL_DEV]:
        direct = ttl_dir / f"{kg_name}.ttl"
        if direct.exists():
            return direct
        hits = list(ttl_dir.glob(f"**/{kg_name}.ttl"))
        if hits:
            return hits[0]
    return None

# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _safe_name(uri_or_str) -> str:
    s = str(uri_or_str)
    name = s.split("#")[-1] if "#" in s else s.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name) or "unknown"


def _nt_triple(s, p, o) -> str:
    """Serialize a single triple as one N-Triples line."""
    from rdflib import BNode, Literal

    def _term(t):
        if isinstance(t, URIRef):
            return f"<{t}>"
        if isinstance(t, BNode):
            return f"_:{t}"
        lit = str(t).replace("\\", "\\\\").replace('"', '\\"') \
                    .replace("\n", "\\n").replace("\r", "\\r")
        if t.language:
            return f'"{lit}"@{t.language}'
        if t.datatype:
            return f'"{lit}"^^<{t.datatype}>'
        return f'"{lit}"'

    return f"{_term(s)} {_term(p)} {_term(o)} .\n"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2+3 (small KGs ≤ STREAMING_THRESHOLD_MB) — in-memory path
# ─────────────────────────────────────────────────────────────────────────────

def analyze_kg(ttl_path: Path) -> dict:
    print(f"  Parsing {ttl_path.name}...", end=" ", flush=True)
    g = Graph()
    g.parse(str(ttl_path), format="turtle")

    base_ns = _extract_base_ns(g, ttl_path)

    local_to_uri = {}
    for s, p, o in g.triples((None, RDF.type, OWL.Class)):
        if isinstance(s, URIRef) and not any(str(s).startswith(ns) for ns in _META_NS):
            local_to_uri[str(s).split("#")[-1].split("/")[-1].lower()] = str(s)
    for s, p, o in g.triples((None, RDF.type, RDFS.Class)):
        if isinstance(s, URIRef) and not any(str(s).startswith(ns) for ns in _META_NS):
            local_to_uri[str(s).split("#")[-1].split("/")[-1].lower()] = str(s)
    if not local_to_uri:
        for s, p, o in g.triples((None, RDF.type, None)):
            if isinstance(o, URIRef) and not any(str(o).startswith(ns) for ns in _META_NS):
                local_to_uri[str(o).split("#")[-1].split("/")[-1].lower()] = str(o)

    subject_to_class = {}
    for s, p, o in g.triples((None, RDF.type, None)):
        if isinstance(o, URIRef) and not any(str(o).startswith(ns) for ns in _META_NS):
            subject_to_class[s] = o

    class_triples   = defaultdict(list)
    untyped_triples = []
    for s, p, o in g:
        if s in subject_to_class:
            class_triples[subject_to_class[s]].append((s, p, o))
        else:
            untyped_triples.append((s, p, o))

    all_classes = set(class_triples.keys())
    print(f"{len(g)} triples | {len(all_classes)} classes")

    return {
        "graph":            g,
        "base_ns":          base_ns,
        "local_to_uri":     local_to_uri,
        "subject_to_class": subject_to_class,
        "class_triples":    dict(class_triples),
        "untyped_triples":  untyped_triples,
        "all_classes":      all_classes,
    }


def shard_kg(analysis: dict, kg_name: str) -> dict:
    shard_dir = OUTPUT_DIR / "shards" / kg_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    g = analysis["graph"]
    shards = {}

    for cls_uri, triples in analysis["class_triples"].items():
        shard = Graph()
        for prefix, ns in g.namespaces():
            shard.bind(prefix, ns)
        for s, p, o in triples:
            shard.add((s, p, o))
        for s, p, o in triples:
            if isinstance(o, URIRef) and o in analysis["subject_to_class"]:
                if analysis["subject_to_class"][o] == cls_uri:
                    shard.add((o, RDF.type, cls_uri))

        class_name    = _safe_name(cls_uri)
        endpoint_name = f"{kg_name}__{class_name}"
        shard_path    = shard_dir / f"{class_name}.ttl"
        shard.serialize(str(shard_path), format="turtle")

        shards[str(cls_uri)] = {
            "path":          str(shard_path),
            "class_name":    class_name,
            "endpoint_name": endpoint_name,
            "triple_count":  len(shard),
            "endpoint_url":  f"{FUSEKI_URL}/{endpoint_name}/sparql",
        }
        print(f"    [{class_name}] {len(shard)} triples")

    if analysis["untyped_triples"]:
        shared = Graph()
        for prefix, ns in g.namespaces():
            shared.bind(prefix, ns)
        for s, p, o in analysis["untyped_triples"]:
            shared.add((s, p, o))
        endpoint_name = f"{kg_name}__shared"
        shared_path   = shard_dir / "_shared.ttl"
        shared.serialize(str(shared_path), format="turtle")
        shards["_shared"] = {
            "path":          str(shared_path),
            "class_name":    "_shared",
            "endpoint_name": endpoint_name,
            "triple_count":  len(shared),
            "endpoint_url":  f"{FUSEKI_URL}/{endpoint_name}/sparql",
        }

    with open(shard_dir / "metadata.json", "w") as f:
        json.dump(shards, f, indent=2)
    return shards

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2+3 (large KGs > STREAMING_THRESHOLD_MB) — two-pass streaming path
# ─────────────────────────────────────────────────────────────────────────────

def analyze_and_shard_streaming(ttl_path: Path, kg_name: str) -> dict | None:
    shard_dir = OUTPUT_DIR / "shards" / kg_name
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [pass1]...", end=" ", flush=True)
    try:
        g_types = Graph()
        g_types.parse(str(ttl_path), format="turtle")
    except Exception as e:
        print(f"SKIP: {e}")
        return None

    namespaces           = {str(pref): str(ns) for pref, ns in g_types.namespaces()}
    base_ns              = _extract_base_ns(g_types, ttl_path)
    subject_to_class_str = {}
    local_to_uri         = {}

    for s, p, o in g_types.triples((None, RDF.type, None)):
        if isinstance(o, URIRef) and not any(str(o).startswith(ns) for ns in _META_NS):
            subject_to_class_str[str(s)] = str(o)
            local = str(o).split("#")[-1].split("/")[-1]
            local_to_uri[local.lower()] = str(o)

    for s, p, o in g_types.triples((None, RDF.type, OWL.Class)):
        if isinstance(s, URIRef) and not any(str(s).startswith(ns) for ns in _META_NS):
            local = str(s).split("#")[-1].split("/")[-1]
            local_to_uri[local.lower()] = str(s)

    all_classes = set(subject_to_class_str.values())
    del g_types

    if not all_classes:
        print("SKIP (no classes)")
        return None

    print(f"{len(all_classes)} classes | [pass2]...", end=" ", flush=True)

    shard_nt_paths = {cls: shard_dir / f"{_safe_name(cls)}.nt" for cls in all_classes}
    shared_nt_path = shard_dir / "_shared.nt"

    for p in list(shard_nt_paths.values()) + [shared_nt_path]:
        if p.exists():
            p.unlink()

    shard_counts = defaultdict(int)
    shared_count = 0
    total        = 0

    shard_handles = {}
    shared_handle = None
    try:
        shard_handles = {cls: open(path, "w", encoding="utf-8")
                         for cls, path in shard_nt_paths.items()}
        shared_handle = open(shared_nt_path, "w", encoding="utf-8")

        g_full = Graph()
        g_full.parse(str(ttl_path), format="turtle")

        for s, p, o in g_full:
            total += 1
            line   = _nt_triple(s, p, o)
            s_str  = str(s)
            if s_str in subject_to_class_str:
                cls = subject_to_class_str[s_str]
                shard_handles[cls].write(line)
                shard_counts[cls] += 1
            else:
                shared_handle.write(line)
                shared_count += 1

        del g_full

    except Exception as e:
        print(f"SKIP (pass2 error): {e}")
        return None
    finally:
        for h in shard_handles.values():
            h.close()
        if shared_handle:
            shared_handle.close()

    print(f"{total:,} triples")

    shards = {}
    for cls in all_classes:
        nt_path       = shard_nt_paths[cls]
        class_name    = _safe_name(cls)
        endpoint_name = f"{kg_name}__{class_name}"
        ttl_out       = shard_dir / f"{class_name}.ttl"

        g_shard = Graph()
        for pref, ns in namespaces.items():
            try:
                g_shard.bind(pref, ns)
            except Exception:
                pass
        g_shard.parse(str(nt_path), format="nt")
        g_shard.serialize(str(ttl_out), format="turtle")
        nt_path.unlink()
        del g_shard

        n = shard_counts[cls]
        print(f"    [{class_name}] {n:,} triples")
        shards[cls] = {
            "path":          str(ttl_out),
            "class_name":    class_name,
            "endpoint_name": endpoint_name,
            "triple_count":  n,
            "endpoint_url":  f"{FUSEKI_URL}/{endpoint_name}/sparql",
        }

    if shared_count > 0:
        g_shared = Graph()
        for pref, ns in namespaces.items():
            try:
                g_shared.bind(pref, ns)
            except Exception:
                pass
        g_shared.parse(str(shared_nt_path), format="nt")
        ttl_shared = shard_dir / "_shared.ttl"
        g_shared.serialize(str(ttl_shared), format="turtle")
        shared_nt_path.unlink()
        del g_shared
        shards["_shared"] = {
            "path":          str(ttl_shared),
            "class_name":    "_shared",
            "endpoint_name": f"{kg_name}__shared",
            "triple_count":  shared_count,
            "endpoint_url":  f"{FUSEKI_URL}/{kg_name}__shared/sparql",
        }

    with open(shard_dir / "metadata.json", "w") as f:
        json.dump(shards, f, indent=2)

    return {
        "graph":            None,
        "base_ns":          base_ns,
        "local_to_uri":     local_to_uri,
        "subject_to_class": {URIRef(k): URIRef(v)
                             for k, v in subject_to_class_str.items()},
        "class_triples":    {},
        "untyped_triples":  [],
        "all_classes":      {URIRef(c) for c in all_classes},
        "shards":           shards,
    }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Rewrite SPARQL queries with SERVICE clauses
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that indicate SQL-style syntax leaked into the SPARQL translation
# and will always cause a parse error on any standards-compliant endpoint.
_BAD_SPARQL_PATTERNS = [
    # SQL-style alias in ORDER BY:  ORDER BY expr AS ?var
    re.compile(r'\bORDER\s+BY\b[^{]*?\bAS\b', re.IGNORECASE | re.DOTALL),
    # Consecutive AS outside of (expr AS ?var): catches   ) AS ?x AS ?y
    re.compile(r'\)\s+AS\s+\??\w+\s+AS\b', re.IGNORECASE),
    # Nested blank nodes at statement level:  [ [
    re.compile(r'\[\s*\['),
]


def _sparql_is_well_formed(sparql: str) -> bool:
    """Return False if the query matches a known-bad syntax pattern."""
    for pat in _BAD_SPARQL_PATTERNS:
        if pat.search(sparql):
            return False
    return True


def _extract_prefixes(sparql: str) -> dict:
    prefixes = {}
    for m in re.finditer(r'PREFIX\s+(\w*:)\s+<([^>]+)>', sparql, re.IGNORECASE):
        prefixes[m.group(1)] = m.group(2)
    return prefixes


def _expand_illegal_prefixed_names(sparql_body: str, prefixes: dict) -> str:
    """Expand prefixed names whose local part contains '#' or other characters
    that are illegal in SPARQL local names."""
    def _replace(m):
        prefix_colon = m.group(1)
        local        = m.group(2)
        ns = prefixes.get(prefix_colon)
        if ns is None:
            return m.group(0)
        return f"<{ns}{local}>"

    pattern = re.compile(r'(?<![<"/])(\w*:)([\w\-]+#[\w\-#]+)')
    return pattern.sub(_replace, sparql_body)


def _extract_local_class_names(sparql: str):
    tokens = re.findall(r'\?\w+\s+(?:a|rdf:type)\s+:?([\w\-]+)', sparql, re.IGNORECASE)
    full_uris = re.findall(r'\?\w+\s+(?:a|rdf:type)\s+<([^>]+)>', sparql, re.IGNORECASE)
    return tokens, full_uris


def _resolve_local_to_endpoint(local_name: str, analysis: dict, shards: dict) -> str | None:
    key = local_name.lower()
    if key in analysis["local_to_uri"]:
        full_uri = analysis["local_to_uri"][key]
        if full_uri in shards:
            return shards[full_uri]["endpoint_url"]
    for shard_uri, shard_info in shards.items():
        if shard_uri == "_shared":
            continue
        if shard_uri.split("#")[-1].split("/")[-1].lower() == key:
            return shard_info["endpoint_url"]
    if analysis.get("base_ns"):
        candidate = analysis["base_ns"] + local_name
        if candidate in shards:
            return shards[candidate]["endpoint_url"]
    return None


def _rewrite_subquery(subquery_block: str, shards: dict, analysis: dict) -> str:
    inner_match = re.search(r'WHERE\s*\{(.*)\}', subquery_block, re.DOTALL | re.IGNORECASE)
    if not inner_match:
        return subquery_block
    inner_body   = inner_match.group(1).strip()
    select_match = re.match(r'\{\s*(SELECT\b.*?)WHERE', subquery_block, re.DOTALL | re.IGNORECASE)
    subq_select  = select_match.group(1).strip() if select_match else "SELECT *"
    local_names, full_uris = _extract_local_class_names(inner_body)
    endpoint = None
    for name in local_names:
        endpoint = _resolve_local_to_endpoint(name, analysis, shards)
        if endpoint:
            break
    if not endpoint:
        for uri in full_uris:
            if uri in shards:
                endpoint = shards[uri]["endpoint_url"]
                break
    if not endpoint:
        return subquery_block
    inner_body_expanded = _expand_illegal_prefixed_names(inner_body, _extract_prefixes(subquery_block))
    return (
        f"{{\n"
        f"    {subq_select} WHERE {{\n"
        f"      SERVICE <{endpoint}> {{\n"
        f"        {' '.join(inner_body_expanded.split())}\n"
        f"      }}\n"
        f"    }}\n"
        f"  }}"
    )


def _split_patterns(where_body: str) -> list:
    depth, current, patterns = 0, [], []
    i = 0
    text = where_body
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
            current.append(char)
        elif char == "}":
            depth -= 1
            current.append(char)
            if depth == 0:
                p = "".join(current).strip()
                if p:
                    patterns.append(p)
                current = []
        elif char == "." and depth == 0:
            p = "".join(current).strip()
            if p:
                patterns.append(p)
            current = []
        elif char in ("F", "f") and depth == 0 and text[i:i+6].upper() == "FILTER":
            j = text.find("(", i)
            if j == -1:
                current.append(char)
            else:
                paren_depth = 0
                k = j
                while k < len(text):
                    if text[k] == "(":
                        paren_depth += 1
                    elif text[k] == ")":
                        paren_depth -= 1
                        if paren_depth == 0:
                            break
                    k += 1
                patterns.append(text[i:k+1].strip())
                i = k + 1
                continue
        else:
            current.append(char)
        i += 1
    last = "".join(current).strip()
    if last:
        patterns.append(last)
    return [p for p in patterns if p.strip()]


def _find_cross_shard_join_vars(endpoint_patterns: dict) -> set:
    """Return variables that appear in patterns assigned to MORE THAN ONE
    endpoint.  These are cross-shard join keys that Fuseki can only propagate
    if they are explicitly projected out of each SERVICE block that binds them.

    Without explicit projection, Fuseki has no way to inject the bound URI
    values as VALUES bindings into the subsequent SERVICE call, causing the
    cross-shard join to return zero results.
    """
    var_to_eps: dict[str, set] = defaultdict(set)
    for ep, pats in endpoint_patterns.items():
        for pat in pats:
            for v in re.findall(r'\?(\w+)', pat):
                var_to_eps[v].add(ep)
    return {v for v, eps in var_to_eps.items() if len(eps) > 1}


def _has_aggregation(select_clause: str, modifiers: str) -> bool:
    """Return True if the query uses aggregation (COUNT, SUM, AVG, MIN, MAX,
    GROUP_CONCAT) or GROUP BY.

    This matters for SERVICE block construction: when aggregation is present,
    wrapping each SERVICE block in an inner SELECT pre-collapses rows before
    the outer GROUP BY / COUNT can see them, producing wrong counts.  Bare
    graph patterns must be used instead so that the outer query operates on
    the full un-aggregated solution sequence.
    """
    _AGG_RE = re.compile(
        r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT)\s*\(',
        re.IGNORECASE,
    )
    _GROUPBY_RE = re.compile(r'\bGROUP\s+BY\b', re.IGNORECASE)
    text = select_clause + " " + modifiers
    return bool(_AGG_RE.search(text) or _GROUPBY_RE.search(text))


def rewrite_as_federated(sparql: str, shards: dict, analysis: dict) -> str | None:
    prefixes = _extract_prefixes(sparql)

    if ":" not in prefixes and analysis.get("base_ns"):
        prefixes[":"] = analysis["base_ns"]

    local_names, full_uri_classes = _extract_local_class_names(sparql)
    if not local_names and not full_uri_classes:
        return None

    local_to_endpoint = {}
    for name in local_names:
        ep = _resolve_local_to_endpoint(name, analysis, shards)
        if ep:
            local_to_endpoint[name.lower()] = ep
    for uri in full_uri_classes:
        if uri in shards:
            local_to_endpoint[uri] = shards[uri]["endpoint_url"]
    if not local_to_endpoint:
        return None

    select_match = re.match(
        r'(SELECT\b.*?|ASK\s*|CONSTRUCT\b.*?|DESCRIBE\b.*?)WHERE',
        sparql, re.DOTALL | re.IGNORECASE
    )
    select_clause = select_match.group(1).strip() if select_match else "SELECT *"

    where_start = re.search(r'WHERE\s*\{', sparql, re.IGNORECASE)
    if not where_start:
        return None

    brace_pos = where_start.end() - 1
    depth = 0
    outer_close = -1
    for i in range(brace_pos, len(sparql)):
        if sparql[i] == '{':
            depth += 1
        elif sparql[i] == '}':
            depth -= 1
            if depth == 0:
                outer_close = i
                break
    if outer_close == -1:
        return None

    where_body_raw = sparql[brace_pos + 1 : outer_close]
    after_where    = sparql[outer_close + 1 :].strip()

    _MODIFIER_KWS = re.compile(
        r'(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET)\b.*$',
        re.IGNORECASE | re.DOTALL
    )
    inline_modifiers = ""
    wb = where_body_raw
    _kw_pat = re.compile(
        r'\b(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET)\b',
        re.IGNORECASE
    )
    first_mod_pos = None
    for kw_m in _kw_pat.finditer(wb):
        d = 0
        for ch in wb[:kw_m.start()]:
            if ch == '{': d += 1
            elif ch == '}': d -= 1
        if d == 0:
            first_mod_pos = kw_m.start()
            break
    if first_mod_pos is not None:
        inline_modifiers = wb[first_mod_pos:].strip()
        wb = wb[:first_mod_pos]
    where_body = wb.strip()

    modifiers_parts = []
    if after_where:
        modifiers_parts.append(after_where)
    if inline_modifiers:
        modifiers_parts.append(inline_modifiers)
    modifiers = " ".join(modifiers_parts).strip()

    patterns           = _split_patterns(where_body)
    endpoint_patterns  = defaultdict(list)
    filters_and_blocks = []
    outer_subqueries   = []
    unassigned         = []

    for pattern in patterns:
        p = pattern.strip()
        pu = p.upper()
        if re.match(r'^\{\s*SELECT\b', p, re.IGNORECASE):
            outer_subqueries.append(p)
            continue
        if any(pu.startswith(kw) for kw in ("FILTER","OPTIONAL","UNION","NOT EXISTS","MINUS")):
            filters_and_blocks.append(p)
            continue
        assigned = False
        pl = p.lower()
        for name_or_uri in sorted(local_to_endpoint, key=len, reverse=True):
            endpoint = local_to_endpoint[name_or_uri]
            if re.search(
                r'(?<![:\w]):' + re.escape(name_or_uri) + r'(?![\w])',
                pl, re.IGNORECASE
            ) or name_or_uri in pl:
                endpoint_patterns[endpoint].append(p)
                assigned = True
                break
        if not assigned:
            unassigned.append(p)

    def _best_ep(var_set):
        best_ep, best_n = None, 0
        for ep, ep_pats in endpoint_patterns.items():
            ep_vars = {v for pt in ep_pats for v in re.findall(r'\?(\w+)', pt)}
            n = len(var_set & ep_vars)
            if n > best_n:
                best_n, best_ep = n, ep
        return best_ep or (max(endpoint_patterns, key=lambda e: len(endpoint_patterns[e]))
                           if endpoint_patterns else None)

    for p in unassigned:
        t = _best_ep(set(re.findall(r'\?(\w+)', p)))
        if t:
            endpoint_patterns[t].append(p)
    for b in filters_and_blocks:
        t = _best_ep(set(re.findall(r'\?(\w+)', b)))
        if t:
            endpoint_patterns[t].append(b)

    if not endpoint_patterns:
        return None

    # Only produce a federated rewrite when ≥2 distinct endpoints are used.
    if len(endpoint_patterns) <= 1:
        return None

    # ── FIX 1 (final): cross-shard join variable projection ──────────────────
    #
    # Cross-shard joins require that Fuseki know the bound URI values of the
    # join key when it executes the second SERVICE block.  Fuseki achieves this
    # via VALUES injection — but only when the join variable is explicitly
    # projected by an inner SELECT inside the FIRST SERVICE block.
    #
    # The complication is aggregation:
    #
    #   • Wrapping the COUNTING block (e.g. student) in an inner SELECT causes
    #     Fuseki to evaluate it as a complete sub-relation first, collapsing
    #     duplicate rows.  The outer COUNT then counts collapsed rows → wrong.
    #
    #   • Wrapping the DEFINING block (e.g. faculty, which has ?T1 a :faculty)
    #     only projects the join key — it produces one row per faculty member,
    #     not one row per student — so the outer COUNT still operates on the
    #     full un-aggregated student rows from the second bare SERVICE block.
    #
    # Correct strategy (works for both agg and non-agg queries):
    #
    #   1. Identify "defining" endpoints: those whose patterns contain
    #      `?joinVar a :SomeClass` (they bind the join key as a subject).
    #   2. Wrap ONLY the defining endpoint in an inner SELECT that projects
    #      just the join-key variable(s) plus any other variables that the
    #      outer SELECT or GROUP BY references from that endpoint.
    #   3. Leave all other endpoints as bare graph patterns.
    #   4. Order defining endpoints first so Fuseki sees bound values before
    #      executing the bare consuming blocks.
    #
    # This ensures VALUES injection fires for the consuming block while keeping
    # the consuming block's rows un-collapsed for correct outer aggregation.

    join_vars = _find_cross_shard_join_vars(endpoint_patterns)
    use_agg   = _has_aggregation(select_clause, modifiers)

    # Collect variables referenced in the outer SELECT clause and GROUP BY
    # so we know what the defining block must project beyond the join key.
    outer_select_vars = set(re.findall(r'\?(\w+)', select_clause))
    outer_select_vars |= set(re.findall(r'\?(\w+)', modifiers))

    def _ep_defines_join_var(ep_pats: list, jvars: set) -> bool:
        """True if this endpoint's patterns contain  ?joinVar a :SomeClass ."""
        for p in ep_pats:
            for jv in jvars:
                if re.search(
                    r'\?' + re.escape(jv) + r'\s+(?:a\b|rdf:type\b)',
                    p, re.IGNORECASE
                ):
                    return True
        return False

    # Sort: defining endpoints first so their bound join-key values are
    # available when Fuseki starts evaluating the consuming bare blocks.
    items = list(endpoint_patterns.items())
    if join_vars:
        items.sort(
            key=lambda kv: (0 if _ep_defines_join_var(kv[1], join_vars) else 1)
        )

    prefix_block   = "\n".join(f"PREFIX {k} <{v}>" for k, v in prefixes.items())
    service_blocks = []

    for endpoint_url, ep_pats in items:
        ep_all_vars  = {v for pat in ep_pats for v in re.findall(r'\?(\w+)', pat)}
        ep_join_vars = join_vars & ep_all_vars
        is_defining  = bool(ep_join_vars) and _ep_defines_join_var(ep_pats, ep_join_vars)

        lines = []
        for p in ep_pats:
            p = p.strip()
            if not p:
                continue
            pu = p.upper()
            is_block = any(pu.startswith(kw) for kw in
                           ("FILTER", "OPTIONAL", "UNION", "NOT EXISTS", "MINUS"))
            if lines and not is_block:
                prev_is_block = any(lines[-1].strip().upper().startswith(kw) for kw in
                                    ("FILTER", "OPTIONAL", "UNION", "NOT EXISTS", "MINUS"))
                if not prev_is_block:
                    lines[-1] += " ."
            lines.append(p)

        body = "\n      ".join(lines)
        if not body:
            continue
        body = _expand_illegal_prefixed_names(body, prefixes)

        if is_defining:
            # Project the join key(s) plus any variables from this endpoint
            # that the outer SELECT / GROUP BY references.  Projecting ONLY
            # what is needed avoids the sub-query collapsing rows that belong
            # to a consuming (counting) block.
            #
            # SELECT DISTINCT is critical for performance: Fuseki's federated
            # engine builds a single batched VALUES clause from the inner SELECT
            # rows and injects it into the consuming SERVICE in one HTTP call.
            # Without DISTINCT, duplicate ?T1 rows (one per attribute triple on
            # the same subject) cause Fuseki to repeat the injection once per
            # duplicate, triggering N sequential SERVICE calls to the consuming
            # endpoint instead of one batched call — causing timeouts whenever
            # the join-key entity has more than a few attribute triples.
            projected = ep_join_vars | (ep_all_vars & outer_select_vars)
            # For non-aggregation queries project all vars so the outer SELECT
            # can still access e.g. ?name, ?location from this block.
            if not use_agg:
                projected = ep_all_vars
            var_list = " ".join(f"?{v}" for v in sorted(projected))
            block = (
                f"  SERVICE <{endpoint_url}> {{\n"
                f"    SELECT DISTINCT {var_list} WHERE {{\n"
                f"      {body}\n"
                f"    }}\n"
                f"  }}"
            )
        else:
            # Consuming block (or no join vars): bare graph pattern.
            # Fuseki injects the bound join-key values from the preceding
            # defining block via a single batched VALUES clause automatically.
            block = f"  SERVICE <{endpoint_url}> {{\n    {body}\n  }}"

        service_blocks.append(block)

    if not service_blocks:
        return None

    outer_sq_str = ""
    if outer_subqueries:
        outer_sq_str = "\n  " + "\n  ".join(
            _rewrite_subquery(sq, shards, analysis) for sq in outer_subqueries
        )

    # ── Aggregation + cross-shard join: wrap SERVICE blocks in an inner SELECT
    #
    # When the outer query aggregates (COUNT, SUM, GROUP BY …) over a join
    # spanning two SERVICE endpoints, Fuseki's federated optimizer may reorder
    # the SERVICE blocks — executing the larger (consuming) block first without
    # any bound values, then joining.  COUNT then operates over the full
    # un-joined set, inflating results by the number of join-key values (16×).
    #
    # Fix: wrap all SERVICE blocks in a single inner SELECT that lists every
    # variable needed by the outer aggregation.  This forces Fuseki to fully
    # materialise the cross-shard join before the outer GROUP BY / COUNT sees
    # any rows.  The defining block's inner SELECT DISTINCT still fires VALUES
    # injection into the consuming block within that inner scope, so execution
    # remains efficient (one batched call per join-key set, not N calls).
    if use_agg and join_vars and len(service_blocks) > 1:
        # Variables the outer query needs from inside the SERVICE blocks
        inner_vars: set = set()
        for ep_pats in endpoint_patterns.values():
            for pat in ep_pats:
                inner_vars |= set(re.findall(r'\?(\w+)', pat))
        # Project ALL variables so that rows differing only in un-projected
        # variables (e.g. distinct ?T2 allergy entities for the same ?T1 student)
        # are preserved as separate rows for the outer COUNT / GROUP BY.
        # Restricting the projection collapses those rows inside the sub-query
        # even without DISTINCT, because Fuseki treats a projected sub-query
        # result as a relation and may deduplicate identical projected tuples.
        needed = inner_vars
        inner_var_list = " ".join(f"?{v}" for v in sorted(needed))
        joined_blocks  = "\n".join(service_blocks)
        inner_sq = (
            f"  {{\n"
            f"    SELECT {inner_var_list} WHERE {{\n"
            f"{joined_blocks}\n"
            f"    }}\n"
            f"  }}"
        )
        where_body = inner_sq + outer_sq_str
    else:
        where_body = "\n".join(service_blocks) + outer_sq_str

    result = (
        (f"{prefix_block}\n\n" if prefix_block else "")
        + f"{select_clause} WHERE {{\n"
        + where_body
        + "\n}"
    )
    if modifiers:
        result += f"\n{modifiers}"
    return result

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Deploy shards to Fuseki
# ─────────────────────────────────────────────────────────────────────────────

def _fuseki_create_dataset(name: str, retries: int = 3) -> bool:
    """Create a TDB2 dataset on Fuseki; tolerate 409 (already exists)."""
    auth = (FUSEKI_ADMIN_USER, FUSEKI_ADMIN_PASS)
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                f"{FUSEKI_URL}/$/datasets",
                data={"dbName": name, "dbType": "tdb2"},
                auth=auth,
                timeout=30,
            )
            if r.status_code in (200, 201, 409):
                return True
            print(f"    WARNING create {name}: HTTP {r.status_code}")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"    WARNING create {name}: connection error (attempt {attempt}/{retries}): {e}")
            if attempt == retries:
                return False
        except requests.exceptions.Timeout:
            print(f"    WARNING create {name}: timeout (attempt {attempt}/{retries})")
            if attempt == retries:
                return False
    return False


def deploy_to_fuseki(shard_info: dict, retries: int = 3) -> bool:
    name = shard_info["endpoint_name"]
    auth = (FUSEKI_ADMIN_USER, FUSEKI_ADMIN_PASS)

    if not _fuseki_create_dataset(name):
        return False

    for attempt in range(1, retries + 1):
        try:
            with open(shard_info["path"], "rb") as f:
                r = requests.post(
                    f"{FUSEKI_URL}/{name}/data",
                    data=f,
                    headers={"Content-Type": "text/turtle"},
                    auth=auth,
                    timeout=300,
                )
            ok = r.status_code in (200, 201)
            print(f"    {'✓' if ok else '✗ HTTP '+str(r.status_code)} {name} "
                  f"({shard_info['triple_count']:,} triples)")
            return ok
        except requests.exceptions.ConnectionError as e:
            print(f"    ✗ {name}: connection error (attempt {attempt}/{retries}): {e}")
            if attempt == retries:
                return False
        except requests.exceptions.Timeout:
            print(f"    ✗ {name}: upload timeout (attempt {attempt}/{retries})")
            if attempt == retries:
                return False
    return False


def deploy_full_kg(kg_name: str, ttl_path: Path, retries: int = 3) -> str | None:
    """Upload the complete KG TTL as a single dataset used to run original
    (non-federated) queries during validation."""
    name = f"{kg_name}__full"
    auth = (FUSEKI_ADMIN_USER, FUSEKI_ADMIN_PASS)

    if not _fuseki_create_dataset(name):
        return None

    for attempt in range(1, retries + 1):
        try:
            with open(ttl_path, "rb") as f:
                r = requests.post(
                    f"{FUSEKI_URL}/{name}/data",
                    data=f,
                    headers={"Content-Type": "text/turtle"},
                    auth=auth,
                    timeout=600,
                )
            if r.status_code in (200, 201):
                ep = f"{FUSEKI_URL}/{name}/sparql"
                print(f"    ✓ {name} (full KG for validation)")
                return ep
            print(f"    ✗ HTTP {r.status_code} {name} (full KG)")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"    ✗ {name}: connection error (attempt {attempt}/{retries}): {e}")
            if attempt == retries:
                return None
        except requests.exceptions.Timeout:
            print(f"    ✗ {name}: upload timeout (attempt {attempt}/{retries})")
            if attempt == retries:
                return None
    return None


def ensure_federation_endpoint() -> str:
    """Create (once) an empty dataset that acts as the SPARQL executor for
    federated queries."""
    _fuseki_create_dataset(FEDERATION_DATASET)
    return f"{FUSEKI_URL}/{FEDERATION_DATASET}/sparql"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Validate rewritten queries
# ─────────────────────────────────────────────────────────────────────────────

_XSD_INT_TYPES = frozenset({
    "http://www.w3.org/2001/XMLSchema#integer",
    "http://www.w3.org/2001/XMLSchema#int",
    "http://www.w3.org/2001/XMLSchema#long",
    "http://www.w3.org/2001/XMLSchema#short",
    "http://www.w3.org/2001/XMLSchema#byte",
    "http://www.w3.org/2001/XMLSchema#nonNegativeInteger",
    "http://www.w3.org/2001/XMLSchema#positiveInteger",
    "http://www.w3.org/2001/XMLSchema#unsignedLong",
    "http://www.w3.org/2001/XMLSchema#unsignedInt",
})
_XSD_DECIMAL_TYPES = frozenset({
    "http://www.w3.org/2001/XMLSchema#decimal",
    "http://www.w3.org/2001/XMLSchema#float",
    "http://www.w3.org/2001/XMLSchema#double",
})

# ── FIX 2: tolerance for floating-point aggregation mismatches ────────────────
# SUM / AVG over xsd:float or xsd:double can differ by a small epsilon between
# the full-KG endpoint and the federated endpoint because floating-point
# addition is NOT associative and Fuseki may receive values in a different
# order when they arrive from separate SERVICE responses.
# We normalise decimal values to a rounded representation so that differences
# smaller than FLOAT_TOLERANCE are treated as equal.
FLOAT_TOLERANCE = 1e-6   # relative tolerance


def _normalise_binding(var: str, cell: dict) -> tuple:
    """Return a normalised (var, value) pair from a SPARQL JSON result cell.

    Normalisation rules:
    - All XSD integer-family types → canonical integer string (no leading zeros)
    - All XSD decimal/float types  → rounded to FLOAT_TOLERANCE so that tiny
      floating-point aggregation differences do not produce false mismatches
    - URIs and plain literals      → unchanged
    """
    t     = cell.get("type", "")
    val   = cell.get("value", "")
    dtype = cell.get("datatype", "")

    if t == "typed-literal" or (t == "literal" and dtype):
        if dtype in _XSD_INT_TYPES:
            try:
                return (var, str(int(val)))
            except ValueError:
                pass
        elif dtype in _XSD_DECIMAL_TYPES:
            # Round to a fixed number of significant figures so that tiny
            # floating-point differences in aggregation order do not cause
            # spurious mismatches (e.g. 34205888.0 vs 34205890.0 from SUM).
            try:
                fval = float(val)
                if fval != 0.0:
                    import math
                    # Round to 6 significant figures
                    magnitude  = math.floor(math.log10(abs(fval)))
                    rounded    = round(fval, -int(magnitude) + 5)
                    return (var, str(rounded))
                return (var, "0.0")
            except ValueError:
                pass

    return (var, val)


def _run_sparql(endpoint: str, query: str) -> frozenset | None:
    try:
        r = requests.get(
            endpoint,
            params={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            auth=(FUSEKI_ADMIN_USER, FUSEKI_ADMIN_PASS),
            timeout=60,
        )
        if r.status_code != 200:
            preview = r.text[:200].replace("\n", " ")
            print(f"    SPARQL HTTP {r.status_code} from {endpoint} — {preview}")
            return None
        try:
            data = r.json()
        except Exception as e:
            preview = r.text[:200].replace("\n", " ")
            print(f"    SPARQL JSON parse error: {e} — body: {preview}")
            return None

        if "results" in data and "bindings" in data["results"]:
            return frozenset(
                frozenset(_normalise_binding(k, v) for k, v in row.items())
                for row in data["results"]["bindings"]
            )
        if "boolean" in data:
            return frozenset({("__ask__", str(data["boolean"]).lower())})

        print(f"    SPARQL unexpected response shape: {list(data.keys())}")
        return None

    except requests.exceptions.Timeout:
        print(f"    SPARQL timeout ({endpoint})")
        return None
    except Exception as e:
        print(f"    SPARQL error: {e}")
        return None


def _ensure_base_prefix(sparql: str, base_ns: str | None) -> str:
    """Prepend PREFIX : <base_ns> if the query uses bare :name tokens but has
    no PREFIX : declaration."""
    if base_ns is None:
        return sparql
    if re.search(r'PREFIX\s*:\s*<', sparql, re.IGNORECASE):
        return sparql
    if not re.search(r'(?<![<"/@\w]):[A-Za-z_]', sparql):
        return sparql
    return f"PREFIX : <{base_ns}>\n{sparql}"


def validate_pair(
    original_sparql: str,
    federated_sparql: str,
    original_endpoint: str,
    federation_endpoint: str,
    base_ns: str | None = None,
) -> dict:
    orig_query = _ensure_base_prefix(original_sparql, base_ns)
    orig = _run_sparql(original_endpoint,  orig_query)
    fed  = _run_sparql(federation_endpoint, federated_sparql)

    if orig is None or fed is None:
        return {"valid": False, "error": "execution_failed"}

    match = orig == fed
    result = {
        "valid":           match,
        "original_count":  len(orig),
        "federated_count": len(fed),
        "missing":         len(orig - fed),
        "extra":           len(fed  - orig),
    }
    if not match:
        result["error"] = "result_mismatch"
        sample_missing = sorted(orig - fed, key=str)[:2]
        sample_extra   = sorted(fed - orig, key=str)[:2]
        if sample_missing:
            rows = [dict(r) for r in sample_missing]
            print(f"      sample missing rows: {rows}")
        if sample_extra:
            rows = [dict(r) for r in sample_extra]
            print(f"      sample extra rows:   {rows}")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# RESUME / CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_PROGRESS_FILE = OUTPUT_DIR / "progress.json"


def _load_progress() -> dict:
    if _PROGRESS_FILE.exists():
        try:
            with open(_PROGRESS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"processed_kgs": [], "stats": {}}


def _save_progress(processed_kgs: list, stats: dict):
    _PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PROGRESS_FILE, "w") as f:
        json.dump({"processed_kgs": processed_kgs, "stats": dict(stats)}, f, indent=2)


def _save_analysis_meta(kg_name: str, analysis: dict):
    """Persist lightweight analysis metadata so the TTL need not be re-parsed."""
    shard_dir = OUTPUT_DIR / "shards" / kg_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_ns":      analysis.get("base_ns"),
        "local_to_uri": analysis.get("local_to_uri", {}),
    }
    with open(shard_dir / "analysis_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _load_kg_from_cache(kg_name: str):
    """Return (shards, analysis) loaded from disk, or None if cache is absent."""
    shard_dir   = OUTPUT_DIR / "shards" / kg_name
    meta_path   = shard_dir / "metadata.json"
    ameta_path  = shard_dir / "analysis_meta.json"
    if not meta_path.exists() or not ameta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            shards = json.load(f)
        with open(ameta_path) as f:
            ameta = json.load(f)
        analysis = {
            "graph":            None,
            "base_ns":          ameta.get("base_ns"),
            "local_to_uri":     ameta.get("local_to_uri", {}),
            "subject_to_class": {},
            "class_triples":    {},
            "untyped_triples":  [],
            "all_classes":      {k for k in shards if k != "_shared"},
        }
        return shards, analysis
    except Exception as e:
        print(f"  [cache] Could not load cache for {kg_name}: {e}")
        return None


def _save_benchmark_incremental(benchmark: list):
    bench_path = OUTPUT_DIR / "SPIDER4FedSPARQL_benchmark.json"
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)


def _save_shard_endpoints_index() -> list:
    """Build and save a flat index of every shard SPARQL endpoint."""
    shard_root = OUTPUT_DIR / "shards"
    endpoints = []

    if shard_root.exists():
        for meta_path in sorted(shard_root.glob("*/metadata.json")):
            kg_name = meta_path.parent.name
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    shards = json.load(f)
            except Exception as e:
                print(f"  [warn] Cannot read {meta_path}: {e}")
                continue

            for class_uri, info in shards.items():
                endpoints.append({
                    "kg_name": kg_name,
                    "class": class_uri,
                    "class_name": info.get("class_name"),
                    "endpoint_name": info.get("endpoint_name"),
                    "endpoint_url": info.get("endpoint_url"),
                })

    out_path = OUTPUT_DIR / "shard_endpoints.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(endpoints, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(endpoints):,} shard endpoints → {out_path}")
    return endpoints


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def redeploy_shards(validate: bool = False):
    """Re-upload every cached shard to Fuseki without re-running sharding or rewriting.
    Use this when Fuseki was restarted and datasets are empty."""
    shard_root = OUTPUT_DIR / "shards"
    if not shard_root.exists():
        print("No shard cache found. Run with --run --deploy first.")
        return

    all_pairs  = load_pairs(TRAIN_DIR, "train") + load_pairs(DEV_DIR, "dev")
    kg_names   = sorted({p["kg_name"] for p in all_pairs})

    if validate:
        ensure_federation_endpoint()

    deployed = 0
    for kg_name in kg_names:
        shard_dir  = shard_root / kg_name
        meta_path  = shard_dir / "metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            shards = json.load(f)

        print(f"\n── {kg_name}  ({len(shards)} shards)")
        for s in shards.values():
            if deploy_to_fuseki(s):
                deployed += 1

        if validate:
            ttl_path = find_ttl(kg_name)
            if ttl_path:
                deploy_full_kg(kg_name, ttl_path)

    _save_shard_endpoints_index()
    print(f"\n[redeploy] {deployed} shards uploaded.")


def run_pipeline(deploy: bool = False, validate: bool = False, reset: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Reset / resume ────────────────────────────────────────────────────────
    if reset and _PROGRESS_FILE.exists():
        _PROGRESS_FILE.unlink()
        print("[reset] Progress cleared — starting from scratch")

    progress      = _load_progress()
    processed_set = set(progress.get("processed_kgs", []))

    benchmark: list = []
    bench_path = OUTPUT_DIR / "SPIDER4FedSPARQL_benchmark.json"
    if not reset and bench_path.exists() and processed_set:
        try:
            with open(bench_path) as f:
                benchmark = json.load(f)
            print(f"[resume] {len(benchmark)} existing entries, "
                  f"{len(processed_set)} KGs already done")
        except Exception:
            benchmark = []

    stats: defaultdict = defaultdict(int)
    if not reset:
        for k, v in progress.get("stats", {}).items():
            stats[k] = v

    all_pairs = load_pairs(TRAIN_DIR, "train") + load_pairs(DEV_DIR, "dev")
    kg_names  = sorted({p["kg_name"] for p in all_pairs})
    print(f"\nUnique KGs: {len(kg_names)}")

    pairs_by_kg = defaultdict(list)
    for pair in all_pairs:
        pairs_by_kg[pair["kg_name"]].append(pair)

    stats["total"] = len(all_pairs)

    federation_ep: str | None = None
    if validate:
        federation_ep = ensure_federation_endpoint()
        print(f"  Federation executor: {federation_ep}")

    for kg_name in kg_names:
        # ── Skip KGs already fully processed ─────────────────────────────────
        if kg_name in processed_set:
            print(f"  [skip] {kg_name} (already processed)")
            continue

        ttl_path = find_ttl(kg_name)
        if ttl_path is None:
            print(f"  SKIP: no .ttl for '{kg_name}'")
            stats["no_kg"] += len(pairs_by_kg[kg_name])
            processed_set.add(kg_name)
            _save_progress(list(processed_set), stats)
            continue

        print(f"\n── {kg_name}")
        ttl_mb = ttl_path.stat().st_size / (1024 * 1024)

        # ── Try disk cache first ──────────────────────────────────────────────
        cached = _load_kg_from_cache(kg_name)
        if cached is not None:
            shards, analysis = cached
            print(f"  [cache] {len(shards)} shards loaded from disk")
        elif ttl_mb > STREAMING_THRESHOLD_MB:
            print(f"  ({ttl_mb:.0f} MB — streaming path)")
            result = analyze_and_shard_streaming(ttl_path, kg_name)
            if result is None:
                stats["no_kg"] += len(pairs_by_kg[kg_name])
                processed_set.add(kg_name)
                _save_progress(list(processed_set), stats)
                continue
            analysis = result
            shards   = result["shards"]
            _save_analysis_meta(kg_name, analysis)
        else:
            try:
                analysis = analyze_kg(ttl_path)
            except Exception as e:
                print(f"  SKIP: parse error: {e}")
                stats["no_kg"] += len(pairs_by_kg[kg_name])
                processed_set.add(kg_name)
                _save_progress(list(processed_set), stats)
                continue
            if not analysis["all_classes"]:
                print(f"  SKIP: no classes found")
                stats["no_kg"] += len(pairs_by_kg[kg_name])
                del analysis
                processed_set.add(kg_name)
                _save_progress(list(processed_set), stats)
                continue
            shards = shard_kg(analysis, kg_name)
            _save_analysis_meta(kg_name, analysis)

        full_kg_ep: str | None = f"{FUSEKI_URL}/{kg_name}__full/sparql"
        if deploy:
            print(f"  Deploying {len(shards)} shards...")
            for s in shards.values():
                deploy_to_fuseki(s)
            if validate:
                deploy_full_kg(kg_name, ttl_path)

        for pair in pairs_by_kg[kg_name]:
            if not _sparql_is_well_formed(pair["sparql"]):
                stats["bad_sparql"] += 1
                continue

            federated = rewrite_as_federated(pair["sparql"], shards, analysis)
            if federated is None:
                local_names, _ = _extract_local_class_names(pair["sparql"])
                stats["no_class" if not local_names else "no_shard"] += 1
                continue

            stats["rewritten"] += 1

            entry = {
                "id":               f"{kg_name}_{stats['rewritten']}",
                "kg_name":          kg_name,
                "split":            pair["split"],
                "question":         pair["question"],
                "original_sparql":  pair["sparql"],
                "federated_sparql": federated,
                "full_sparql_url":  full_kg_ep,
                "endpoints": [
                    {"class": k, "url": v["endpoint_url"]}
                    for k, v in shards.items()
                ],
            }

            if validate:
                if full_kg_ep is None or federation_ep is None:
                    val_result = {"valid": False, "error": "endpoints_not_deployed"}
                    stats["validation_error"] += 1
                else:
                    val_result = validate_pair(
                        original_sparql=pair["sparql"],
                        federated_sparql=federated,
                        original_endpoint=full_kg_ep,
                        federation_endpoint=federation_ep,
                        base_ns=analysis.get("base_ns"),
                    )
                    if val_result.get("error") == "execution_failed":
                        stats["validation_error"] += 1
                    elif val_result["valid"]:
                        stats["validation_ok"] += 1
                    else:
                        stats["validation_fail"] += 1

                entry["validation"] = val_result
                status = (
                    "✓" if val_result.get("valid")
                    else ("ERR" if val_result.get("error") == "execution_failed" else "✗")
                )
                print(
                    f"    [{status}] {pair['question'][:60]}"
                    + (
                        f"  orig={val_result.get('original_count')}  "
                        f"fed={val_result.get('federated_count')}"
                        if "original_count" in val_result else ""
                    )
                )

            benchmark.append(entry)

        del analysis

        # ── Checkpoint after each KG ──────────────────────────────────────────
        processed_set.add(kg_name)
        _save_progress(list(processed_set), stats)
        _save_benchmark_incremental(benchmark)
        print(f"  [checkpoint] {kg_name} done "
              f"({len(processed_set)}/{len(kg_names)} KGs)")

    def _save(data, fname):
        path = OUTPUT_DIR / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data):,} entries → {path}")

    _save(benchmark, "SPIDER4FedSPARQL_benchmark.json")
    _save([e for e in benchmark if e["split"] == "train"], "SPIDER4FedSPARQL_train.json")
    _save([e for e in benchmark if e["split"] == "dev"],   "SPIDER4FedSPARQL_dev.json")
    _save_shard_endpoints_index()

    print(f"\n{'='*50}")
    print(f"  Total pairs:          {stats['total']:>6}")
    print(f"  Rewritten:            {stats['rewritten']:>6}")
    print(f"  Skipped (bad SPARQL): {stats['bad_sparql']:>6}")
    print(f"  Skipped (no class):   {stats['no_class']:>6}")
    print(f"  Skipped (no shard):   {stats['no_shard']:>6}")
    print(f"  Skipped (no KG):      {stats['no_kg']:>6}")
    if validate:
        print(f"  Validation ✓:       {stats['validation_ok']:>6}")
        print(f"  Validation ✗:       {stats['validation_fail']:>6}")
        print(f"  Validation ERR:     {stats['validation_error']:>6}")
    return benchmark

# ─────────────────────────────────────────────────────────────────────────────
# INSPECTION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def inspect_pairs(n: int = 3):
    for split_dir, name in [(TRAIN_DIR, "train"), (DEV_DIR, "dev")]:
        print(f"\n── {name} ({split_dir})")
        files = sorted(split_dir.glob("**/*.json"))
        print(f"  Files: {[f.name for f in files]}")
        if not files:
            continue
        with open(files[0]) as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else [data]
        for e in entries[:n]:
            print(f"  keys:     {list(e.keys())}")
            print(f"  kg_name:  {e.get('kg_name', e.get('db_id', '?'))}")
            print(f"  question: {(e.get('question') or e.get('nl', '?'))[:80]}")
            q = e.get("query") or e.get("sparql") or ""
            print(f"  sparql:   {q[:80]}")
            print()


def inspect_ttl(kg_name: str):
    ttl = find_ttl(kg_name)
    if ttl is None:
        print(f"No .ttl for '{kg_name}'")
        return
    analysis = analyze_kg(ttl)
    print(f"\nKG: {kg_name}  |  {ttl}")
    print(f"Base NS: {analysis['base_ns']}  |  Triples: {len(analysis['graph'])}")
    print(f"Classes ({len(analysis['all_classes'])}):")
    for cls in sorted(analysis["all_classes"], key=str):
        print(f"  {cls}  →  {len(analysis['class_triples'][cls])} triples")
    print(f"Local name map:")
    for local, uri in sorted(analysis["local_to_uri"].items()):
        print(f"  :{local}  →  {uri}")


def show_rewrite(kg_name: str, sparql: str):
    ttl = find_ttl(kg_name)
    if ttl is None:
        print(f"No .ttl for '{kg_name}'")
        return
    analysis = analyze_kg(ttl)
    shards = {
        str(cls): {
            "endpoint_url": f"{FUSEKI_URL}/{kg_name}__{_safe_name(cls)}/sparql",
            "class_name":   _safe_name(cls),
        }
        for cls in analysis["all_classes"]
    }
    print(f"\nORIGINAL:\n{sparql}\n\nFEDERATED:")
    result = rewrite_as_federated(sparql, shards, analysis)
    print(result or "Could not rewrite (no class info matched any shard, or only 1 endpoint involved)")


def show_sample_rewrites(kg_name: str, n: int = 3):
    all_pairs = load_pairs(TRAIN_DIR, "train") + load_pairs(DEV_DIR, "dev")
    pairs = [p for p in all_pairs if p["kg_name"] == kg_name]
    if not pairs:
        print(f"No pairs found for kg_name='{kg_name}'")
        return
    ttl = find_ttl(kg_name)
    if ttl is None:
        print(f"No .ttl for '{kg_name}'")
        return
    analysis = analyze_kg(ttl)
    shards = {
        str(cls): {
            "endpoint_url": f"{FUSEKI_URL}/{kg_name}__{_safe_name(cls)}/sparql",
            "class_name":   _safe_name(cls),
        }
        for cls in analysis["all_classes"]
    }
    shown = 0
    for pair in pairs:
        result = rewrite_as_federated(pair["sparql"], shards, analysis)
        if result:
            print(f"\n{'─'*60}")
            print(f"Question: {pair['question']}")
            print(f"\nORIGINAL:\n{pair['sparql']}\n\nFEDERATED:\n{result}")
            shown += 1
        if shown >= n:
            break
    if shown == 0:
        print(f"No rewritable queries found for '{kg_name}' (all touch only 1 endpoint)")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Spider4SPARQL class sharding pipeline")
    p.add_argument("--inspect-pairs",  action="store_true")
    p.add_argument("--inspect-ttl",    type=str, metavar="KG_NAME")
    p.add_argument("--show-rewrite",   type=str, metavar="KG_NAME")
    p.add_argument("--sparql",         type=str)
    p.add_argument("--run",            action="store_true")
    p.add_argument("--deploy",         action="store_true")
    p.add_argument("--redeploy",       action="store_true",
                   help="Re-upload all cached shards to Fuseki (skips rewriting/benchmark)")
    p.add_argument("--validate",       action="store_true")
    p.add_argument("--reset",          action="store_true",
                   help="Clear saved progress and restart from scratch")
    args = p.parse_args()

    if args.inspect_pairs:
        inspect_pairs()
    if args.inspect_ttl:
        inspect_ttl(args.inspect_ttl)
    if args.show_rewrite:
        if args.sparql:
            show_rewrite(args.show_rewrite, args.sparql)
        else:
            show_sample_rewrites(args.show_rewrite)
    if args.redeploy:
        redeploy_shards(validate=args.validate)
    if args.run:
        run_pipeline(deploy=args.deploy, validate=args.validate, reset=args.reset)