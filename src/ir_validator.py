import re
from typing import Optional

# SPARQL keywords that must not appear as standalone words inside a triple pattern token.
_SPARQL_KEYWORDS = re.compile(
    r"\b(FILTER|OPTIONAL|UNION|BIND|VALUES|SELECT|WHERE|GRAPH|SERVICE|GROUP|HAVING|ORDER|LIMIT|OFFSET)\b",
    re.IGNORECASE,
)

def _validate_token(key: str, val: str):
    """Return an error string if *val* is not a valid single triple-pattern token, else None."""
    if "[" in val or "]" in val:
        return (
            f"Pattern '{key}' contains blank node syntax which is not allowed in the IR: {val!r}. "
            "Use a named variable (e.g. ?bn) instead of anonymous blank nodes."
        )
    # String literals may contain anything; skip further checks.
    if val.startswith('"') or val.startswith("'"):
        return None
    # Non-literal tokens must be a single term — no embedded whitespace.
    if re.search(r"\s", val):
        return (
            f"Pattern '{key}' contains whitespace, which indicates an embedded SPARQL clause: {val!r}. "
            "Each subject/predicate/object must be a single token (variable, URI, prefixed name, or literal). "
            "Move FILTER expressions to the top-level 'filters' array."
        )
    # No SPARQL keywords embedded in a bare token.
    m = _SPARQL_KEYWORDS.search(val)
    if m:
        return (
            f"Pattern '{key}' contains SPARQL keyword '{m.group()}' which must not appear inside a pattern token: {val!r}. "
            "Move FILTER/OPTIONAL/etc. to the appropriate top-level IR field."
        )
    return None

def validate_ir(ir: dict, schema_summary: dict) -> tuple[bool, Optional[str]]:
    """Validate the intermediate representation (IR) of a SPARQL query against the schema summary of the endpoints."""
    if not ir.get("endpoints"):
        return False, "IR must contain at least one endpoint."

    all_pattern_vars = set()
    for endpoint in ir["endpoints"]:
        url = endpoint.get("url")
        if not url in schema_summary:
            return False, f"Endpoint {url} in IR is not in the schema summary." # Hallucinated endpoint

        for pattern in endpoint.get("patterns", []):
            if not isinstance(pattern, dict):
                return False, "Each pattern must be a dict with 'subject', 'predicate', and 'object' keys."
            if not all(k in pattern for k in ("subject", "predicate", "object")):
                return False, f"Pattern is missing required keys ('subject', 'predicate', 'object'): {pattern}"
            for key in ("subject", "predicate", "object"):
                err = _validate_token(key, pattern[key])
                if err:
                    return False, err
            for token in [pattern["subject"], pattern["object"]]:
                if token.startswith("?"):
                    all_pattern_vars.add(token)

    for select_var in ir.get("select", []):
        # Aggregate expressions like "(COUNT(?x) AS ?alias)" are not pattern vars — skip them.
        if select_var.startswith("("):
            continue
        if select_var not in all_pattern_vars:
            return False, f"Selected variable {select_var} is not used in any triple pattern." # Unused variable

    # If GROUP BY is present, every non-aggregate SELECT variable must appear in it.
    group_by_vars = ir.get("group_by", [])
    if group_by_vars:
        for select_var in ir.get("select", []):
            if select_var.startswith("("):
                continue
            if select_var not in group_by_vars:
                return False, (
                    f"Selected variable {select_var} is not in GROUP BY. "
                    "Add it to 'group_by' or wrap it in an aggregate expression."
                )

    for f in ir.get("filters", []):
        if re.match(r'^FILTER\s*\(', f.strip(), re.IGNORECASE):
            return False, (
                f"Filter expression already contains a FILTER() wrapper: {f!r}. "
                "The 'filters' array must contain only the inner expression (e.g. '?x = \"value\"'), "
                "not the full FILTER(...) clause — the compiler adds that automatically."
            )

    for join_var in ir.get("join_variables", []):
        appearances = sum(
            1 for ep in ir["endpoints"]
            if any(isinstance(p, dict) and join_var in [p.get("subject"), p.get("object")] for p in ep.get("patterns", []))
        )
        if appearances < 2:
            return False, f"Join variable {join_var} appears in only one endpoint."

    return True, None

def _sparql_term(token: str) -> str:
    """Normalize a token to valid SPARQL syntax.
    Variables (?x) and literals ("...") are returned as-is.
    Full URIs (http/https) are wrapped in <> if not already.
    Prefixed names (prefix:local) and keywords (a) are returned as-is.
    """
    if token.startswith("?") or token.startswith('"') or token.startswith("'"):
        return token
    if token.startswith("<") and token.endswith(">"):
        return token  # already wrapped
    if token.startswith("http://") or token.startswith("https://"):
        return f"<{token}>"
    return token  # prefixed name, keyword like 'a', or literal

def _build_filter(f: str) -> str:
    """Wrap filter expression in FILTER(), stripping any existing FILTER() wrapper the LLM may have added."""
    stripped = f.strip()
    if re.match(r'^FILTER\s*\(', stripped, re.IGNORECASE):
        # Remove outer FILTER( ... ) to avoid FILTER(FILTER(...))
        inner = re.sub(r'^FILTER\s*\(', '', stripped, flags=re.IGNORECASE)
        # Drop matching trailing paren
        if inner.endswith(")"):
            inner = inner[:-1]
        stripped = inner.strip()
    return f"FILTER({stripped})"


def compile_ir_to_sparql(ir: dict) -> str:
    """Compile the intermediate representation (IR) into a SPARQL query string."""
    prefix_clauses = "\n".join(f"PREFIX {alias}: <{uri.strip('<>')}>" for alias, uri in ir.get("prefixes", {}).items())
    distinct = "DISTINCT " if ir.get("distinct") else ""
    select_clause = "SELECT " + distinct + " ".join(ir.get("select", []))
    service_clauses = []

    for endpoint in ir.get("endpoints", []):
        patterns = endpoint.get("patterns", [])
        if patterns:
            url = endpoint['url'].strip("<>")
            triples = " . ".join(f"{_sparql_term(p['subject'])} {_sparql_term(p['predicate'])} {_sparql_term(p['object'])}" for p in patterns)
            service_clauses.append(f"SERVICE <{url}> {{ {triples} }}")

    filters = "\n  ".join(_build_filter(f) for f in ir.get("filters", []))
    group_by = f"GROUP BY {' '.join(ir['group_by'])}" if ir.get("group_by") else ""
    having = "HAVING (" + " && ".join(ir["having"]) + ")" if ir.get("having") else ""
    order_by = f"ORDER BY {ir['order_by']}" if ir.get("order_by") else ""
    limit = f"LIMIT {ir['limit']}" if ir.get("limit") else ""
    where_body = "\n  ".join(service_clauses)
    if filters:
        where_body += "\n  " + filters

    query = f"{select_clause} WHERE {{\n  {where_body}\n}}"
    if prefix_clauses:
        query = prefix_clauses + "\n" + query
    if group_by:
        query += f"\n{group_by}"
    if having:
        query += f"\n{having}"
    if order_by:
        query += f"\n{order_by}"
    if limit:
        query += f"\n{limit}"

    return query

