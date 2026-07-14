import re
from typing import Any, Optional

# SPARQL keywords that must not appear as standalone words inside a triple pattern token.
_SPARQL_KEYWORDS = re.compile(
    r"\b(FILTER|OPTIONAL|UNION|BIND|VALUES|SELECT|WHERE|GRAPH|SERVICE|GROUP|HAVING|ORDER|LIMIT|OFFSET)\b",
    re.IGNORECASE,
)

def _validate_token(key: str, val: Any):
    """Return an error string if *val* is not a valid single triple-pattern token, else None."""
    if not isinstance(val, str):
        return (
            f"Pattern '{key}' must be a single string token, got {type(val).__name__}: {val!r}. "
            "Use one triple pattern per subject/predicate/object combination."
        )
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

def _schema_contains_endpoint(schema_summary: Any, url: str) -> bool:
    if isinstance(schema_summary, dict):
        return url in schema_summary
    if isinstance(schema_summary, str):
        return url in schema_summary
    return False

def _validate_string_list(ir: dict, field: str) -> tuple[bool, Optional[str]]:
    value = _list_field(ir, field)
    if not isinstance(value, list):
        return False, f"IR field '{field}' must be a list, got {type(value).__name__}: {value!r}."
    for item in value:
        if not isinstance(item, str):
            return False, f"IR field '{field}' must contain only strings, got {type(item).__name__}: {item!r}."
    return True, None

def _list_field(ir: dict, field: str) -> Any:
    value = ir.get(field, [])
    return [] if value is None else value

def _endpoint_patterns(endpoint: dict) -> Any:
    patterns = endpoint.get("patterns", [])
    return [] if patterns is None else patterns

def validate_ir(ir: dict, schema_summary: dict) -> tuple[bool, Optional[str]]:
    """Validate the intermediate representation (IR) of a SPARQL query against the schema summary of the endpoints."""
    if not isinstance(ir, dict):
        return False, f"IR must be a dict, got {type(ir).__name__}: {ir!r}."

    endpoints = ir.get("endpoints")
    if not endpoints:
        return False, "IR must contain at least one endpoint."
    if not isinstance(endpoints, list):
        return False, f"IR field 'endpoints' must be a list, got {type(endpoints).__name__}: {endpoints!r}."

    for field in ("select", "filters", "group_by", "having", "join_variables"):
        is_valid, err = _validate_string_list(ir, field)
        if not is_valid:
            return False, err

    all_pattern_vars = set()
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            return False, f"Each endpoint must be a dict, got {type(endpoint).__name__}: {endpoint!r}."
        url = endpoint.get("url")
        if not isinstance(url, str):
            return False, f"Endpoint 'url' must be a string, got {type(url).__name__}: {url!r}."
        if not _schema_contains_endpoint(schema_summary, url):
            return False, f"Endpoint {url} in IR is not in the schema summary." # Hallucinated endpoint

        patterns = _endpoint_patterns(endpoint)
        if not isinstance(patterns, list):
            return False, f"Endpoint {url} field 'patterns' must be a list, got {type(patterns).__name__}: {patterns!r}."

        for pattern in patterns:
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

    for select_var in _list_field(ir, "select"):
        # Aggregate expressions like "(COUNT(?x) AS ?alias)" are not pattern vars — skip them.
        if select_var.startswith("("):
            continue
        if select_var not in all_pattern_vars:
            return False, f"Selected variable {select_var} is not used in any triple pattern." # Unused variable

    # If GROUP BY is present, every non-aggregate SELECT variable must appear in it.
    # Variables inside non-aggregate expressions (COALESCE, CONCAT, IF, …) must also be in GROUP BY.
    _AGGREGATE_EXPR = re.compile(
        r'^\(\s*(COUNT|SUM|AVG|MIN|MAX|SAMPLE|GROUP_CONCAT)\s*\(', re.IGNORECASE
    )
    group_by_vars = _list_field(ir, "group_by")
    if group_by_vars:
        for select_var in _list_field(ir, "select"):
            if not select_var.startswith("("):
                # Plain variable
                if select_var not in group_by_vars:
                    return False, (
                        f"Selected variable {select_var} is not in GROUP BY. "
                        "Add it to 'group_by' or wrap it in an aggregate expression."
                    )
            elif not _AGGREGATE_EXPR.match(select_var):
                # Non-aggregate expression like (COALESCE(...) AS ?alias) —
                # every ?var referenced inside must be in GROUP BY (excluding the alias itself).
                as_match = re.search(r'\bAS\s+\?(\w+)\s*\)\s*$', select_var, re.IGNORECASE)
                alias = f"?{as_match.group(1)}" if as_match else None
                for var_name in re.findall(r'\?(\w+)', select_var):
                    full_var = f"?{var_name}"
                    if full_var == alias:
                        continue
                    if full_var not in group_by_vars:
                        return False, (
                            f"Variable {full_var} inside a non-aggregate SELECT expression "
                            f"({select_var!r}) is not in GROUP BY. "
                            "Add it to 'group_by', or use SAMPLE(?var) to pick an arbitrary value per group."
                        )

    for f in _list_field(ir, "filters"):
        if re.match(r'^FILTER\s*\(', f.strip(), re.IGNORECASE):
            return False, (
                f"Filter expression already contains a FILTER() wrapper: {f!r}. "
                "The 'filters' array must contain only the inner expression (e.g. '?x = \"value\"'), "
                "not the full FILTER(...) clause — the compiler adds that automatically."
            )

    for join_var in _list_field(ir, "join_variables"):
        appearances = sum(
            1 for ep in ir["endpoints"]
            if any(isinstance(p, dict) and join_var in [p.get("subject"), p.get("object")] for p in _endpoint_patterns(ep))
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
    select_clause = "SELECT " + distinct + " ".join(_list_field(ir, "select"))
    service_clauses = []

    for endpoint in ir.get("endpoints", []):
        patterns = _endpoint_patterns(endpoint)
        if patterns:
            url = endpoint['url'].strip("<>")
            triples = " . ".join(f"{_sparql_term(p['subject'])} {_sparql_term(p['predicate'])} {_sparql_term(p['object'])}" for p in patterns)
            service_clauses.append(f"SERVICE <{url}> {{ {triples} }}")

    filters = "\n  ".join(_build_filter(f) for f in _list_field(ir, "filters"))
    group_by_vars = _list_field(ir, "group_by")
    having_exprs = _list_field(ir, "having")
    group_by = f"GROUP BY {' '.join(group_by_vars)}" if group_by_vars else ""
    having = "HAVING (" + " && ".join(having_exprs) + ")" if having_exprs else ""
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
