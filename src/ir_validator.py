from typing import Optional

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
            for token in [pattern["subject"], pattern["object"]]:
                if token.startswith("?"):
                    all_pattern_vars.add(token)

    for select_var in ir.get("select", []):
        if select_var not in all_pattern_vars:
            return False, f"Selected variable {select_var} is not used in any triple pattern." # Unused variable

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

    filters = "\n  ".join(f"FILTER({f})" for f in ir.get("filters", []))
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

