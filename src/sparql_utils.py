import time
import requests
from functools import lru_cache

FEDERATED_ENDPOINT = "http://host.docker.internal:3030/fkgqa_federation/sparql"
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def strip_sparql_code_fence(query: str) -> str:
    """Remove a wrapping SPARQL markdown code fence when an LLM includes one."""
    if not isinstance(query, str):
        return query

    stripped = query.strip()
    for fence in ("```", "'''"):
        opening = f"{fence}sparql"
        if stripped.lower().startswith(opening) and stripped.endswith(fence):
            inner = stripped[len(opening):-len(fence)]
            return inner.strip()
    return query


class SPARQLExecutionError(Exception):
    """Raised when a SPARQL endpoint returns an error response (e.g. 400 parse error)."""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


def execute_sparql_on_endpoint(endpoint_url: str, query: str, retries: int = 5, retry_delay: float = 15.0) -> list:
    """Execute a SPARQL query on a given endpoint and return the results.

    Returns an empty list when the query succeeds but has no results.
    Raises SPARQLExecutionError on HTTP errors (e.g. 400 parse errors).
    Retries on transient transport and server errors which can occur with
    federated queries that trigger sub-queries to remote endpoints.
    """
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    last_exc = None
    for attempt in range(retries):
        try:
            response = requests.post(
                endpoint_url,
                data={"query": query},
                headers=headers,
                timeout=1000
            )
            response.raise_for_status()
            response_json = response.json()
            bindings = response_json.get("results", {}).get("bindings", [])
            return [{k: v["value"] for k, v in row.items()} for row in bindings]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            if attempt < retries - 1:
                print(
                    f"Transient SPARQL error on attempt {attempt + 1}/{retries}, "
                    f"retrying in {retry_delay}s: {e}"
                )
                time.sleep(retry_delay)
                continue
        except requests.exceptions.RequestException as e:
            resp = getattr(e, 'response', None)
            status = resp.status_code if resp is not None else None
            text = resp.text if resp is not None else str(e)
            if status in TRANSIENT_STATUS_CODES:
                last_exc = e
                if attempt < retries - 1:
                    print(
                        f"Transient SPARQL HTTP error {status} on attempt {attempt + 1}/{retries}, "
                        f"retrying in {retry_delay}s."
                    )
                    time.sleep(retry_delay)
                    continue
            print(f"Error executing SPARQL query on endpoint {endpoint_url}: {e}")
            print(f"Response status: {status if status is not None else 'N/A'}")
            print(f"Response text: {text}")
            raise SPARQLExecutionError(text, status_code=status) from e

    resp = getattr(last_exc, 'response', None)
    status = resp.status_code if resp is not None else None
    text = resp.text if resp is not None else str(last_exc)
    print(f"Error executing SPARQL query on endpoint {endpoint_url}: {last_exc}")
    print(f"Response status: {status if status is not None else 'N/A'}")
    print(f"Response text: {text}")
    raise SPARQLExecutionError(text, status_code=status) from last_exc


@lru_cache(maxsize=32)
def get_void_description(endpoint_url: str) -> dict:
    """Get the VoID description of a SPARQL endpoint.

    Result is cached per endpoint URL so Fuseki is only hit once per
    endpoint regardless of how many times this is called across agent turns.
    This prevents read-lock accumulation that causes HTTP 500 errors.
    """
    schema_query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?class ?property WHERE {
            { ?s a ?class }
            UNION
            { ?s ?property ?o
            FILTER(?property != rdf:type) }
        } LIMIT 100
    """
    results = execute_sparql_on_endpoint(endpoint_url, schema_query)
    classes = []
    properties = []
    for result in results:
        if "class" in result:
            classes.append(result["class"])
        if "property" in result:
            properties.append(result["property"])

    # Small delay between queries to let Fuseki release read locks
    time.sleep(0.3)

    sample_query = """
        SELECT ?property (SAMPLE(?o) AS ?sample) WHERE {
            ?s ?property ?o .
            FILTER(?property != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
            FILTER(isLiteral(?o))
        } GROUP BY ?property LIMIT 100
    """
    sample_results = execute_sparql_on_endpoint(endpoint_url, sample_query)
    sample_values = {
        r["property"]: r["sample"]
        for r in sample_results
        if "property" in r and "sample" in r
    }

    time.sleep(0.3)

    join_query = """
        SELECT DISTINCT ?property ?objectClass WHERE {
            ?s ?property ?o .
            ?o a ?objectClass .
            FILTER(?property != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
        } LIMIT 100
    """
    join_results = execute_sparql_on_endpoint(endpoint_url, join_query)
    join_keys = [
        {"property": r["property"], "links_to_class": r["objectClass"]}
        for r in join_results
        if "property" in r and "objectClass" in r
    ]

    return {
        "endpoint": endpoint_url,
        "classes": classes,
        "properties": properties,
        "sample_values": sample_values,
        "join_keys": join_keys,
    }


def probe_class(endpoint_url: str, class_uri: str) -> int:
    """Probe a SPARQL endpoint to check how many instances of a given class it contains."""
    class_uri = class_uri.strip("<>")
    query = f"""
        SELECT (COUNT(?s) AS ?count) WHERE {{
            ?s a <{class_uri}> .
        }}
    """
    try:
        results = execute_sparql_on_endpoint(endpoint_url, query)
    except SPARQLExecutionError:
        return 0
    if results and "count" in results[0]:
        return int(results[0]["count"])
    return 0


def probe_property(endpoint_url: str, property_uri: str) -> list:
    """Probe a SPARQL endpoint to check if it contains a given property."""
    property_uri = property_uri.strip("<>")
    query = f"""
        SELECT DISTINCT ?val WHERE {{
            ?s <{property_uri}> ?val .
        }} LIMIT 10
    """
    try:
        return execute_sparql_on_endpoint(endpoint_url, query)
    except SPARQLExecutionError:
        return []


def execute_sparql_query(query: str) -> list:
    """Execute a federated SPARQL query on the local Fuseki federation endpoint.

    Returns an empty list on empty results.
    Raises SPARQLExecutionError on HTTP errors.
    """
    return execute_sparql_on_endpoint(FEDERATED_ENDPOINT, query)
