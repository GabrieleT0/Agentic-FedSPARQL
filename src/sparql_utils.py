import requests

FEDERATED_ENDPOINT = "http://host.docker.internal:3030/fkgqa_federation/sparql"

def execute_sparql_on_endpoint(endpoint_url: str, query: str) -> list:
    """Execute a SPARQL query on a given endpoint and return the results."""
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
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
    except requests.exceptions.RequestException as e:
        print(f"Error executing SPARQL query on endpoint {endpoint_url}: {e}")
        print(f"Response status: {e.response.status_code if getattr(e, 'response', None) is not None else 'N/A'}")
        print(f"Response text: {e.response.text if getattr(e, 'response', None) is not None else 'N/A'}")
        return []
    
def get_void_description(endpoint_url: str) -> dict:
    """Get the VoID description of a SPARQL endpoint, including sample literal values and join-key candidates."""
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

    # Sample literal values per property so the LLM can infer filter conditions
    sample_query = """
            SELECT ?property (SAMPLE(?o) AS ?sample) WHERE {
                ?s ?property ?o .
                FILTER(?property != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
                FILTER(isLiteral(?o))
            } GROUP BY ?property LIMIT 100
        """
    sample_results = execute_sparql_on_endpoint(endpoint_url, sample_query)
    sample_values = {r["property"]: r["sample"] for r in sample_results if "property" in r and "sample" in r}

    # Detect join-key candidates: properties whose objects are instances of a class
    join_query = """
            SELECT DISTINCT ?property ?objectClass WHERE {
                ?s ?property ?o .
                ?o a ?objectClass .
                FILTER(?property != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
            } LIMIT 50
        """
    join_results = execute_sparql_on_endpoint(endpoint_url, join_query)
    join_keys = [{"property": r["property"], "links_to_class": r["objectClass"]} for r in join_results if "property" in r and "objectClass" in r]

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
    results = execute_sparql_on_endpoint(endpoint_url, query)
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
    results = execute_sparql_on_endpoint(endpoint_url, query)
    return results

def execute_sparql_query(query: str) -> list:
    """Execute a federated SPARQL query on the local Fuseki federation endpoint."""
    return execute_sparql_on_endpoint(FEDERATED_ENDPOINT, query)