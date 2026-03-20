import requests

def execute_sparql_on_endpoint(endpoint_url: str, query: str) -> list:
    """Execute a SPARQL query on a given endpoint and return the results."""
    headers = {
        "Accept": "application/sparql-results+json"
    }
    try:
        response = requests.post(endpoint_url, data={"query": query}, headers=headers, timeout=1000)
        response.raise_for_status()
        response_json = response.json()
        bindings = response_json.get("results", {}).get("bindings", [])

        return [{k: v["value"] for k, v in row.items()} for row in bindings]
    except requests.exceptions.RequestException as e:
        print(f"Error executing SPARQL query on endpoint {endpoint_url}: {e}")
        return []
    
def get_void_description(endpoint_url: str) -> dict:
    """Get the VoID description of a SPARQL endpoint."""
    query = """
            SELECT DISTINCT ?class ?property WHERE {
                { ?s a ?class }
                UNION
                { ?s ?property ?o 
                FILTER(?property != rdf:type) }
            } LIMIT 100
        """
    results = execute_sparql_on_endpoint(endpoint_url, query)
    classes = list()
    properties = list()
    for result in results:
        if "class" in result:
            classes.append(result["class"])
        if "property" in result:
            properties.append(result["property"])
    return {"endpoint": endpoint_url, "classes": classes, "properties": properties}

def probe_class(endpoint_url: str, class_uri: str) -> int:
    """Probe a SPARQL endpoint to check how many instances of a given class it contains."""
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
    query = f"""
                SELECT DISTINCT ?val WHERE {{
                    ?s <{property_uri}> ?val .
                }} LIMIT 10
        """
    results = execute_sparql_on_endpoint(endpoint_url, query)
    return results