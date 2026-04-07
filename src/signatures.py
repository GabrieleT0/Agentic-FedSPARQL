import dspy
from typing import Literal

class EvaluateEndpoints(dspy.Signature):
    """Given a natural language question, a set of new SPARQL endpoint descriptions to evaluate, and the endpoints
    already selected in previous iterations, determine which new endpoints are relevant and whether the full set
    (already selected + newly relevant) is now sufficient to answer the question completely.
    If some required data still appears to be missing, mark as insufficient so more endpoints can be retrieved."""
    question: str = dspy.InputField()
    already_selected_descriptions: str = dspy.InputField(desc='JSON: {endpoint_url: {classes: [...], properties: [...]}} — endpoints already chosen in previous batches. Empty object if this is the first batch.')
    candidate_descriptions: str = dspy.InputField(desc='JSON: {endpoint_url: {classes: [...], properties: [...]}} — new endpoints to evaluate in this batch.')
    relevant_endpoints: list = dspy.OutputField(desc="List of endpoint URLs from candidate_descriptions that are relevant to answering the question. Empty list if none are relevant.")
    is_sufficient: bool = dspy.OutputField(desc="True if already_selected + relevant_endpoints collectively contain all data needed to fully answer the question. False if more endpoints are likely needed.")

class FilterSchema(dspy.Signature):
    """Given in input a natural language question, and different VoID descriptions for the SPARQL endpoint that must contain the answers, create
       a JSON string with the relevant classes, properties, sample values, and join keys that are needed to answer the question for each endpoint."""
    question: str = dspy.InputField()
    void_descriptions: str = dspy.InputField(desc='JSON string: {endpoint: {classes: [...], properties: [...], sample_values: {property_uri: "example_value"}, join_keys: [{property: "", links_to_class: ""}]}}')
    schema_summary: str = dspy.OutputField(desc='Same structure as input: {endpoint: {classes: [...], properties: [...], sample_values: {property_uri: "example_value"}, join_keys: [{property: "", links_to_class: ""}]}} but only containing entries relevant to the question')

class IdentifyJoins(dspy.Signature):
    """Given a natural language question and the relevant schema for each SPARQL endpoint (including join_keys that show which properties link to instances of other classes),
       identify pairs of endpoints that can be joined and the SPARQL variable that would link them.
       Use the join_keys field to find properties whose objects are instances of a class present in another endpoint."""
    question: str = dspy.InputField()
    schema_summary: str = dspy.InputField(desc='JSON string: {endpoint: {classes: [...], properties: [...], sample_values: {property_uri: "example_value"}, join_keys: [{property: "", links_to_class: ""}]}}')
    join_candidates: str = dspy.OutputField(desc='JSON string structured as: [{"endpoint_a": "", "endpoint_b": "", "property_a": "", "property_b": "", "join_variable": ""}...]')

class QueryBuilder(dspy.Signature):
    """Given a natural language question, the relevant schema for each SPARQL endpoint (including sample values showing how data is stored),
       and the identified join candidates, create a JSON query plan for a federated SPARQL query.

       Rules:
       - Use SERVICE clauses for every endpoint whose data is needed.
       - If join_candidates is non-empty, always produce a multi-endpoint federated query using the join variables.
       - When the question mentions specific names, values, or entities, always add FILTER clauses using the sample_values
         to understand the correct property and value format (e.g. FILTER(?fname = "Michael" && ?lname = "Goodrich")).
       - Use full URIs for all classes and properties (no angle brackets in the JSON values).
       - Declare all namespace prefixes used."""
    question: str = dspy.InputField()
    schema_summary: str = dspy.InputField(desc='JSON string: {endpoint: {classes: [...], properties: [...], sample_values: {property_uri: "example_value"}, join_keys: [{property: "", links_to_class: ""}]}}')
    join_candidates: str = dspy.InputField(desc='JSON string structured as: [{"endpoint_a": "", "endpoint_b": "", "property_a": "", "property_b": "", "join_variable": ""}...]')
    previous_error: str = dspy.InputField(desc="Error message from the previous execution of the generated SPARQL query, if any. \"none\" otherwise.")
    query_plan: str = dspy.OutputField(desc="""{"prefixes": {"foaf": "http://xmlns.com/foaf/0.1/", "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
                                        "select": ["?var1", "?var2"],
                                        "distinct": false,
                                        "endpoints": [{"url": "...", "patterns": [{"subject": "", "predicate": "", "object": ""}]}],
                                        "join_variables": ["?var"],
                                        "filters": ["?fname = \\"Michael\\" && ?lname = \\"Goodrich\\""],
                                        "group_by": ["?var1"],
                                        "having": ["COUNT(*) >= 2"],
                                        "order_by": null,
                                        "limit": null
                                        }""")

class DiagnoseEmptyResult(dspy.Signature):
    """
    Given in input a natural language question, the generated SPARQL query, and the results obtained by probing the 
    SPARQL endpoint to check if the different endpoints contain the relevant data, diagnose whether the empty result is due to wrong endpoints (i.e., the SPARQL query is correct but the endpoints do not contain the relevant data) 
    or to a wrong schema (i.e., the SPARQL query is not correctly formulated according to the actual schema of the endpoints). In the case of wrong endpoints, returns also the list of endpoints that are likely wrong and should be replaced in the next discovery iteration.
    """
    question: str = dspy.InputField()
    sparql_query: str = dspy.InputField()
    probe_results: str = dspy.InputField(desc='{"endpoint_url": {"classes": {"ClassName": 42}, "properties": {"propName": true}}}')
    diagnosis: Literal['wrong_endpoints', 'wrong_schema'] = dspy.OutputField()
    wrong_endpoints: list = dspy.OutputField(desc="List of objects [{\"url\": \"http://...\", \"confidence\": 0.95}] for endpoints likely wrong and to be replaced in the next discovery iteration. confidence is a float 0.0–1.0 reflecting how clearly the probe results support excluding that endpoint. Empty list if diagnosis is 'wrong_schema'.")