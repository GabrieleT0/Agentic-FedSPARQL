import dspy
from typing import Literal

class FilterSchema(dspy.Signature):
    """Given in input a natural language question, and different VoID descriptions for the SPARQL endpoint that must contain the answers, create
       a JSON string with the relevant classes and properties that are needed to answer the question for each endpoint."""
    question: str = dspy.InputField()
    void_descriptions: str = dspy.InputField(desc="JSON string structured as: {endpoint: {classes: [class1, class2, ...], properties: [property1, property2, ...]}}")
    schema_summary: str = dspy.OutputField(desc="Same structure as input: {endpoint: {classes: [class1, class2, ...], properties: [property1, property2, ...]}} but only containing entries relevant to the question")

class IdentifyJoins(dspy.Signature):
    """Given a natural language question and the relevant classes and properties for each SPARQL endpoint, identify 
        pairs of endpoints that can be joined and the SPARQL variable that would link them."""
    question: str = dspy.InputField()
    schema_summary: str = dspy.InputField(desc="JSON string structured as: {endpoint: {classes: [class1, class2, ...], properties: [property1, property2, ...]}}")
    join_candidates: str = dspy.OutputField(desc='JSON string structured as: [{"endpoint_a": "", "endpoint_b": "", "property_a": "", "property_b": "", "join_variable": ""}...]')

class QueryBuilder(dspy.Signature):
    """Given a natural language question, the relevant classes and properties for each
       SPARQL endpoint, and the identified join candidates, create a JSON string with all information needed to build the final SPARQL query."""
    question: str = dspy.InputField()
    schema_summary: str = dspy.InputField(desc="Same structure as input: {endpoint: {classes: [class1, class2, ...], properties: [property1, property2, ...]}}")
    join_candidates: str = dspy.InputField(desc='JSON string structured as: [{"endpoint_a": "", "endpoint_b": "", "property_a": "", "property_b": "", "join_variable": ""}...]')
    previous_error: str = dspy.InputField(desc="Error message from the previous execution of the generated SPARQL query, if any. \"none\" otherwise.")
    query_plan: str = dspy.OutputField(desc="""{"select": ["?var1", "?var2"],"endpoints": [{"url": "...", "patterns": [{"subject": "", "predicate": "", "object": ""}]}],
                                        "join_variables": ["?var"],
                                        "filters": [],
                                        "order_by": null,
                                        "limit": null
                                        }""")

class DiagnoseEmptyResult(dspy.Signature):
    """
    Given in input a natural language question, the generated SPARQL query, and the results obtained by probing the 
    SPARQL endpoint to check if the different endpoints contain the relevant data, diagnose whether the empty result is due to 
    wrong endpoints (i.e., the SPARQL query is correct but the endpoints do not contain the relevant data) or to a wrong schema 
    (i.e., the SPARQL query is not correctly formulated according to the actual schema of the endpoints).
    """
    question: str = dspy.InputField()
    sparql_query: str = dspy.InputField()
    probe_results: str = dspy.InputField(desc='{"endpoint_url": {"classes": {"ClassName": 42}, "properties": {"propName": true}}}')
    diagnosis: Literal['wrong_endpoints', 'wrong_schema'] = dspy.OutputField()