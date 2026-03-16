# state.py
from typing import TypedDict, Optional, List

class FederatedQAState(TypedDict):
    # Input
    question: str

    # Agent 1 output
    candidate_endpoints: List[str]
    discovery_attempts: int

    # Agent 2 output
    schema_summary: dict           # {endpoint_url: {classes, properties, ...}}

    # Agent 3 output
    intermediate_representation: dict   # the JSON plan
    ir_error: Optional[str]             # error from IR validator, if any
    sparql_query: str                   # compiled SPARQL

    # Agent 4 output
    query_result: Optional[list]
    error_type: Optional[str]      # "wrong_endpoints" | "wrong_schema" | None
    error_message: Optional[str]
    refinement_attempts: int

    # Final output
    final_answer: Optional[str]