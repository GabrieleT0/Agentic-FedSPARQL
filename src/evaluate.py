import json
import math
import os
import tempfile
import sparql_utils
from metrics import execution_accuracy, f1_score, discovery_accuracy

DEFAULT_RESULT_FIELDS = {
    "discovery_internal_retries": 0,
    "query_builder_internal_retries": 0,
    "schema_summary_retries": 0,
}

def _row_key(row: dict) -> tuple:
    """Normalize a result row to its values only, ignoring variable names."""
    return tuple(row.values())

def precision_score(predicted: list, gold: list) -> float:
    """Calculate answer precision between the predicted and gold answers."""
    predicted_set = {_row_key(row) for row in predicted}
    gold_set = {_row_key(row) for row in gold}

    if not predicted_set and not gold_set:
        return 1.0
    if not predicted_set:
        return 0.0

    return len(predicted_set & gold_set) / len(predicted_set)

def recall_score(predicted: list, gold: list) -> float:
    """Calculate answer recall between the predicted and gold answers."""
    predicted_set = {_row_key(row) for row in predicted}
    gold_set = {_row_key(row) for row in gold}

    if not predicted_set and not gold_set:
        return 1.0
    if not gold_set:
        return 0.0

    return len(predicted_set & gold_set) / len(gold_set)

def ensure_result_defaults(row: dict) -> None:
    row.pop("discovery_attempts", None)
    for field, default in DEFAULT_RESULT_FIELDS.items():
        row.setdefault(field, default)
    if "precision" not in row and "predicted_answers" in row and "gold_answers" in row:
        row["precision"] = precision_score(row["predicted_answers"], row["gold_answers"])
    if "recall" not in row and "predicted_answers" in row and "gold_answers" in row:
        row["recall"] = recall_score(row["predicted_answers"], row["gold_answers"])
    if row.get("execution_accuracy") == 1.0:
        row["manually_check"] = False

def load_benchmark(benchmark_path: str) -> list:
    """Load the benchmark dataset from a JSON file."""
    with open(benchmark_path, 'r') as f:
        return json.load(f)

def load_existing_results(output_path: str) -> tuple[list, set]:
    """Load previously saved results and return them with a set of already-processed questions."""
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)
        # Support both new format {"config": ..., "results": [...]} and legacy flat array
        results = data.get("results", data) if isinstance(data, dict) else data
        for row in results:
            ensure_result_defaults(row)
        processed = {r['question'] for r in results}
        print(f"Resuming: found {len(results)} previously processed question(s), skipping them.")
        return results, processed
    return [], set()


def save_results(output_path: str, results: list, config: dict | None = None) -> None:
    """Persist results atomically to avoid leaving partially written JSON behind."""
    payload = {"config": config or {}, "results": results}
    output_dir = os.path.dirname(output_path) or "."

    with tempfile.NamedTemporaryFile("w", dir=output_dir, delete=False) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp_path = tmp.name

    os.replace(tmp_path, output_path)

def evaluate(pipeline, benchmark_path: str, mode: str = 'full', output_path: str = 'evaluation_results.json', config: dict = None):
    examples = load_benchmark(benchmark_path)
    results, processed_questions = load_existing_results(output_path)
    skipped_questions = 0

    for example in examples:
        if example['validation']['valid'] == True and example['validation']['original_count'] > 0:
            question = example['question']

            if question in processed_questions:
                continue

            gold_endpoints = [ep['url'] for ep in example['endpoints']]
            gold_sparql = example['federated_sparql']
            try:
                gold_answers = sparql_utils.execute_sparql_query(gold_sparql)
            except sparql_utils.SPARQLExecutionError as e:
                skipped_questions += 1
                print(f"Skipping question after gold-query failure: {question[:80]}")
                print(f"Gold query error: {e}")
                continue

            prediction = pipeline(question=question)
            predicted_endpoints = prediction.candidate_endpoints or []
            predicted_answers = prediction.query_results or []
            predicted_sparql = prediction.sparql_query
            refinement_attempts = prediction.refinement_attempts or 0
            discovery_internal_retries = prediction.discovery_internal_retries or 0
            query_builder_internal_retries = prediction.query_builder_internal_retries or 0
            schema_summary_retries = prediction.schema_summary_retries or 0

            discovery_acc = discovery_accuracy(predicted_endpoints, gold_endpoints)
            exec_acc = execution_accuracy(predicted_answers, gold_answers)
            precision = precision_score(predicted_answers, gold_answers)
            recall = recall_score(predicted_answers, gold_answers)
            f1 = f1_score(predicted_answers, gold_answers)

            # I.e. the case of first name instead of last name, to check if
            # there's error in the benchmark or if the LLM is making a mistake in the query generation.
            manually_check = len(predicted_answers) == len(gold_answers) and exec_acc != 1.0
                                                                    
            result = {
                "question": question,
                "predicted_endpoints": predicted_endpoints,
                "gold_endpoints": gold_endpoints,
                "predicted_sparql": predicted_sparql,
                "gold_sparql": gold_sparql,
                "predicted_answers": predicted_answers,
                "gold_answers": gold_answers,
                "discovery_accuracy": discovery_acc,
                "execution_accuracy": exec_acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "manually_check": manually_check,
                "refinement_attempts": refinement_attempts,
                "discovery_internal_retries": discovery_internal_retries,
                "query_builder_internal_retries": query_builder_internal_retries,
                "schema_summary_retries": schema_summary_retries,
            }
            results.append(result)
            processed_questions.add(question)

            # Save incrementally after each question
            save_results(output_path, results, config=config)
            print(f"[{len(results)}] Saved result for: {question[:80]}")

    if results:
        n = len(results)

        def mean(key):
            return sum(r[key] for r in results) / n

        def std(key):
            m = mean(key)
            return math.sqrt(sum((r[key] - m) ** 2 for r in results) / n)

        print(f"Average Discovery Accuracy: {mean('discovery_accuracy'):.4f}")
        print(f"Average Execution Accuracy: {mean('execution_accuracy'):.4f}")
        print(f"Average Precision:          {mean('precision'):.4f}")
        print(f"Average Recall:             {mean('recall'):.4f}")
        print(f"Average F1 Score:           {mean('f1_score'):.4f}")
        print(f"Avg Refinement Attempts:           {mean('refinement_attempts'):.4f}  (std: {std('refinement_attempts'):.4f})")
        print(f"Avg Discovery Internal Retries:    {mean('discovery_internal_retries'):.4f}  (std: {std('discovery_internal_retries'):.4f})")
        print(f"Avg QueryBuilder Internal Retries: {mean('query_builder_internal_retries'):.4f}  (std: {std('query_builder_internal_retries'):.4f})")
        print(f"Avg Schema Summary Retries:        {mean('schema_summary_retries'):.4f}  (std: {std('schema_summary_retries'):.4f})")
        print(f"Skipped Questions:                 {skipped_questions}")

    return results
