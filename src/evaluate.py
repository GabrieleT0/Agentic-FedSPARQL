import json
import math
import os
import sparql_utils
from metrics import execution_accuracy, f1_score, discovery_accuracy

def load_benchmark(benchmark_path: str) -> list:
    """Load the benchmark dataset from a JSON file."""
    with open(benchmark_path, 'r') as f:
        return json.load(f)

def load_existing_results(output_path: str) -> tuple[list, set]:
    """Load previously saved results and return them with a set of already-processed questions."""
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)
        processed = {r['question'] for r in results}
        print(f"Resuming: found {len(results)} previously processed question(s), skipping them.")
        return results, processed
    return [], set()

def evaluate(pipeline, benchmark_path: str, mode: str = 'full', output_path: str = 'evaluation_results.json'):
    examples = load_benchmark(benchmark_path)
    results, processed_questions = load_existing_results(output_path)

    for example in examples:
        if example['validation']['valid'] == True:
            question = example['question']

            if question in processed_questions:
                continue

            manually_check = False
            gold_endpoints = [ep['url'] for ep in example['endpoints']]
            gold_sparql = example['federated_sparql']
            gold_answers = sparql_utils.execute_sparql_query(gold_sparql)

            prediction = pipeline(question=question)
            predicted_endpoints = prediction.candidate_endpoints or []
            predicted_answers = prediction.query_results or []
            predicted_sparql = prediction.sparql_query
            refinement_attempts = prediction.refinement_attempts or 0
            discovery_attempts = prediction.discovery_attempts or 0

            # I.e. the case of first name instead of last name, to check if
            # there's error in the benchmark or if the LLM is making a mistake in the query generation
            if len(predicted_answers) == len(gold_answers):
                manually_check = True

            discovery_acc = discovery_accuracy(predicted_endpoints, gold_endpoints)
            exec_acc = execution_accuracy(predicted_answers, gold_answers)
            f1 = f1_score(predicted_answers, gold_answers)
                                                                    
            result = {
                "question": question,
                "predicted_endpoints": predicted_endpoints,
                "gold_endpoints": gold_endpoints,
                "predicted_sparql": predicted_sparql,
                "gold:sparql": gold_sparql,
                "predicted_answers": predicted_answers,
                "gold_answers": gold_answers,
                "discovery_accuracy": discovery_acc,
                "execution_accuracy": exec_acc,
                "f1_score": f1,
                "manually_check": manually_check,
                "refinement_attempts": refinement_attempts,
                "discovery_attempts": discovery_attempts,
            }
            results.append(result)
            processed_questions.add(question)

            # Save incrementally after each question
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
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
        print(f"Average F1 Score:           {mean('f1_score'):.4f}")
        print(f"Avg Refinement Attempts:    {mean('refinement_attempts'):.4f}  (std: {std('refinement_attempts'):.4f})")
        print(f"Avg Discovery Attempts:     {mean('discovery_attempts'):.4f}  (std: {std('discovery_attempts'):.4f})")

    return results
