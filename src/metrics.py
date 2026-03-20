def execution_accuracy(predicted: list, gold: list) -> float:
    """Calculate the execution accuracy between the predicted and gold answers."""
    predicted_set = {frozenset(row.items()) for row in predicted}
    gold_set = {frozenset(row.items()) for row in gold}
    return 1.0 if predicted_set == gold_set else 0.0

def f1_score(predicted: list, gold: list) -> float:
    """Calculate the F1 score between the predicted and gold answers."""
    predicted_set = {frozenset(row.items()) for row in predicted}
    gold_set = {frozenset(row.items()) for row in gold}
    
    if not predicted_set and not gold_set:
        return 1.0  # Both are empty, perfect match
    if not predicted_set or not gold_set:
        return 0.0  # One is empty and the other is not, no match
    
    tp = len(predicted_set & gold_set)  
    precision = tp / len(predicted_set) 
    recall = tp / len(gold_set)

    if precision + recall == 0:
        return 0.0  
    return 2 * (precision * recall) / (precision + recall)

def discovery_accuracy(predicted_endpoints: list, gold_endpoints: list) -> float:
    """Calculate the discovery accuracy between the predicted and gold endpoints."""
    predicted_set = set(predicted_endpoints)
    gold_set = set(gold_endpoints)
    return 1.0 if gold_set.issubset(predicted_set) else 0.0