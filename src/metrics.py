from decimal import Decimal, InvalidOperation


def _canonical_value(value) -> str:
    """Normalize SPARQL JSON binding values for answer comparison."""
    text = str(value).strip()
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return text

    if decimal == decimal.to_integral_value():
        return str(decimal.quantize(Decimal(1)))
    return format(decimal.normalize(), "f")


def _row_key(row: dict) -> tuple:
    """Normalize a result row to its value multiset, ignoring variable names/order."""
    return tuple(sorted(_canonical_value(value) for value in row.values()))


def answer_set(rows: list) -> set[tuple]:
    return {_row_key(row) for row in rows}

def execution_accuracy(predicted: list, gold: list) -> float:
    """Calculate the execution accuracy between the predicted and gold answers."""
    predicted_set = answer_set(predicted)
    gold_set = answer_set(gold)
    return 1.0 if predicted_set == gold_set else 0.0

def f1_score(predicted: list, gold: list) -> float:
    """Calculate the F1 score between the predicted and gold answers."""
    predicted_set = answer_set(predicted)
    gold_set = answer_set(gold)
    
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


def precision_score(predicted: list, gold: list) -> float:
    """Calculate answer precision between the predicted and gold answers."""
    predicted_set = answer_set(predicted)
    gold_set = answer_set(gold)

    if not predicted_set and not gold_set:
        return 1.0
    if not predicted_set:
        return 0.0

    return len(predicted_set & gold_set) / len(predicted_set)


def recall_score(predicted: list, gold: list) -> float:
    """Calculate answer recall between the predicted and gold answers."""
    predicted_set = answer_set(predicted)
    gold_set = answer_set(gold)

    if not predicted_set and not gold_set:
        return 1.0
    if not gold_set:
        return 0.0

    return len(predicted_set & gold_set) / len(gold_set)

def discovery_accuracy(predicted_endpoints: list, gold_endpoints: list) -> float:
    """Calculate exact-match discovery accuracy between predicted and gold endpoints."""
    predicted_set = set(predicted_endpoints)
    gold_set = set(gold_endpoints)
    return 1.0 if predicted_set == gold_set else 0.0
