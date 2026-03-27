def majority_vote(predictions: list[int]) -> int:
    """Return the majority vote from a list of predictions."""
    from collections import Counter

    counter = Counter(predictions)
    majority_vote = counter.most_common(1)[0][0]
    return majority_vote