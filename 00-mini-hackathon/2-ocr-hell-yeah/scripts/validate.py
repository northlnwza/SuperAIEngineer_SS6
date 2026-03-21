def sum_candidates_votes(candidates: list[dict]) -> dict:
    vote_counts = {}
    for candidate in candidates:
        name = candidate['party_name']
        votes = candidate['votes']
        if name in vote_counts:
            vote_counts[name] += votes
        else:
            vote_counts[name] = votes
    return vote_counts

def is_match(predicted: str, actual: str) -> bool:
    return predicted.strip() == actual.strip()

def absolute_diff(predicted: int, actual: int) -> int:
    return abs(predicted - actual)