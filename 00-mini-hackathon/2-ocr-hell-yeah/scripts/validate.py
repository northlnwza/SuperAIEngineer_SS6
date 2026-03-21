from .constants import PARTY_NAMES
import pandas as pd

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

def is_voted_party_record(row: pd.Series, party_idx: int, vote_idx: int, party_list: list) -> bool:
    raw_party = str(row.iloc[party_idx]).strip()
    raw_votes = str(row.iloc[vote_idx]).strip()

    has_party = any(party in raw_party for party in party_list)
    
    if not has_party:
        return False

    from .transform import only_thai_numbers, thai_to_arabic 
    
    clean_votes = thai_to_arabic(only_thai_numbers(raw_votes))
    
    return clean_votes.isdigit()