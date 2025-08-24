from simhash import Simhash

def fingerprint(text: str) -> int:
    return Simhash(text).value

def is_duplicate(fp: int, seen: set[int], tol: int = 3) -> bool:
    # basic hash set check; extend with Hamming distance buckets if needed
    return fp in seen
