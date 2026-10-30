import numpy as np

def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.

    Args:
        candidates: list of N lists, each containing K candidate outputs.
        scores: list of N lists, each containing K reward scores.

    Returns:
        List of N selected candidates.
    """
    select = []
    for candidate, topk in zip(candidates, scores):
        x = np.argmax(np.array(topk))
        select.append(candidate[x])
    return select
    pass