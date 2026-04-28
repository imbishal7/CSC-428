"""Side-channel evaluation metrics: guessing entropy and success rate.

For each attack trace t with plaintext p, we score every candidate key byte k* in 0..255 by
the model's predicted log-probability of the leakage label that k* would produce. We
accumulate these log-probabilities across traces; the rank of the true key in the resulting
score vector is the metric. Averaging across many random orderings of the attack set gives
guessing entropy as a function of the number of traces used.
"""

from __future__ import annotations

import numpy as np

from .data import AES_SBOX, HW_TABLE, TARGET_BYTE


def _candidate_label_matrix(plaintexts: np.ndarray, label_kind: str) -> np.ndarray:
    """Returns (n_traces, 256) array: label that each candidate key would produce."""
    p = plaintexts[:, TARGET_BYTE].astype(np.int64)        # (n_traces,)
    k = np.arange(256, dtype=np.int64)                      # (256,)
    sbox_out = AES_SBOX[(p[:, None] ^ k[None, :])]          # (n_traces, 256)
    if label_kind == "hw":
        return HW_TABLE[sbox_out]
    if label_kind == "sbox":
        return sbox_out
    raise ValueError(f"unknown label_kind {label_kind!r}")


def key_rank_curve(model_probs: np.ndarray,
                   plaintexts: np.ndarray,
                   true_key_byte: int,
                   label_kind: str = "hw",
                   n_attacks: int = 50,
                   max_traces: int | None = None,
                   seed: int = 0) -> np.ndarray:
    """Average rank of the true key vs number of attack traces used.

    model_probs: (n_traces, n_classes) softmax outputs from the attack set.
    Returns array of length max_traces with the mean rank (0 = key recovered).
    """
    n_traces, n_classes = model_probs.shape
    max_traces = max_traces or n_traces
    log_probs = np.log(model_probs + 1e-36)                 # (n_traces, n_classes)

    cand_labels = _candidate_label_matrix(plaintexts, label_kind)   # (n_traces, 256)
    # Per-trace, per-candidate score = log P(label that candidate predicts)
    rows = np.arange(n_traces)[:, None]
    per_trace_scores = log_probs[rows, cand_labels]         # (n_traces, 256)

    rng = np.random.default_rng(seed)
    rank_sum = np.zeros(max_traces, dtype=np.float64)
    for _ in range(n_attacks):
        order = rng.permutation(n_traces)[:max_traces]
        cumulative = np.cumsum(per_trace_scores[order], axis=0)   # (max_traces, 256)
        # Rank of true key at each step: count of candidates with higher score.
        ranks = (cumulative > cumulative[:, true_key_byte:true_key_byte + 1]).sum(axis=1)
        rank_sum += ranks

    return rank_sum / n_attacks


def traces_to_recover(rank_curve: np.ndarray, threshold: int = 0) -> int | None:
    """First index where the average rank stays <= threshold. None if never."""
    below = rank_curve <= threshold
    if not below.any():
        return None
    return int(np.argmax(below))
