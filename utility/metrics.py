# utility/metrics.py (UPDATED, ROBUST + BACKWARD COMPATIBLE)

import numpy as np
from sklearn.metrics import roc_auc_score


# ============================================================
# Core ranking metrics for Top-K recommendation
# ============================================================
# IMPORTANT CONVENTION USED IN THIS PROJECT
# ----------------------------------------
# Most functions here operate on:
#   r: a binary relevance list, where
#      r[i] = 1 if the i-th recommended item is relevant (hit),
#      r[i] = 0 otherwise.
#
# This matches how batch_test.py builds r:
#   r is aligned with the top-K ranked items for a user.
#
# Many metrics need "k" (cutoff) and sometimes "all_pos_num" (ground truth size).
# ============================================================


def _to_binary_array(r):
    """
    Convert input list/array to a 1D numpy array of floats or ints.
    We treat any non-zero value as relevant (1).
    """
    arr = np.asarray(r).astype(np.float32).reshape(-1)
    # Ensure strict {0,1} relevance
    arr = (arr > 0).astype(np.float32)
    return arr


def precision_at_k(r, k):
    """
    Precision@K = (# hits in top-K) / K

    Args:
        r: binary relevance list aligned with ranked recommendations
        k: cutoff

    Returns:
        float precision in [0,1]
    """
    k = int(k)
    if k <= 0:
        return 0.0
    r = _to_binary_array(r)[:k]
    if r.size == 0:
        return 0.0
    # If k > len(r), this definition uses available results only
    # (safer than asserting).
    return float(np.mean(r))


def recall_at_k(r, k, all_pos_num):
    """
    Recall@K = (# hits in top-K) / (# ground truth positives)

    Args:
        r: binary relevance list aligned with ranked recommendations
        k: cutoff
        all_pos_num: number of relevant items in ground truth

    Returns:
        float recall in [0,1]
    """
    k = int(k)
    if k <= 0:
        return 0.0
    if all_pos_num is None or all_pos_num <= 0:
        return 0.0
    r = _to_binary_array(r)[:k]
    return float(np.sum(r) / float(all_pos_num))


def hit_at_k(r, k):
    """
    Hit@K = 1 if there is at least one hit in top-K, else 0.
    """
    k = int(k)
    if k <= 0:
        return 0.0
    r = _to_binary_array(r)[:k]
    return 1.0 if np.sum(r) > 0 else 0.0


def mrr_at_k(r, k):
    """
    MRR@K (Mean Reciprocal Rank at cutoff):
      MRR@K = 1 / (rank of first relevant item), if exists in top-K
            = 0 otherwise
    """
    k = int(k)
    if k <= 0:
        return 0.0
    r = _to_binary_array(r)[:k]
    for i in range(len(r)):
        if r[i] == 1:
            return float(1.0 / (i + 1))
    return 0.0


def average_precision(r, cut):
    """
    AP@cut (Average Precision at cutoff):
      AP@cut = average of Precision@k values at ranks where r[k]==1,
               computed only up to 'cut'.

    This implementation matches your baseline logic, but is safer.
    """
    cut = int(cut)
    if cut <= 0:
        return 0.0

    r = _to_binary_array(r)[:cut]
    if r.size == 0:
        return 0.0

    precisions = []
    for i in range(len(r)):
        if r[i] == 1:
            precisions.append(np.mean(r[: i + 1]))

    if len(precisions) == 0:
        return 0.0

    # Divide by number of hits inside cutoff (standard AP@K)
    return float(np.mean(precisions))


def mean_average_precision(rs, k):
    """
    MAP@K: mean of AP@K over multiple users.

    Args:
        rs: list of relevance vectors (each is r for one user)
        k: cutoff
    """
    if rs is None or len(rs) == 0:
        return 0.0
    return float(np.mean([average_precision(r, k) for r in rs]))


def dcg_at_k(r, k, method=1):
    """
    DCG@K (Discounted Cumulative Gain):
    Assumes r contains graded relevance values, but binary works too.

    method=0: r[0] + sum_{i=1..k-1} r[i] / log2(i+1)
    method=1: sum_{i=0..k-1} r[i] / log2(i+2)  (common)
    """
    k = int(k)
    if k <= 0:
        return 0.0

    r = np.asfarray(r).reshape(-1)[:k]
    if r.size == 0:
        return 0.0

    if method == 0:
        return float(r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1))))
    elif method == 1:
        return float(np.sum(r / np.log2(np.arange(2, r.size + 2))))
    else:
        raise ValueError('method must be 0 or 1.')


def ndcg_at_k(r, k, method=1):
    """
    NDCG@K = DCG@K / IDCG@K

    Where IDCG@K is DCG@K of the ideal ranking (sorted by relevance desc).
    """
    k = int(k)
    if k <= 0:
        return 0.0

    r = np.asfarray(r).reshape(-1)
    if r.size == 0:
        return 0.0

    # Ideal ranking: sort only within top-K scope
    r_max = sorted(r[:k], reverse=True)
    dcg_max = dcg_at_k(r_max, k, method)
    if dcg_max <= 0:
        return 0.0

    return float(dcg_at_k(r, k, method) / dcg_max)


def F1(pre, rec):
    """
    F1 score given precision and recall.
    """
    pre = float(pre)
    rec = float(rec)
    denom = pre + rec
    if denom <= 0:
        return 0.0
    return float((2.0 * pre * rec) / denom)


# ============================================================
# Legacy helper kept for compatibility
# ============================================================
def recall(rank, ground_truth, N):
    """
    Legacy recall definition used in some older scripts:
      recall = |topN ∩ ground_truth| / |ground_truth|
    This expects rank to be a list of item IDs, not binary r.
    """
    if ground_truth is None or len(set(ground_truth)) == 0:
        return 0.0
    return float(len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth))))


# ============================================================
# AUC (binary classification quality for ranking scores)
# ============================================================
def auc(ground_truth, prediction):
    """
    Computes ROC-AUC for a ranked list:
      ground_truth: list of 0/1 labels aligned with prediction scores
      prediction  : list of real-valued scores

    Important edge cases:
    - roc_auc_score fails if y_true has only one class (all 0s or all 1s)
    - We return 0.0 in that case to keep the pipeline stable.
    """
    try:
        y_true = np.asarray(ground_truth).reshape(-1)
        y_score = np.asarray(prediction).reshape(-1)

        # Need at least 2 classes present for ROC-AUC
        if len(np.unique(y_true)) < 2:
            return 0.0

        res = roc_auc_score(y_true=y_true, y_score=y_score)
        return float(res)
    except Exception:
        return 0.0
