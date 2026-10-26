# utility/batch_test.py (FIXED: fold-safe multiprocessing + faster candidate set)

import heapq
import multiprocessing
from typing import Dict, List, Tuple

import numpy as np
import torch

import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import Data

args = parse_args()
Ks = eval(args.Ks)

# ------------------------------------------------------------
# Globals populated per-process (important for multiprocessing)
# ------------------------------------------------------------
data_generator = None
USR_NUM = ITEM_NUM = None
BATCH_SIZE = args.batch_size

cores = max(1, multiprocessing.cpu_count() // 2)

# Precomputed all-items set (per-process)
_ALL_ITEMS_SET = None


def set_eval_seed(seed: int = 2026):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_worker(dataset: str, data_path: str, batch_size: int):
    """
    Initializer called once per worker process.
    Ensures every worker loads the correct fold-specific Data object.
    """
    global data_generator, USR_NUM, ITEM_NUM, BATCH_SIZE, _ALL_ITEMS_SET
    data_generator = Data(path=data_path + dataset, batch_size=batch_size)
    USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
    BATCH_SIZE = batch_size
    _ALL_ITEMS_SET = set(range(ITEM_NUM))


def ranklist_by_heapq(u: int,
                      user_pos_test: List[int],
                      test_items: List[int],
                      rating: np.ndarray,
                      Ks: List[int]) -> Tuple[List[int], float]:
    item_score = {i: rating[i] for i in test_items}
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    user_pos_test_set = set(user_pos_test)
    r = [1 if i in user_pos_test_set else 0 for i in K_max_item_score]
    return r, 0.0


def get_auc(item_score: Dict[int, float], user_pos_test: List[int]) -> float:
    item_score_sorted = sorted(item_score.items(), key=lambda kv: kv[1], reverse=True)
    item_sort = [x[0] for x in item_score_sorted]
    posterior = [x[1] for x in item_score_sorted]

    user_pos_test_set = set(user_pos_test)
    r = [1 if i in user_pos_test_set else 0 for i in item_sort]
    return metrics.auc(ground_truth=r, prediction=posterior)


def ranklist_by_sorted(u: int,
                       user_pos_test: List[int],
                       test_items: List[int],
                       rating: np.ndarray,
                       Ks: List[int]) -> Tuple[List[int], float]:
    item_score = {i: rating[i] for i in test_items}
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    user_pos_test_set = set(user_pos_test)
    r = [1 if i in user_pos_test_set else 0 for i in K_max_item_score]
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test: List[int], r: List[int], auc: float, Ks: List[int]) -> Dict[str, np.ndarray]:
    precision, recall, ndcg, hit_ratio = [], [], [], []
    map_list, fone_list, mrr_list = [], [], []

    gt_size = len(user_pos_test)
    for K in Ks:
        pre_k = metrics.precision_at_k(r, K)
        rec_k = metrics.recall_at_k(r, K, gt_size)

        precision.append(pre_k)
        recall.append(rec_k)
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))
        map_list.append(metrics.average_precision(r, K))
        mrr_list.append(metrics.mrr_at_k(r, K))
        fone_list.append(metrics.F1(pre_k, rec_k))

    return {
        'recall': np.array(recall),
        'precision': np.array(precision),
        'ndcg': np.array(ndcg),
        'hit_ratio': np.array(hit_ratio),
        'auc': auc,
        'map': np.array(map_list),
        'mrr': np.array(mrr_list),
        'fone': np.array(fone_list),
    }


def test_one_user(x):
    """
    Worker: x = (rating_vector, user_id)
    Requires data_generator to be initialized via _init_worker.
    """
    global data_generator, _ALL_ITEMS_SET
    rating, u = x[0], x[1]

    training_items = data_generator.train_items.get(u, [])
    user_pos_test = data_generator.test_set[u]

    # Candidate set: all - training
    test_items = list(_ALL_ITEMS_SET - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(u, user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(u, user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test_torch(ua_embeddings: torch.Tensor,
               ia_embeddings: torch.Tensor,
               users_to_test: List[int],
               drop_flag: bool = False,
               batch_test_flag: bool = False,
               use_multiprocessing: bool = True,
               eval_seed: int = None) -> Dict[str, np.ndarray]:
    if eval_seed is not None:
        set_eval_seed(eval_seed)

    result = {
        'precision': np.zeros(len(Ks)),
        'recall': np.zeros(len(Ks)),
        'ndcg': np.zeros(len(Ks)),
        'hit_ratio': np.zeros(len(Ks)),
        'auc': 0.0,
        'map': np.zeros(len(Ks)),
        'mrr': np.zeros(len(Ks)),
        'fone': np.zeros(len(Ks)),
    }

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batches = n_test_users // u_batch_size + 1
    count = 0

    for u_batch_id in range(n_user_batches):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start:end]
        if not user_batch:
            continue

        # Compute scores
        if batch_test_flag:
            n_item_batches = ia_embeddings.shape[0] // i_batch_size + 1
            rate_batch = np.zeros((len(user_batch), ia_embeddings.shape[0]), dtype=np.float32)

            for i_batch_id in range(n_item_batches):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ia_embeddings.shape[0])
                if i_start >= i_end:
                    continue

                u_g = ua_embeddings[user_batch]
                i_g = ia_embeddings[i_start:i_end]
                scores = torch.matmul(u_g, i_g.t()).detach().cpu().numpy()
                rate_batch[:, i_start:i_end] = scores
        else:
            u_g = ua_embeddings[user_batch]
            scores = torch.matmul(u_g, ia_embeddings.t()).detach().cpu().numpy()
            rate_batch = scores

        user_batch_rating_uid = list(zip(rate_batch, user_batch))

        if use_multiprocessing and user_batch_rating_uid:
            # fold-safe: init worker with dataset params
            with multiprocessing.Pool(
                processes=cores,
                initializer=_init_worker,
                initargs=(args.dataset, args.data_path, args.batch_size),
            ) as pool:
                batch_result = pool.map(test_one_user, user_batch_rating_uid)
        else:
            # Ensure single-process also has data_generator ready
            if data_generator is None:
                _init_worker(args.dataset, args.data_path, args.batch_size)
            batch_result = [test_one_user(x) for x in user_batch_rating_uid]

        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users
            result['map'] += re['map'] / n_test_users
            result['mrr'] += re['mrr'] / n_test_users
            result['fone'] += re['fone'] / n_test_users

    assert count == n_test_users, f"Evaluated users={count}, expected={n_test_users}"
    return result
