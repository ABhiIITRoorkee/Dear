# utility/load_data.py (UPDATED v3, KD-READY, SAFE SAMPLING + OPTIONAL CONTROLS)

import numpy as np
import random as rd
import scipy.sparse as sp
from time import time


class Data:
    """
    Data loader for the Mashup-API recommendation task.

    Expected files in each fold directory:
      - TE.csv           : train mashup -> APIs
      - RS.csv           : test mashup -> APIs
      - mashup_tags.csv  : mashup -> mashup_tag_ids
      - api_tags.csv     : api -> api_tag_ids

    This version focuses on:
      1) Robust negative sampling (no infinite loops)
      2) Safe tag sampling even if tags are missing
      3) KD-ready utilities for router distillation

    NEW (optional control):
      - skip_tag_loss_when_missing:
          If enabled, missing positive tags return -1 sentinel.
          main.py should mask tag losses where pos_tag == -1.
          If disabled (default), missing tags fall back to random positive tags
          (backward compatible with your current pipeline).
    """

    _MAX_TRIES = 50
    _FALLBACK_RANDOM_TAG_FOR_MISSING_POS = True

    def __init__(self, path, batch_size, skip_tag_loss_when_missing=0):
        self.path = path
        self.batch_size = batch_size
        self.skip_tag_loss_when_missing = (int(skip_tag_loss_when_missing) == 1)

        train_file = path + '/TE.csv'
        test_file = path + '/RS.csv'
        api_tag_file = path + '/api_tags.csv'
        mashup_tag_file = path + '/mashup_tags.csv'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.n_api_tag, self.n_mashup_tag = 0, 0
        self.n_api, self.n_mashup = 0, 0

        self.neg_pools = {}
        self.exist_users = []
        self.recommendResult = {}

        # ------------------------------------------------------------
        # Pass 1: compute max ids + counts from TE/RS
        # ------------------------------------------------------------
        with open(train_file, encoding="utf-8") as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                parts = [p for p in l.strip('\n').split(' ') if p != ""]
                if len(parts) <= 1:
                    continue

                uid = int(parts[0])
                items = [int(i) for i in parts[1:]]

                self.exist_users.append(uid)
                self.n_users = max(self.n_users, uid)
                self.n_items = max(self.n_items, max(items))
                self.n_train += len(items)

        with open(test_file, encoding="utf-8") as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                parts = [p for p in l.strip('\n').split(' ') if p != ""]
                if len(parts) <= 1:
                    continue
                try:
                    items = [int(i) for i in parts[1:]]
                except Exception:
                    continue
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)

        self.n_items += 1
        self.n_users += 1

        # ------------------------------------------------------------
        # Pass 2: compute tag vocab sizes + api/mashup counts
        # ------------------------------------------------------------
        with open(api_tag_file, encoding="utf-8") as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                parts = [p for p in l.strip('\n').split(' ') if p != ""]
                if len(parts) <= 1:
                    continue
                try:
                    api_id = int(parts[0])
                    tag_ids = [int(x) for x in parts[1:]]
                except Exception:
                    continue
                self.n_api = max(self.n_api, api_id)
                self.n_api_tag = max(self.n_api_tag, max(tag_ids))

        with open(mashup_tag_file, encoding="utf-8") as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                parts = [p for p in l.strip('\n').split(' ') if p != ""]
                if len(parts) <= 1:
                    continue
                try:
                    mashup_id = int(parts[0])
                    tag_ids = [int(x) for x in parts[1:]]
                except Exception:
                    continue
                self.n_mashup = max(self.n_mashup, mashup_id)
                self.n_mashup_tag = max(self.n_mashup_tag, max(tag_ids))

        self.n_mashup += 1
        self.n_api += 1
        self.n_mashup_tag += 1
        self.n_api_tag += 1

        # ------------------------------------------------------------
        # Sparse matrices (kept for compatibility)
        # ------------------------------------------------------------
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Api = sp.dok_matrix((self.n_api, self.n_api_tag), dtype=np.float32)
        self.R_Mashup = sp.dok_matrix((self.n_mashup, self.n_mashup_tag), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        self.mashup_items, self.api_set = {}, {}

        # Read TE.csv
        with open(train_file, encoding="utf-8") as f_train:
            for l in f_train.readlines():
                if len(l) == 0:
                    break
                parts = [p for p in l.strip('\n').split(' ') if p != ""]
                if len(parts) <= 1:
                    continue
                uid = int(parts[0])
                apis = [int(i) for i in parts[1:]]
                self.train_items[uid] = apis
                for api in apis:
                    self.R[uid, api] = 1.0

        # Read RS.csv
        with open(test_file, encoding="utf-8") as f_test:
            for l in f_test.readlines():
                if len(l) == 0:
                    break
                parts = [p for p in l.strip('\n').split(' ') if p != ""]
                if len(parts) == 0:
                    continue
                uid = int(parts[0])
                apis = [int(i) for i in parts[1:]] if len(parts) > 1 else []
                self.test_set[uid] = apis

        # Read mashup_tags.csv
        with open(mashup_tag_file, encoding="utf-8") as f_mashup:
            for l in f_mashup.readlines():
                if len(l) == 0:
                    break
                parts = [p for p in l.strip('\n').split(' ') if p != ""]
                if len(parts) <= 1:
                    continue
                uid = int(parts[0])
                tags = [int(i) for i in parts[1:]]
                self.mashup_items[uid] = tags
                for t in tags:
                    self.R_Mashup[uid, t] = 1.0

        # Read api_tags.csv
        with open(api_tag_file, encoding="utf-8") as f_api:
            for l in f_api.readlines():
                if len(l) == 0:
                    break
                parts = [p for p in l.strip('\n').split(' ') if p != ""]
                if len(parts) <= 1:
                    continue
                api_id = int(parts[0])
                tags = [int(i) for i in parts[1:]]
                self.api_set[api_id] = tags
                for t in tags:
                    self.R_Api[api_id, t] = 1.0

        self.train_items_set = {u: set(v) for u, v in self.train_items.items()}
        self.mashup_tags_set = {u: set(v) for u, v in self.mashup_items.items()}
        self.api_tags_set = {i: set(v) for i, v in self.api_set.items()}

        self.exist_users = sorted(list(set(self.exist_users)))

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        sparsity = (self.n_train + self.n_test) / (self.n_users * self.n_items)
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, sparsity))

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            used = self.train_items_set.get(u, set())
            neg_items = list(set(range(self.n_items)) - used)
            if len(neg_items) == 0:
                self.neg_pools[u] = []
                continue
            self.neg_pools[u] = [rd.choice(neg_items) for _ in range(200)]
        print("negative_pool built in %.2fs" % (time() - t1))

    def _safe_sample_one(self, arr):
        return arr[rd.randint(0, len(arr) - 1)]

    def _sample_pos_item(self, u):
        return self._safe_sample_one(self.train_items[u])

    def _sample_neg_item(self, u):
        used = self.train_items_set.get(u, set())

        for _ in range(self._MAX_TRIES):
            neg = rd.randint(0, self.n_items - 1)
            if neg not in used:
                return neg

        candidates = list(set(range(self.n_items)) - used)
        if not candidates:
            return rd.randint(0, self.n_items - 1)
        return self._safe_sample_one(candidates)

    def _sample_pos_mashup_tag(self, u):
        tags = self.mashup_items.get(u, [])
        if not tags:
            return None
        return self._safe_sample_one(tags)

    def _sample_neg_mashup_tag(self, u):
        used = self.mashup_tags_set.get(u, set())
        for _ in range(self._MAX_TRIES):
            neg = rd.randint(0, self.n_mashup_tag - 1)
            if neg not in used:
                return neg
        return rd.randint(0, self.n_mashup_tag - 1)

    def _sample_pos_api_tag(self, api_id):
        tags = self.api_set.get(api_id, [])
        if not tags:
            return None
        return self._safe_sample_one(tags)

    def _sample_neg_api_tag(self, api_id):
        used = self.api_tags_set.get(api_id, set())
        for _ in range(self._MAX_TRIES):
            neg = rd.randint(0, self.n_api_tag - 1)
            if neg not in used:
                return neg
        return rd.randint(0, self.n_api_tag - 1)

    def _fallback_pos_tag(self, n_tags: int):
        # If we want to SKIP tag loss when missing positives, return sentinel -1.
        if self.skip_tag_loss_when_missing:
            return -1

        # Otherwise keep backward-compatible behavior: random fallback positive tag.
        if self._FALLBACK_RANDOM_TAG_FOR_MISSING_POS:
            return rd.randint(0, n_tags - 1)
        return -1

    def sample(self):
        if self.batch_size <= len(self.exist_users):
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        pos_items, neg_items = [], []
        pos_mashup_tags, neg_mashup_tags = [], []
        pos_api_tags, neg_api_tags = [], []

        for u in users:
            pos_api = self._sample_pos_item(u)
            neg_api = self._sample_neg_item(u)
            pos_items.append(pos_api)
            neg_items.append(neg_api)

            # -------- mashup tags --------
            pos_mt = self._sample_pos_mashup_tag(u)
            if pos_mt is None:
                pos_mt = self._fallback_pos_tag(self.n_mashup_tag)
                # neg won't be used if pos_mt == -1 (masked in main.py), but keep valid int
                neg_mt = 0
            else:
                neg_mt = self._sample_neg_mashup_tag(u)

            pos_mashup_tags.append(pos_mt)
            neg_mashup_tags.append(neg_mt)

            # -------- api tags --------
            pos_at = self._sample_pos_api_tag(pos_api)
            if pos_at is None:
                pos_at = self._fallback_pos_tag(self.n_api_tag)
                neg_at = 0
            else:
                neg_at = self._sample_neg_api_tag(pos_api)

            pos_api_tags.append(pos_at)
            neg_api_tags.append(neg_at)

        return users, pos_items, neg_items, pos_mashup_tags, neg_mashup_tags, pos_api_tags, neg_api_tags

    # ============================================================
    # KD-READY UTILITIES (OPTIONAL)
    # ============================================================
    def get_train_mashup_ids(self):
        return list(self.train_items.keys())

    def sample_kd_router_batch(self, n_queries=16, pool_size=300, exclude_self=True):
        train_ids = self.get_train_mashup_ids()
        if len(train_ids) == 0:
            return [], []

        n_queries = min(n_queries, len(train_ids))
        query_ids = rd.sample(train_ids, n_queries)

        candidate_ids = []
        for q in query_ids:
            ps = min(pool_size, len(train_ids))
            pool = rd.sample(train_ids, ps)
            if exclude_self and q in pool:
                pool.remove(q)
            candidate_ids.append(pool)

        return query_ids, candidate_ids

    def sample_kd_router_batch_np(self, n_queries=16, pool_size=300, exclude_self=True):
        q_ids, pools = self.sample_kd_router_batch(n_queries, pool_size, exclude_self)
        if len(q_ids) == 0:
            return np.array([], dtype=np.int64), np.zeros((0, 0), dtype=np.int64), np.array([], dtype=np.int64)

        Q = len(q_ids)
        P = max(len(p) for p in pools) if pools else 0

        pools_np = np.full((Q, P), fill_value=-1, dtype=np.int64)
        lens = np.zeros(Q, dtype=np.int64)

        for i, p in enumerate(pools):
            lens[i] = len(p)
            if len(p) > 0:
                pools_np[i, :len(p)] = np.array(p, dtype=np.int64)

        return np.array(q_ids, dtype=np.int64), pools_np, lens

    def sample_all_users_pos_items(self):
        self.all_train_users = []
        self.all_train_pos_items = []
        for u in self.exist_users:
            self.all_train_users += [u] * len(self.train_items[u])
            self.all_train_pos_items += self.train_items[u]

    def epoch_sample(self):
        def sample_neg_items_for_u(u):
            used = self.train_items_set.get(u, set())
            for _ in range(self._MAX_TRIES):
                neg = rd.randint(0, self.n_items - 1)
                if neg not in used:
                    return neg
            candidates = list(set(range(self.n_items)) - used)
            if not candidates:
                return rd.randint(0, self.n_items - 1)
            return self._safe_sample_one(candidates)

        neg_items = [sample_neg_items_for_u(u) for u in self.all_train_users]

        perm = np.random.permutation(len(self.all_train_users))
        users = np.array(self.all_train_users)[perm]
        pos_items = np.array(self.all_train_pos_items)[perm]
        neg_items = np.array(neg_items)[perm]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()
            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')
        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')
        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = {}

        for uid in all_users_to_test:
            train_iids = self.train_items.get(uid, [])
            test_iids = self.test_set.get(uid, [])
            n_iids = len(train_iids) + len(test_iids)
            user_n_iid.setdefault(n_iids, []).append(uid)

        split_uids = []
        temp = []
        count = 1
        n_rates = 0
        split_state = []

        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)
                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)
                temp = []
                n_rates = 0
                count += 1

            if idx == len(user_n_iid.keys()) - 1:
                split_uids.append(temp)
                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state
