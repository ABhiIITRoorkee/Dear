# Models.py (UPDATED, KD-READY, BACKWARD COMPATIBLE, FAST H5 LOAD, IMPORT-SAFE)

import h5py
import numpy as np
import torch
import torch.nn as nn


class HCF(nn.Module):
    """
    HCF: Hypergraph Collaborative Filtering backbone.

    Backward compatible with your main.py:
        ua_embeddings, ia_embeddings, ma_embeddings, aa_embeddings = model(...)
    """

    def __init__(
        self,
        n_users,
        n_items,
        n_mashup,
        n_mashup_tag,
        n_api,
        n_api_tag,
        embedding_dim,
        layer_num,
        dropout_list
    ):
        super().__init__()

        # ----------------------------
        # Basic size config
        # ----------------------------
        self.n_users = n_users
        self.n_items = n_items
        self.n_mashup = n_mashup
        self.n_mashup_tag = n_mashup_tag
        self.n_api = n_api
        self.n_api_tag = n_api_tag

        # Desired embedding dim from args (may differ from H5 dim; we will not crash)
        self.embedding_dim = embedding_dim

        # Number of propagation layers (L)
        self.n_layers = layer_num

        # Dropout per layer
        assert len(dropout_list) == self.n_layers, (
            f"dropout_list must have length={self.n_layers}, got {len(dropout_list)}"
        )
        self.dropout_list = nn.ModuleList([nn.Dropout(p) for p in dropout_list])

        # ----------------------------
        # Placeholder embeddings (overwritten by _init_weight_)
        # ----------------------------
        torch.manual_seed(50)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.mashup_tag_embedding = nn.Embedding(n_mashup_tag, embedding_dim)
        self.api_tag_embedding = nn.Embedding(n_api_tag, embedding_dim)

        # Load pretrained textual vectors (from H5)
        self._init_weight_()

        # ----------------------------
        # Trainable layer weights over (L+1) states
        # ----------------------------
        self.u_weights = nn.Parameter(torch.ones(self.n_layers + 1))
        self.i_weights = nn.Parameter(torch.ones(self.n_layers + 1))
        self.m_weights = nn.Parameter(torch.ones(self.n_layers + 1))
        self.a_weights = nn.Parameter(torch.ones(self.n_layers + 1))

    # ------------------------------------------------------------
    # FAST H5 bulk loader
    # ------------------------------------------------------------
    @staticmethod
    def _bulk_load_h5_vectors(file_path: str, n_expected: int, name: str) -> torch.Tensor:
        """
        Load vectors [0..n_expected-1] from an h5 file (key=str(i)) in one file open.

        Returns:
            torch.FloatTensor shape (n_expected, dim)

        Raises:
            ValueError if any id is missing or vectors have inconsistent dims.
        """
        if not file_path:
            raise ValueError(f"[{name}] empty file_path")

        with h5py.File(file_path, "r") as f:
            # Determine dimension from first available key
            if str(0) in f:
                sample = f[str(0)][:]
            else:
                keys = list(f.keys())
                if not keys:
                    raise ValueError(f"[{name}] {file_path} is empty (no keys).")
                sample = f[keys[0]][:]

            if sample.ndim != 1:
                raise ValueError(f"[{name}] expected 1D vectors in {file_path}, got shape={sample.shape}")

            dim = int(sample.shape[0])
            mat = np.zeros((n_expected, dim), dtype=np.float32)

            missing = []
            for i in range(n_expected):
                k = str(i)
                if k not in f:
                    missing.append(i)
                    continue
                v = f[k][:]
                if v.ndim != 1 or int(v.shape[0]) != dim:
                    raise ValueError(
                        f"[{name}] inconsistent vector shape at id={i} in {file_path}. "
                        f"Expected ({dim},) got {v.shape}"
                    )
                mat[i] = v.astype(np.float32, copy=False)

            if missing:
                raise ValueError(
                    f"[{name}] missing {len(missing)} vectors in {file_path}. "
                    f"First missing ids: {missing[:10]}"
                )

        return torch.from_numpy(mat)

    # ------------------------------------------------------------
    # Initialize embedding tables from stored vectors
    # ------------------------------------------------------------
    def _init_weight_(self):
        """
        Initialize embeddings from:
          - mashup vectors     : data/vectors.h5
          - API vectors        : data/API_vectors.h5
          - mashup tag vectors : data/mashup_tag_vector.h5
          - api tag vectors    : data/api_tag_vector.h5
        """

        mashup_tensor = self._bulk_load_h5_vectors("data/vectors.h5", self.n_users, "mashup")
        api_tensor = self._bulk_load_h5_vectors("data/API_vectors.h5", self.n_items, "api")
        mashup_tag_tensor = self._bulk_load_h5_vectors("data/mashup_tag_vector.h5", self.n_mashup_tag, "mashup_tag")
        api_tag_tensor = self._bulk_load_h5_vectors("data/api_tag_vector.h5", self.n_api_tag, "api_tag")

        # Sanity: user/item dims must match for dot-product scoring
        d_u = mashup_tensor.shape[1]
        d_i = api_tensor.shape[1]
        d_mt = mashup_tag_tensor.shape[1]
        d_at = api_tag_tensor.shape[1]

        if not (d_u == d_i == d_mt == d_at):
            raise ValueError(
                "Embedding dimension mismatch across h5 files:\n"
                f"  mashup={d_u}, api={d_i}, mashup_tag={d_mt}, api_tag={d_at}\n"
                "All must be same for your current training losses."
            )

        # If args.embed_size differs from H5 dim, we keep H5 dim (do NOT crash silently later)
        if int(self.embedding_dim) != int(d_u):
            print(
                f"[WARN] args.embed_size={self.embedding_dim} but loaded H5 dim={d_u}. "
                f"Using H5 dim={d_u} inside model embeddings."
            )
            self.embedding_dim = int(d_u)

        self.user_embedding = nn.Embedding.from_pretrained(mashup_tensor, freeze=False)
        self.item_embedding = nn.Embedding.from_pretrained(api_tensor, freeze=False)
        self.mashup_tag_embedding = nn.Embedding.from_pretrained(mashup_tag_tensor, freeze=False)
        self.api_tag_embedding = nn.Embedding.from_pretrained(api_tag_tensor, freeze=False)

        # Make sure they are trainable
        self.user_embedding.weight.requires_grad = True
        self.item_embedding.weight.requires_grad = True
        self.mashup_tag_embedding.weight.requires_grad = True
        self.api_tag_embedding.weight.requires_grad = True

    # ------------------------------------------------------------
    # One hypergraph propagation block
    # ------------------------------------------------------------
    def _propagate(self, base_emb, adj1, adj2, weights, dropout_modules):
        """
        base_emb: (N, D)
        states:   [layer0, layer1, ..., layerL] => L+1 states
        """
        states = [base_emb]

        for layer in range(self.n_layers):
            t = torch.sparse.mm(adj2, states[-1])
            t = torch.sparse.mm(adj1, t)
            t = dropout_modules[layer](t)
            states.append(t)

        stack = torch.stack(states, dim=1)  # (N, L+1, D)
        w = torch.softmax(weights, dim=0).view(1, self.n_layers + 1, 1)
        return torch.sum(stack * w, dim=1)  # (N, D)

    # ------------------------------------------------------------
    # Forward (same outputs as before)
    # ------------------------------------------------------------
    def forward(self, adj_u1, adj_u2, adj_i1, adj_i2, adj_m1, adj_m2, adj_a1, adj_a2):
        u_emb = self._propagate(self.user_embedding.weight, adj_u1, adj_u2, self.u_weights, self.dropout_list)
        i_emb = self._propagate(self.item_embedding.weight, adj_i1, adj_i2, self.i_weights, self.dropout_list)
        m_emb = self._propagate(self.mashup_tag_embedding.weight, adj_m1, adj_m2, self.m_weights, self.dropout_list)
        a_emb = self._propagate(self.api_tag_embedding.weight, adj_a1, adj_a2, self.a_weights, self.dropout_list)
        return u_emb, i_emb, m_emb, a_emb

    # ============================================================
    # KD-READY HELPERS (do not break anything)
    # ============================================================
    def encode_all(self, adj_u1, adj_u2, adj_i1, adj_i2, adj_m1, adj_m2, adj_a1, adj_a2):
        return self.forward(adj_u1, adj_u2, adj_i1, adj_i2, adj_m1, adj_m2, adj_a1, adj_a2)

    @staticmethod
    def compose_query_embedding(neighbor_embeddings: torch.Tensor, weights: torch.Tensor):
        if weights.dim() != 1:
            weights = weights.view(-1)
        weights = weights / (weights.sum() + 1e-12)
        return torch.sum(neighbor_embeddings * weights.unsqueeze(1), dim=0)


__all__ = ["HCF"]
