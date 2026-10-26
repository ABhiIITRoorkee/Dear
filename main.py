# main.py
# ------------------------------------------------------------
# KD-READY + (OPTIONAL) Latent Hypergraph Inference (SPHINX-style)
#
# Works with your provided:
# - utility/parser.py
# - Models.py
#
# Key fixes + compat:
# - Robust arg casting (handles accidental "" for numeric args)
# - Robust tensor casting for Data.sample() outputs (fixes masking / indexing bugs)
# - FIX for your crash:
#     "Trying to backward through the graph a second time"
#   Root cause was caching latent adj/reg WITH autograd graph and reusing it across batches.
#   => Training mode now REBUILDS latent hypergraph per batch (trainable, safe).
#   => Inference/eval mode caches under torch.no_grad() (fast, safe).
# - Safe self-exclusion in KD neighbor pools
#
# Improvements added (to help KD actually beat baseline):
# - Router outputs are L2-normalized so logits behave like cosine similarity (aligns with teacher).
# - Inference weighting temperature uses args.infer_tau if present; otherwise defaults to 1.0
#   (keeps kd_tau for teacher distillation, avoids overly-peaky inference weights).
#
# Restored behavior you asked for:
# 1) ALWAYS save recommendations under: output/<run_tag>/...
#    - If --run_tag is empty, auto-generate a unique run_tag (so parallel runs never overlap).
# 2) Recommendation filename pattern matches the older style:
#    output/<run_tag>/recommend_<method>_<run_tag>_sim5.0_a..._kd..._<fold>.json
#    + KD filenames include pool/w/r to match your metric.py naming expectation.
# 3) After saving recommendations, automatically run metrics_single.py
#    and print the metrics to stdout (so they appear in your nohup logs).
# ------------------------------------------------------------

import os
import sys
import json
import h5py
import shutil
import warnings
import subprocess
from datetime import datetime
from time import time
from collections import deque

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy.spatial.distance import cosine
import scipy.sparse as sp

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from Models import HCF
from utility.helper import ensureDir
from utility.parser import parse_args
from utility.load_data import Data

warnings.filterwarnings("ignore")


# ============================================================
# Safe arg casting helpers (important if run.py passes "")
# ============================================================
def _to_int(v, default: int) -> int:
    if v is None:
        return int(default)
    if isinstance(v, str) and v.strip() == "":
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _to_float(v, default: float) -> float:
    if v is None:
        return float(default)
    if isinstance(v, str) and v.strip() == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_long_tensor(x, device):
    """Convert list/np/torch -> torch.long tensor on device."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=torch.long)
    return torch.tensor(x, device=device, dtype=torch.long)


# ============================================================
# Args + GPU
# ============================================================
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(args, "gpu_id", 0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Data (fold-specific) (backward-safe init)
# ============================================================
_skip_flag = _to_int(getattr(args, "skip_tag_loss_when_missing", 0), 0)
try:
    data_generator = Data(
        path=args.data_path + args.dataset,
        batch_size=_to_int(getattr(args, "batch_size", 512), 512),
        skip_tag_loss_when_missing=_skip_flag,
    )
except TypeError:
    data_generator = Data(
        path=args.data_path + args.dataset,
        batch_size=_to_int(getattr(args, "batch_size", 512), 512),
    )

USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_API_TAG, N_MASHUP_TAG = data_generator.n_api_tag, data_generator.n_mashup_tag
N_API, N_MASHUP = data_generator.n_api, data_generator.n_mashup

alpha1, alpha2, alpha3, alpha4 = args.alpha1, args.alpha2, args.alpha3, args.alpha4


# ============================================================
# Helpers
# ============================================================
def lamb(epoch: int) -> float:
    return 0.95 ** (epoch / 14)


def jaccard_similarity(matrix: np.ndarray) -> np.ndarray:
    intersection = np.dot(matrix, matrix.T)
    square_sum = np.diag(intersection)
    union = square_sum[:, None] + square_sum - intersection
    return np.divide(intersection, union + 1e-12)


def cosine_similarity_scipy(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return 1.0 - cosine(vec1, vec2)


def _gpu_mem() -> str:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserv = torch.cuda.memory_reserved() / (1024 ** 3)
        return f"{alloc:.2f}G/{reserv:.2f}G"
    return "NA"


def sparse_stats(name: str, t: torch.Tensor) -> str:
    if not t.is_sparse:
        v = t.detach()
        return (
            f"{name}: dense shape={tuple(v.shape)} "
            f"min={v.min().item():.4e} max={v.max().item():.4e} "
            f"mean={v.mean().item():.4e} std={v.std().item():.4e} "
            f"l2={torch.norm(v).item():.4e}"
        )

    v = t._values()
    if v.numel() == 0:
        return f"{name}: sparse shape={tuple(t.shape)} nnz=0"

    return (
        f"{name}: sparse shape={tuple(t.shape)} nnz={v.numel()} "
        f"min={v.min().item():.4e} max={v.max().item():.4e} "
        f"mean={v.mean().item():.4e} std={v.std().item():.4e} "
        f"l2={torch.norm(v).item():.4e}"
    )


def embed_stats(name: str, x: torch.Tensor) -> str:
    with torch.no_grad():
        v = x.detach()
        row_norm = torch.norm(v, dim=1)
        return (
            f"{name}: shape={tuple(v.shape)} "
            f"rowNorm(mean)={row_norm.mean().item():.4e} rowNorm(max)={row_norm.max().item():.4e} "
            f"min={v.min().item():.4e} max={v.max().item():.4e} mean={v.mean().item():.4e}"
        )


def _safe_tag(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _fmt_float_short(v: float) -> str:
    # stable short string: 0.03, 0.1, 5, 1e-3, etc.
    try:
        return f"{float(v):.6g}"
    except Exception:
        return str(v)


def _auto_run_tag() -> str:
    """
    If user didn't pass --run_tag, generate a unique one so parallel runs never overlap.
    Matches your older naming style + adds timestamp for uniqueness.
    """
    method = str(getattr(args, "method", "baseline"))
    ds = str(getattr(args, "dataset", "fold_1"))

    tag = method
    if method == "kd":
        tag = (
            f"kd_w{_to_int(getattr(args,'kd_warmup_epochs',0),0)}"
            f"_r{_to_int(getattr(args,'kd_ramp_epochs',0),0)}"
            f"_l{_fmt_float_short(_to_float(getattr(args,'kd_lambda',0.0),0.0))}"
            f"_tau{_fmt_float_short(_to_float(getattr(args,'kd_tau',0.1),0.1))}"
            f"_topk{_to_int(getattr(args,'kd_topk',100),100)}"
            f"_every{_to_int(getattr(args,'kd_every',5),5)}"
            f"_pool{_to_int(getattr(args,'kd_pool_mult',3),3)}"
        )

    if _to_int(getattr(args, "use_latent_hypergraph", 0), 0) == 1:
        tag += (
            f"_latent{_safe_tag(str(getattr(args,'latent_targets','user')))}"
            f"_M{_to_int(getattr(args,'hin_num_edges',256),256)}"
            f"_k{_to_int(getattr(args,'hin_topk_per_node',4),4)}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{tag}_{ds}_{ts}"
    return _safe_tag(tag)


def _proj_root() -> str:
    # keep backward compatible: if proj_path is empty, use current dir
    p = str(getattr(args, "proj_path", "") or ".")
    return p


def _output_root() -> str:
    # if output_path exists in parser, use it; else fallback to "output"
    outp = str(getattr(args, "output_path", "") or "output")
    return os.path.join(_proj_root(), outp)


# ============================================================
# KD Router Module
# ============================================================
class RouterKD(torch.nn.Module):
    """
    Student q_s: learnable projection + cosine-like logits over projected vectors.
    Teacher q_t: mixed similarity (text cosine + usage jaccard) or latent teacher (optional).
    """
    def __init__(self, dim_in: int, dim_out: int, hidden: int = 256):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(dim_in, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, dim_out),
        )

    def forward(self, query_vec: torch.Tensor, cand_vecs: torch.Tensor) -> torch.Tensor:
        q = self.proj(query_vec.unsqueeze(0))   # (1, dim_out)
        c = self.proj(cand_vecs)                # (K, dim_out)

        # Align student logits with cosine-sim teacher by normalizing.
        q = F.normalize(q, p=2, dim=1)
        c = F.normalize(c, p=2, dim=1)

        logits = torch.sum(q * c, dim=1)        # (K,) cosine-like
        return logits


# ============================================================
# Latent Hypergraph Inference Network (SPHINX-style)
# ============================================================
class LatentHIN(torch.nn.Module):
    """
    Learns a latent incidence matrix H (N x M) from node features.

    build_ops(node_feat):
      - predicts node->edge logits (N x M)
      - takes top-k edges per node
      - assigns weights
      - builds two sparse operators:
          adj2: (M x N) edge <- node  with De^{-1} normalization
          adj1: (N x M) node <- edge  with Dv^{-1} normalization
    """
    def __init__(self, dim_in: int, num_edges: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        assert layers >= 1
        self.num_edges = int(num_edges)

        net = []
        d = int(dim_in)
        for _ in range(layers - 1):
            net.append(torch.nn.Linear(d, int(hidden)))
            net.append(torch.nn.ReLU())
            net.append(torch.nn.Dropout(p=float(dropout)))
            d = int(hidden)
        net.append(torch.nn.Linear(d, self.num_edges))
        self.mlp = torch.nn.Sequential(*net)

    def build_ops(
        self,
        node_feat: torch.Tensor,
        topk_per_node: int,
        tau: float,
        straight_through: int = 1,
        reg_entropy_w: float = 0.0,
        reg_balance_w: float = 0.0,
        eps: float = 1e-12,
    ):
        N = int(node_feat.shape[0])
        M = int(self.num_edges)
        k = max(1, min(int(topk_per_node), M))

        logits = self.mlp(node_feat)  # (N, M)
        t = max(float(tau), 1e-6)

        topv, topi = torch.topk(logits, k=k, dim=1, largest=True)   # (N, k), (N, k)

        if int(straight_through) == 1:
            # soft over all, then gather topk + renorm (ST-ish, stable)
            p_all = torch.softmax(logits / t, dim=1)                # (N, M)
            weights = torch.gather(p_all, 1, topi)                  # (N, k)
            weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
        else:
            # softmax only inside selected topk
            weights = torch.softmax(topv / t, dim=1)                # (N, k)

        row = torch.arange(N, device=node_feat.device).unsqueeze(1).expand(N, k).reshape(-1)
        col = topi.reshape(-1)
        val = weights.reshape(-1)

        node_deg = torch.zeros(N, device=node_feat.device, dtype=val.dtype).scatter_add(0, row, val)
        edge_deg = torch.zeros(M, device=node_feat.device, dtype=val.dtype).scatter_add(0, col, val)

        v_norm = val / (node_deg[row] + eps)
        e_norm = val / (edge_deg[col] + eps)

        idx1 = torch.stack([row, col], dim=0)  # (2, nnz) for (N x M)
        idx2 = torch.stack([col, row], dim=0)  # (2, nnz) for (M x N)

        adj1 = torch.sparse_coo_tensor(idx1, v_norm, (N, M), device=node_feat.device).coalesce()
        adj2 = torch.sparse_coo_tensor(idx2, e_norm, (M, N), device=node_feat.device).coalesce()

        reg_loss = torch.zeros((), device=node_feat.device)

        if reg_entropy_w and reg_entropy_w > 0.0:
            ent = -torch.sum(weights * torch.log(weights + eps), dim=1).mean()
            reg_loss = reg_loss + (-float(reg_entropy_w) * ent)

        if reg_balance_w and reg_balance_w > 0.0:
            edge_mass = edge_deg / (edge_deg.sum() + eps)  # (M,)
            uniform = torch.full_like(edge_mass, 1.0 / float(M))
            reg_loss = reg_loss + (float(reg_balance_w) * torch.mean((edge_mass - uniform) ** 2))

        return adj1, adj2, reg_loss


# ============================================================
# Main Wrapper
# ============================================================
class Model_Wrapper(object):
    def __init__(self, data_config, config_mashup, config_api, pretrain_data):
        self.model_type = args.model_type
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.mess_dropout = eval(args.mess_dropout)
        self.pretrain_data = pretrain_data

        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]
        self.n_mashup = config_mashup["n_mashup"]
        self.n_mashup_tag = config_mashup["n_mashup_tag"]
        self.n_api = config_api["n_api"]
        self.n_api_tag = config_api["n_api_tag"]

        self.lr = _to_float(getattr(args, "lr", 0.002), 0.002)
        self.emb_dim = _to_int(getattr(args, "embed_size", 512), 512)
        self.batch_size = _to_int(getattr(args, "batch_size", 512), 512)
        self.layer_num = _to_int(getattr(args, "layer_num", 3), 3)

        self.model_type += "_%s_%s_layers%d" % (self.adj_type, self.alg_type, self.layer_num)
        self.regs = eval(args.regs)
        self.decay = float(self.regs[0])
        self.verbose = args.verbose

        # ---------------- KD controls ----------------
        self.method = getattr(args, "method", "baseline")
        self.use_kd_router = (_to_int(getattr(args, "use_kd_router", 0), 0) == 1)

        self.kd_lambda = _to_float(getattr(args, "kd_lambda", 0.1), 0.1)
        self.kd_tau = _to_float(getattr(args, "kd_tau", 0.1), 0.1)
        self.kd_topk = _to_int(getattr(args, "kd_topk", 100), 100)
        self.kd_every = _to_int(getattr(args, "kd_every", 5), 5)
        self.router_hidden = _to_int(getattr(args, "router_hidden", 256), 256)

        self.kd_pool_mult = _to_int(getattr(args, "kd_pool_mult", 3), 3)
        self.kd_warmup_epochs = _to_int(getattr(args, "kd_warmup_epochs", 0), 0)
        self.kd_ramp_epochs = _to_int(getattr(args, "kd_ramp_epochs", 0), 0)
        self.kd_teacher_mix = _to_float(getattr(args, "kd_teacher_mix", 1.0), 1.0)

        self.rerank_topM = _to_int(getattr(args, "rerank_topM", 0), 0)

        # Inference temperature for router neighbor weighting (separate from kd_tau).
        self.infer_tau = _to_float(getattr(args, "infer_tau", 1.0), 1.0)

        if self.method != "kd":
            self.use_kd_router = False

        # ---------------- Latent hypergraph flags ----------------
        self.use_latent_hg = (_to_int(getattr(args, "use_latent_hypergraph", 0), 0) == 1)
        self.latent_targets = str(getattr(args, "latent_targets", "user")).lower()

        self.hin_num_edges = _to_int(getattr(args, "hin_num_edges", 256), 256)
        self.hin_topk_per_node = _to_int(getattr(args, "hin_topk_per_node", 4), 4)
        self.hin_tau = _to_float(getattr(args, "hin_tau", 0.5), 0.5)
        self.hin_straight_through = _to_int(getattr(args, "hin_straight_through", 1), 1)

        self.hin_hidden = _to_int(getattr(args, "hin_hidden", 256), 256)
        self.hin_layers = _to_int(getattr(args, "hin_layers", 2), 2)
        self.hin_dropout = _to_float(getattr(args, "hin_dropout", 0.1), 0.1)
        self.hin_warmup_epochs = _to_int(getattr(args, "hin_warmup_epochs", 0), 0)
        self.hin_refresh_every = _to_int(getattr(args, "hin_refresh_every", 1), 1)

        self.hin_reg_entropy = _to_float(getattr(args, "hin_reg_entropy", 0.0), 0.0)
        self.hin_reg_balance = _to_float(getattr(args, "hin_reg_balance", 0.0), 0.0)
        self.hin_reg_l2 = _to_float(getattr(args, "hin_reg_l2", 0.0), 0.0)

        self.kd_teacher_from_latent = (_to_int(getattr(args, "kd_teacher_from_latent", 0), 0) == 1)

        print(f"model_type is {self.model_type}")
        print(f"[PIPELINE] method={self.method}, use_kd_router={self.use_kd_router}, kd_teacher_mix={self.kd_teacher_mix}")
        if self.use_kd_router:
            print(f"[KD] infer_tau={self.infer_tau}")
        if self.use_latent_hg:
            print(
                f"[LATENT-HG] enabled | targets={self.latent_targets} | M={self.hin_num_edges} | topk/node={self.hin_topk_per_node} | "
                f"tau={self.hin_tau} | ST={self.hin_straight_through} | warmup={self.hin_warmup_epochs} | refresh_every={self.hin_refresh_every} | "
                f"reg(ent)={self.hin_reg_entropy} reg(balance)={self.hin_reg_balance} reg(l2)={self.hin_reg_l2}"
            )

        # ---------------- weights path ----------------
        self.weights_save_path = (
            "%sweights/%s/%s/emb_size%s/layer_num%s/mess_dropout%s/drop_edge%s/lr%s/reg%s"
            % (
                args.weights_path, args.dataset, self.model_type,
                str(args.embed_size), str(args.layer_num), str(args.mess_dropout),
                str(args.drop_edge), str(args.lr),
                "-".join([str(r) for r in eval(args.regs)]),
            )
        )

        # ---------------- model ----------------
        if self.alg_type in ["hcf"]:
            self.model = HCF(
                self.n_users, self.n_items,
                self.n_mashup, self.n_mashup_tag,
                self.n_api, self.n_api_tag,
                self.emb_dim, self.layer_num, self.mess_dropout
            )
        else:
            raise Exception("Dont know which model to train")

        # Actual embedding dim inside model (Models.py may override based on H5)
        self.model_dim = int(getattr(self.model, "embedding_dim", self.emb_dim))

        # ---------------- baseline hypergraph edges ----------------
        (
            self.norm_u1, self.norm_u2,
            self.norm_i1, self.norm_i2,
            self.norm_m1, self.norm_m2,
            self.norm_a1, self.norm_a2,
        ) = self.build_hyper_edge(
            args.data_path + args.dataset + "/TE.csv",
            args.data_path + args.dataset + "/mashup_tags.csv",
            args.data_path + args.dataset + "/api_tags.csv",
        )

        # ---------------- move to device ----------------
        self.model = self.model.to(device)

        self.norm_u1 = self.norm_u1.to(device)
        self.norm_u2 = self.norm_u2.to(device)
        self.norm_i1 = self.norm_i1.to(device)
        self.norm_i2 = self.norm_i2.to(device)
        self.norm_m1 = self.norm_m1.to(device)
        self.norm_m2 = self.norm_m2.to(device)
        self.norm_a1 = self.norm_a1.to(device)
        self.norm_a2 = self.norm_a2.to(device)

        print("\n[MATRIX STATS] Hypergraph matrices (baseline / heuristic):")
        print(sparse_stats("norm_u1", self.norm_u1))
        print(sparse_stats("norm_u2", self.norm_u2))
        print(sparse_stats("norm_i1", self.norm_i1))
        print(sparse_stats("norm_i2", self.norm_i2))
        print(sparse_stats("norm_m1", self.norm_m1))
        print(sparse_stats("norm_m2", self.norm_m2))
        print(sparse_stats("norm_a1", self.norm_a1))
        print(sparse_stats("norm_a2", self.norm_a2))
        print()

        # ---------------- KD caches ----------------
        self.router = None
        self.mpnet_dim = None

        self.train_mashup_ids = list(getattr(data_generator, "train_items", {}).keys())
        self.train_vectors = self._load_vectors_bulk(self.train_mashup_ids)

        self.train_usage_sets = {}
        train_items = getattr(data_generator, "train_items", {})
        for mid, apis in train_items.items():
            try:
                self.train_usage_sets[int(mid)] = set(map(int, apis))
            except Exception:
                self.train_usage_sets[int(mid)] = set()

        if self.use_kd_router:
            if len(self.train_mashup_ids) == 0:
                raise RuntimeError("[KD] use_kd_router=1 but train_mashup_ids is empty.")
            any_train_id = next(iter(self.train_mashup_ids))
            vec = self.train_vectors.get(int(any_train_id), None)
            if vec is None:
                raise RuntimeError("[KD] vectors.h5 missing MPNet vector for a train mashup id.")
            self.mpnet_dim = int(vec.shape[0])

            self.router = RouterKD(
                dim_in=self.mpnet_dim,
                dim_out=self.model_dim,
                hidden=self.router_hidden,
            ).to(device)

            print(f"[KD] RouterKD initialized: mpnet_dim={self.mpnet_dim}, dim_out={self.model_dim}, hidden={self.router_hidden}")
            print(f"[KD] pool_mult={self.kd_pool_mult}, warmup={self.kd_warmup_epochs}, ramp={self.kd_ramp_epochs}, rerank_topM={self.rerank_topM}")

        # ---------------- latent HG modules ----------------
        self.hin_user = None
        self.hin_item = None
        self.hin_mtag = None
        self.hin_atag = None

        if self.use_latent_hg:
            if self._is_latent("user"):
                self.hin_user = LatentHIN(self.model_dim, self.hin_num_edges, self.hin_hidden, self.hin_layers, self.hin_dropout).to(device)
            if self._is_latent("item"):
                self.hin_item = LatentHIN(self.model_dim, self.hin_num_edges, self.hin_hidden, self.hin_layers, self.hin_dropout).to(device)
            if self._is_latent("mtag"):
                self.hin_mtag = LatentHIN(self.model_dim, self.hin_num_edges, self.hin_hidden, self.hin_layers, self.hin_dropout).to(device)
            if self._is_latent("atag"):
                self.hin_atag = LatentHIN(self.model_dim, self.hin_num_edges, self.hin_hidden, self.hin_layers, self.hin_dropout).to(device)

        # Cache for inference-only latent adjacency (safe because built under torch.no_grad()).
        self._latent_cache = {}  # name -> {"epoch": int, "adj": (adj1, adj2)}

        # For convenience (so train() can report where it wrote results)
        self.last_output_file = None

        optim_params = list(self.model.parameters())
        if self.router is not None:
            optim_params += list(self.router.parameters())
        if self.hin_user is not None:
            optim_params += list(self.hin_user.parameters())
        if self.hin_item is not None:
            optim_params += list(self.hin_item.parameters())
        if self.hin_mtag is not None:
            optim_params += list(self.hin_mtag.parameters())
        if self.hin_atag is not None:
            optim_params += list(self.hin_atag.parameters())

        self.optimizer = optim.Adam(optim_params, lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()

    # ------------------------------------------------------------
    def _is_latent(self, name: str) -> bool:
        if not self.use_latent_hg:
            return False
        t = self.latent_targets
        return (t == "all") or (t == str(name).lower())

    # ------------------------------------------------------------
    def _load_vectors_bulk(self, ids):
        out = {}
        path = "data/vectors.h5"
        if not os.path.exists(path):
            print(f"[WARN] vectors file not found: {path}")
            return out

        with h5py.File(path, "r") as f:
            for mid in ids:
                k = str(int(mid))
                if k in f:
                    out[int(mid)] = f[k][:]
        return out

    # ------------------------------------------------------------
    def get_D_inv(self, Hadj):
        H = sp.coo_matrix(Hadj.shape)
        H.row = Hadj.row.copy()
        H.col = Hadj.col.copy()
        H.data = Hadj.data.copy()

        rowsum = np.array(H.sum(1))
        columnsum = np.array(H.sum(0))

        Dv_inv = np.power(rowsum, -1).flatten()
        De_inv = np.power(columnsum, -1).flatten()
        Dv_inv[np.isinf(Dv_inv)] = 0.0
        De_inv[np.isinf(De_inv)] = 0.0

        Dv_mat_inv = sp.diags(Dv_inv)
        De_mat_inv = sp.diags(De_inv)
        return Dv_mat_inv, De_mat_inv

    # ------------------------------------------------------------
    def build_hyper_edge(self, file, file_mashup, file_api):
        user_inter = np.zeros((USR_NUM, ITEM_NUM), dtype=np.float32)
        items_inter = np.zeros((ITEM_NUM, USR_NUM), dtype=np.float32)
        mashup_inter = np.zeros((N_MASHUP_TAG, N_MASHUP), dtype=np.float32)
        api_inter = np.zeros((N_API_TAG, N_API), dtype=np.float32)

        with open(file, encoding="utf-8") as f:
            for l in f:
                parts = [p for p in l.strip("\n").split(" ") if p != ""]
                if len(parts) <= 1:
                    continue
                uid = int(parts[0])
                items = [int(j) for j in parts[1:]]
                user_inter[uid, items] = 1.0
                items_inter[items, uid] = 1.0

        with open(file_mashup, encoding="utf-8") as f:
            for l in f:
                parts = [p for p in l.strip("\n").split(" ") if p != ""]
                if len(parts) <= 1:
                    continue
                uid = int(parts[0])
                tags = [int(j) for j in parts[1:]]
                mashup_inter[tags, uid] = 1.0

        with open(file_api, encoding="utf-8") as f:
            for l in f:
                parts = [p for p in l.strip("\n").split(" ") if p != ""]
                if len(parts) <= 1:
                    continue
                uid = int(parts[0])
                tags = [int(j) for j in parts[1:]]
                api_inter[tags, uid] = 1.0

        # Users
        J_u = jaccard_similarity(user_inter)
        indices = np.where(J_u >= alpha1)
        print(f"Number of hyperedges for users: {len(indices[0])}")
        values = J_u[indices]
        HEdge_u = sp.coo_matrix((values, indices), (USR_NUM, USR_NUM))
        HuEdge = HEdge_u.T
        Dv_u, De_u = self.get_D_inv(HuEdge)
        norm_u1_sp = (Dv_u @ HuEdge) @ De_u
        norm_u2_sp = HEdge_u

        # Items/APIs
        J_i = jaccard_similarity(items_inter)
        indices = np.where(J_i >= alpha2)
        values = J_i[indices]
        print(f"Number of hyperedges for items: {len(values)}")
        HEdge_i = sp.coo_matrix((values, indices), (ITEM_NUM, ITEM_NUM))
        HiEdge = HEdge_i.T
        Dv_i, De_i = self.get_D_inv(HiEdge)
        norm_i1_sp = (Dv_i @ HiEdge) @ De_i
        norm_i2_sp = HEdge_i

        # Mashup-tag
        J_m = jaccard_similarity(mashup_inter)
        indices = np.where(J_m >= alpha3)
        values = J_m[indices]
        print(f"Number of hyperedges for mashup_tag: {len(values)}")
        HEdge_m = sp.coo_matrix((values, indices), (N_MASHUP_TAG, N_MASHUP_TAG))
        HmEdge = HEdge_m.T
        Dv_m, De_m = self.get_D_inv(HmEdge)
        norm_m1_sp = (Dv_m @ HmEdge) @ De_m
        norm_m2_sp = HEdge_m

        # API-tag
        J_a = jaccard_similarity(api_inter)
        indices = np.where(J_a >= alpha4)
        values = J_a[indices]
        print(f"Number of hyperedges for api tag: {len(values)}")
        HEdge_a = sp.coo_matrix((values, indices), (N_API_TAG, N_API_TAG))
        HaEdge = HEdge_a.T
        Dv_a, De_a = self.get_D_inv(HaEdge)
        norm_a1_sp = (Dv_a @ HaEdge) @ De_a
        norm_a2_sp = HEdge_a

        return (
            self.sparse_mx_to_torch_sparse_tensor(norm_u1_sp),
            self.sparse_mx_to_torch_sparse_tensor(norm_u2_sp),
            self.sparse_mx_to_torch_sparse_tensor(norm_i1_sp),
            self.sparse_mx_to_torch_sparse_tensor(norm_i2_sp),
            self.sparse_mx_to_torch_sparse_tensor(norm_m1_sp),
            self.sparse_mx_to_torch_sparse_tensor(norm_m2_sp),
            self.sparse_mx_to_torch_sparse_tensor(norm_a1_sp),
            self.sparse_mx_to_torch_sparse_tensor(norm_a2_sp),
        )

    # ------------------------------------------------------------
    def set_lr_scheduler(self):
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lamb)

    # ------------------------------------------------------------
    def save_model(self):
        ensureDir(self.weights_save_path)
        torch.save(self.model.state_dict(), self.weights_save_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.weights_save_path, map_location=device))

    # ========================= Latent HG: choose adj matrices =========================
    def _get_adj_pair(self, name: str, epoch: int, training: bool = True):
        """
        IMPORTANT FIX:
        - If training=True: rebuild latent HG per batch (trainable, no graph reuse).
        - If training=False: build under torch.no_grad() and cache per epoch/refresh_every (fast, safe).
        """
        reg = torch.zeros((), device=device)

        if not self.use_latent_hg:
            return self._get_baseline_adj_pair(name), reg
        if int(epoch) < int(self.hin_warmup_epochs):
            return self._get_baseline_adj_pair(name), reg
        if not self._is_latent(name):
            return self._get_baseline_adj_pair(name), reg

        # Select module + features
        if name == "user":
            hin = self.hin_user
            feat = self.model.user_embedding.weight.detach()
        elif name == "item":
            hin = self.hin_item
            feat = self.model.item_embedding.weight.detach()
        elif name == "mtag":
            hin = self.hin_mtag
            feat = self.model.mashup_tag_embedding.weight.detach()
        elif name == "atag":
            hin = self.hin_atag
            feat = self.model.api_tag_embedding.weight.detach()
        else:
            return self._get_baseline_adj_pair(name), reg

        if hin is None:
            return self._get_baseline_adj_pair(name), reg

        # ---------------- TRAINING: rebuild every call (per batch) ----------------
        if training:
            hin.train()
            adj1, adj2, reg_lat = hin.build_ops(
                feat,
                topk_per_node=self.hin_topk_per_node,
                tau=self.hin_tau,
                straight_through=self.hin_straight_through,
                reg_entropy_w=self.hin_reg_entropy,
                reg_balance_w=self.hin_reg_balance,
            )
            reg = reg + reg_lat

            if self.hin_reg_l2 and self.hin_reg_l2 > 0.0:
                l2 = torch.zeros((), device=device)
                for p in hin.parameters():
                    l2 = l2 + torch.sum(p * p)
                reg = reg + (float(self.hin_reg_l2) * l2)

            return (adj1, adj2), reg

        # ---------------- INFERENCE/EVAL: cache under no_grad ----------------
        hin.eval()

        cache = self._latent_cache.get(name, None)
        if cache is not None:
            last_ep = int(cache["epoch"])
            if self.hin_refresh_every == 0:
                return cache["adj"], reg
            if last_ep == int(epoch):
                return cache["adj"], reg
            if int(epoch) % int(self.hin_refresh_every) != 0:
                return cache["adj"], reg

        with torch.no_grad():
            adj1, adj2, _ = hin.build_ops(
                feat,
                topk_per_node=self.hin_topk_per_node,
                tau=self.hin_tau,
                straight_through=self.hin_straight_through,
                reg_entropy_w=0.0,
                reg_balance_w=0.0,
            )
        self._latent_cache[name] = {"epoch": int(epoch), "adj": (adj1, adj2)}
        return (adj1, adj2), reg

    def _get_baseline_adj_pair(self, name: str):
        if name == "user":
            return (self.norm_u1, self.norm_u2)
        if name == "item":
            return (self.norm_i1, self.norm_i2)
        if name == "mtag":
            return (self.norm_m1, self.norm_m2)
        if name == "atag":
            return (self.norm_a1, self.norm_a2)
        raise ValueError(f"Unknown adj name: {name}")

    # ========================= KD teacher (text + usage) =========================
    def _usage_jaccard(self, mid1: int, mid2: int) -> float:
        a = self.train_usage_sets.get(int(mid1), set())
        b = self.train_usage_sets.get(int(mid2), set())
        if not a or not b:
            return 0.0
        inter = len(a & b)
        uni = len(a) + len(b) - inter
        return (inter / uni) if uni > 0 else 0.0

    def _mixed_teacher_sims(self, qid: int, q_vec_np: np.ndarray, cand_ids: list, cand_vecs_np: np.ndarray) -> np.ndarray:
        cos_sims = np.array(
            [cosine_similarity_scipy(q_vec_np, cand_vecs_np[i]) for i in range(cand_vecs_np.shape[0])],
            dtype=np.float32,
        )

        mix = float(self.kd_teacher_mix)
        mix = max(0.0, min(1.0, mix))
        if mix >= 0.999:
            return cos_sims

        jac = np.array([self._usage_jaccard(qid, int(cid)) for cid in cand_ids], dtype=np.float32)
        sims = (mix * cos_sims) + ((1.0 - mix) * jac)
        return sims.astype(np.float32)

    def _latent_teacher_sims(self, q_vec_np: np.ndarray, cand_vecs_np: np.ndarray) -> np.ndarray:
        if self.hin_user is None:
            return None

        self.hin_user.eval()
        with torch.no_grad():
            q = torch.tensor(q_vec_np, dtype=torch.float32, device=device).unsqueeze(0)   # (1,D)
            c = torch.tensor(cand_vecs_np, dtype=torch.float32, device=device)            # (K,D)

            logits_q = self.hin_user.mlp(q)  # (1,M)
            logits_c = self.hin_user.mlp(c)  # (K,M)

            t = max(float(self.hin_tau), 1e-6)
            p_q = torch.softmax(logits_q / t, dim=1)        # (1,M)
            p_c = torch.softmax(logits_c / t, dim=1)        # (K,M)

            sims = torch.sum(p_c * p_q, dim=1)              # (K,)
            return sims.detach().cpu().numpy().astype(np.float32)

    def _teacher_distribution_from_sims(self, sims_np: np.ndarray) -> torch.Tensor:
        temp = max(float(self.kd_tau), 1e-6)
        logits = sims_np / temp
        logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
        return torch.softmax(logits_t, dim=0)

    def _student_distribution(self, query_vec_t, cand_vecs_t):
        logits = self.router(query_vec_t, cand_vecs_t)
        return torch.softmax(logits, dim=0)

    def _candidate_pool(self, q_vec_np: np.ndarray, qid: int, pool_size: int):
        if self.rerank_topM and self.rerank_topM > 0:
            topM = min(int(self.rerank_topM), len(self.train_mashup_ids))
            sim_list = []
            for other_user, ov in self.train_vectors.items():
                if int(other_user) == int(qid):
                    continue
                sim_list.append((cosine_similarity_scipy(q_vec_np, ov), int(other_user)))
            sim_list.sort(key=lambda x: x[0], reverse=True)
            return [mid for _, mid in sim_list[:topM]]

        cand_ids_raw = np.random.choice(self.train_mashup_ids, size=pool_size, replace=False).tolist()
        if int(qid) in cand_ids_raw:
            cand_ids_raw.remove(int(qid))
        return cand_ids_raw

    def compute_router_kd_loss(self):
        if self.router is None:
            return torch.zeros((), device=device)

        n_queries = min(16, len(self.train_mashup_ids))
        if n_queries <= 0:
            return torch.zeros((), device=device)

        query_ids = np.random.choice(self.train_mashup_ids, size=n_queries, replace=False)

        total_kd = torch.zeros((), device=device)
        used = 0

        for qid in query_ids:
            qid = int(qid)
            q_vec_np = self.train_vectors.get(qid, None)
            if q_vec_np is None:
                continue

            pool_mult = max(1, int(self.kd_pool_mult))
            pool_size = min(max(self.kd_topk * pool_mult, self.kd_topk + 1), len(self.train_mashup_ids))

            cand_ids_raw = self._candidate_pool(q_vec_np, qid=qid, pool_size=pool_size)

            cand_ids = []
            cand_vecs = []
            for cid in cand_ids_raw:
                cid = int(cid)
                v = self.train_vectors.get(cid, None)
                if v is not None:
                    cand_ids.append(cid)
                    cand_vecs.append(v)

            if len(cand_vecs) == 0:
                continue

            cand_vecs_np = np.stack(cand_vecs, axis=0)

            sims = None
            if self.kd_teacher_from_latent and self.hin_user is not None:
                sims = self._latent_teacher_sims(q_vec_np, cand_vecs_np)

            if sims is None:
                sims = self._mixed_teacher_sims(qid, q_vec_np, cand_ids, cand_vecs_np)

            topk = min(self.kd_topk, sims.shape[0])
            top_idx = np.argsort(sims)[-topk:][::-1]

            top_sims = sims[top_idx]
            top_cand_vecs_np = cand_vecs_np[top_idx]

            q_t = self._teacher_distribution_from_sims(top_sims)

            q_vec_t = torch.tensor(q_vec_np, dtype=torch.float32, device=device)
            cand_vecs_t = torch.tensor(top_cand_vecs_np, dtype=torch.float32, device=device)
            q_s = self._student_distribution(q_vec_t, cand_vecs_t)

            eps = 1e-12
            kd = torch.sum(q_t * (torch.log(q_t + eps) - torch.log(q_s + eps)))
            total_kd = total_kd + kd
            used += 1

        if used > 0:
            total_kd = total_kd / used

        return total_kd

    def kd_weight(self, epoch: int) -> float:
        if not self.use_kd_router:
            return 0.0

        base = float(self.kd_lambda)
        warm = int(self.kd_warmup_epochs)
        ramp = int(self.kd_ramp_epochs)

        if warm > 0 and epoch < warm:
            return 0.0

        if ramp and ramp > 0:
            t = (epoch - warm) / float(ramp)
            t = max(0.0, min(1.0, t))
            return base * t

        return base

    # ========================= Losses =========================
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)

        regularizer = 0.5 * (users ** 2).sum() + 0.5 * (pos_items ** 2).sum() + 0.5 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        emb_loss = self.decay * regularizer
        reg_loss = torch.zeros((), device=device)
        return mf_loss, emb_loss, reg_loss

    def tag_loss(self, embeddings, pos_tags, neg_tags):
        pos_scores = torch.sum(embeddings * pos_tags, dim=1)
        neg_scores = torch.sum(embeddings * neg_tags, dim=1)

        regularizer = (
            0.5 * (embeddings ** 2).sum()
            + 0.5 * (pos_tags ** 2).sum()
            + 0.5 * (neg_tags ** 2).sum()
        )
        regularizer = regularizer / self.batch_size

        tag_mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        tag_emb_loss = self.decay * regularizer
        reg_loss = torch.zeros((), device=device)
        return tag_mf_loss, tag_emb_loss, reg_loss

    # ========================= Train =========================
    def train(self):
        n_batch = data_generator.n_train // self.batch_size + 1

        print_embed_every = _to_int(getattr(args, "print_embed_every", 1), 1)
        print_hgraph_every = _to_int(getattr(args, "print_hgraph_every", 0), 0)

        use_pbar = (tqdm is not None)
        win = 30
        ma_total = deque(maxlen=win)
        ma_mf = deque(maxlen=win)
        ma_emb = deque(maxlen=win)
        ma_mtag = deque(maxlen=win)
        ma_atag = deque(maxlen=win)
        ma_kd = deque(maxlen=win)

        skip_missing = (_to_int(getattr(args, "skip_tag_loss_when_missing", 0), 0) == 1)

        def _lr():
            return self.optimizer.param_groups[0]["lr"]

        for epoch in range(_to_int(getattr(args, "epoch", 1), 1)):
            t_epoch = time()

            loss_sum = 0.0
            mf_sum = 0.0
            emb_sum = 0.0
            mtag_sum = 0.0
            atag_sum = 0.0
            kd_loss_sum = 0.0
            kd_steps = 0
            latent_reg_sum = 0.0

            it = range(n_batch)
            if use_pbar:
                bar_width = min(120, shutil.get_terminal_size((120, 20)).columns)
                it = tqdm(it, total=n_batch, ncols=bar_width, desc=f"Epoch {epoch}", leave=False)

            w_kd_epoch = self.kd_weight(epoch)

            for idx in it:
                self.model.train()
                if self.router is not None:
                    self.router.train()
                if self.hin_user is not None:
                    self.hin_user.train()
                if self.hin_item is not None:
                    self.hin_item.train()
                if self.hin_mtag is not None:
                    self.hin_mtag.train()
                if self.hin_atag is not None:
                    self.hin_atag.train()

                self.optimizer.zero_grad()

                users, pos_items, neg_items, pos_mashup_tags, neg_mashup_tags, pos_api_tags, neg_api_tags = (
                    data_generator.sample()
                )

                # Ensure all indices/tags are torch.long on device
                users = _to_long_tensor(users, device)
                pos_items = _to_long_tensor(pos_items, device)
                neg_items = _to_long_tensor(neg_items, device)
                pos_mashup_tags = _to_long_tensor(pos_mashup_tags, device)
                neg_mashup_tags = _to_long_tensor(neg_mashup_tags, device)
                pos_api_tags = _to_long_tensor(pos_api_tags, device)
                neg_api_tags = _to_long_tensor(neg_api_tags, device)

                # Choose hypergraphs (baseline or latent) -- training=True rebuilds latent safely per batch
                (u1, u2), reg_u = self._get_adj_pair("user", epoch, training=True)
                (i1, i2), reg_i = self._get_adj_pair("item", epoch, training=True)
                (m1, m2), reg_m = self._get_adj_pair("mtag", epoch, training=True)
                (a1, a2), reg_a = self._get_adj_pair("atag", epoch, training=True)
                reg_latent = reg_u + reg_i + reg_m + reg_a

                ua_embeddings, ia_embeddings, ma_embeddings, aa_embeddings = self.model(
                    u1, u2, i1, i2, m1, m2, a1, a2
                )

                # -------- main BPR loss --------
                u_g = ua_embeddings[users]
                pos_i_g = ia_embeddings[pos_items]
                neg_i_g = ia_embeddings[neg_items]
                batch_mf_loss, batch_emb_loss, _ = self.bpr_loss(u_g, pos_i_g, neg_i_g)

                # -------- tag losses (optionally skip missing) --------
                batch_mtag_mf = torch.zeros((), device=device)
                batch_mtag_emb = torch.zeros((), device=device)
                if not skip_missing:
                    pos_m_g = ma_embeddings[pos_mashup_tags]
                    neg_m_g = ma_embeddings[neg_mashup_tags]
                    batch_mtag_mf, batch_mtag_emb, _ = self.tag_loss(u_g, pos_m_g, neg_m_g)
                else:
                    mask = (pos_mashup_tags >= 0) & (neg_mashup_tags >= 0)
                    if torch.any(mask):
                        pos_m_g = ma_embeddings[pos_mashup_tags[mask]]
                        neg_m_g = ma_embeddings[neg_mashup_tags[mask]]
                        u_g_m = u_g[mask]
                        batch_mtag_mf, batch_mtag_emb, _ = self.tag_loss(u_g_m, pos_m_g, neg_m_g)

                batch_atag_mf = torch.zeros((), device=device)
                batch_atag_emb = torch.zeros((), device=device)
                if not skip_missing:
                    pos_a_g = aa_embeddings[pos_api_tags]
                    neg_a_g = aa_embeddings[neg_api_tags]
                    batch_atag_mf, batch_atag_emb, _ = self.tag_loss(pos_i_g, pos_a_g, neg_a_g)
                else:
                    mask = (pos_api_tags >= 0) & (neg_api_tags >= 0)
                    if torch.any(mask):
                        pos_a_g = aa_embeddings[pos_api_tags[mask]]
                        neg_a_g = aa_embeddings[neg_api_tags[mask]]
                        pos_i_g_a = pos_i_g[mask]
                        batch_atag_mf, batch_atag_emb, _ = self.tag_loss(pos_i_g_a, pos_a_g, neg_a_g)

                batch_loss = (
                    batch_mf_loss + batch_emb_loss
                    + batch_mtag_mf + batch_mtag_emb
                    + batch_atag_mf + batch_atag_emb
                )

                if self.use_latent_hg:
                    batch_loss = batch_loss + reg_latent
                    latent_reg_sum += float(reg_latent.detach().item())

                # KD loss
                kd_loss = None
                if self.use_kd_router and (idx % max(self.kd_every, 1) == 0) and (w_kd_epoch > 0.0):
                    kd_loss = self.compute_router_kd_loss()
                    batch_loss = batch_loss + (w_kd_epoch * kd_loss)
                    kd_loss_sum += kd_loss.detach().item()
                    kd_steps += 1

                if not torch.isfinite(batch_loss):
                    print(f"\n[ERROR] Non-finite loss at epoch={epoch}, batch={idx}. loss={batch_loss}")
                    sys.exit(1)

                batch_loss.backward()
                self.optimizer.step()

                loss_sum += batch_loss.detach().item()
                mf_sum += batch_mf_loss.detach().item()
                emb_sum += batch_emb_loss.detach().item()
                mtag_sum += batch_mtag_mf.detach().item()
                atag_sum += batch_atag_mf.detach().item()

                ma_total.append(batch_loss.detach().item())
                ma_mf.append(batch_mf_loss.detach().item())
                ma_emb.append(batch_emb_loss.detach().item())
                ma_mtag.append(batch_mtag_mf.detach().item())
                ma_atag.append(batch_atag_mf.detach().item())
                if kd_loss is not None:
                    ma_kd.append(kd_loss.detach().item())

                if use_pbar:
                    postfix = {
                        "lr": f"{_lr():.2e}",
                        "L(ma)": f"{np.mean(ma_total):.4f}",
                        "mf": f"{np.mean(ma_mf):.4f}",
                        "emb": f"{np.mean(ma_emb):.4f}",
                        "mTag": f"{np.mean(ma_mtag):.4f}",
                        "aTag": f"{np.mean(ma_atag):.4f}",
                        "mem": _gpu_mem(),
                    }
                    if self.use_latent_hg:
                        postfix["HG"] = "latent"
                    if self.use_kd_router:
                        postfix["kdW"] = f"{w_kd_epoch:.3f}"
                        postfix["kd"] = f"{(np.mean(ma_kd) if len(ma_kd) > 0 else 0.0):.4f}"
                    it.set_postfix(postfix)

            self.lr_scheduler.step()

            avg_kd = (kd_loss_sum / max(1, kd_steps)) if (self.use_kd_router and kd_steps > 0) else 0.0
            avg_latreg = (latent_reg_sum / max(1, n_batch)) if self.use_latent_hg else 0.0

            print(
                f"Epoch {epoch:03d} done in {time() - t_epoch:.1f}s | "
                f"lr={_lr():.2e} | "
                f"loss={loss_sum/n_batch:.5f} mf={mf_sum/n_batch:.5f} emb={emb_sum/n_batch:.5f} "
                f"mTag={mtag_sum/n_batch:.5f} aTag={atag_sum/n_batch:.5f} | "
                f"{('latentReg='+format(avg_latreg, '.5f')+' | ') if self.use_latent_hg else ''}"
                f"{('kdW='+format(w_kd_epoch, '.3f')+' kd='+format(avg_kd, '.5f')+' | ') if self.use_kd_router else ''}"
                f"GPUmem={_gpu_mem()}"
            )

            if print_hgraph_every > 0 and self.use_latent_hg and (epoch % print_hgraph_every == 0):
                # show inference-cache stats (built under no_grad)
                with torch.no_grad():
                    for nm in ["user", "item", "mtag", "atag"]:
                        if self._is_latent(nm):
                            (x1, x2), _ = self._get_adj_pair(nm, epoch, training=False)
                            print(f"[LATENT HG STATS] epoch={epoch} name={nm}")
                            print(sparse_stats(f"{nm}.adj1", x1))
                            print(sparse_stats(f"{nm}.adj2", x2))

            if print_embed_every > 0 and (epoch % print_embed_every == 0):
                with torch.no_grad():
                    self.model.eval()
                    (u1, u2), _ = self._get_adj_pair("user", epoch, training=False)
                    (i1, i2), _ = self._get_adj_pair("item", epoch, training=False)
                    (m1, m2), _ = self._get_adj_pair("mtag", epoch, training=False)
                    (a1, a2), _ = self._get_adj_pair("atag", epoch, training=False)
                    ua_e, ia_e, ma_e, aa_e = self.model(u1, u2, i1, i2, m1, m2, a1, a2)

                print("[EMBED STATS]")
                print(embed_stats("ua_embeddings", ua_e))
                print(embed_stats("ia_embeddings", ia_e))
                print(embed_stats("ma_embeddings", ma_e))
                print(embed_stats("aa_embeddings", aa_e))

        if _to_int(getattr(args, "save_flag", 0), 0) == 1:
            self.save_model()
            print("save the weights in path: ", self.weights_save_path)

        if _to_int(getattr(args, "save_recom", 0), 0) == 1:
            self.save_recResult()

    # ========================= Inference neighbor selection (router) =========================
    def _router_select_neighbors(self, user_vec_np: np.ndarray, top_k: int, user_id: int = None):
        if self.router is None:
            return [], None

        top_k = int(top_k)
        if top_k <= 0:
            return [], None

        # Deterministic rerank: cosine topM -> router rerank -> pick top_k
        if self.rerank_topM and self.rerank_topM > 0 and user_id is not None:
            topM = min(int(self.rerank_topM), len(self.train_mashup_ids))
            sim_list = []
            for other_user, ov in self.train_vectors.items():
                if int(other_user) == int(user_id):
                    continue
                sim_list.append((cosine_similarity_scipy(user_vec_np, ov), int(other_user)))
            sim_list.sort(key=lambda x: x[0], reverse=True)
            cand_ids = [mid for _, mid in sim_list[:topM]]

            cand_vecs = []
            kept_ids = []
            for cid in cand_ids:
                v = self.train_vectors.get(int(cid), None)
                if v is not None:
                    kept_ids.append(int(cid))
                    cand_vecs.append(v)
            if len(cand_vecs) == 0:
                return [], None

            cand_vecs_np = np.stack(cand_vecs, axis=0)

            with torch.no_grad():
                q_t = torch.tensor(user_vec_np, dtype=torch.float32, device=device)
                c_t = torch.tensor(cand_vecs_np, dtype=torch.float32, device=device)
                logits = self.router(q_t, c_t)

                k = min(top_k, logits.numel())
                topv, topi = torch.topk(logits, k=k, largest=True)

                temp = max(float(self.infer_tau), 1e-6)
                w = torch.softmax(topv / temp, dim=0)

            sel_ids = [kept_ids[int(i)] for i in topi.detach().cpu().numpy().tolist()]
            return sel_ids, w

        # Random pool
        pool_mult = max(1, int(self.kd_pool_mult))
        pool_size = max(top_k + 1, int(self.kd_topk) * pool_mult)
        pool_size = min(pool_size, len(self.train_mashup_ids))
        if pool_size <= 0:
            return [], None

        cand_ids_raw = np.random.choice(self.train_mashup_ids, size=pool_size, replace=False).tolist()

        if user_id is not None and int(user_id) in cand_ids_raw:
            cand_ids_raw.remove(int(user_id))

        cand_ids = []
        cand_vecs = []
        for cid in cand_ids_raw:
            cid = int(cid)
            v = self.train_vectors.get(cid, None)
            if v is not None:
                cand_ids.append(cid)
                cand_vecs.append(v)

        if len(cand_vecs) == 0:
            return [], None

        cand_vecs_np = np.stack(cand_vecs, axis=0)

        with torch.no_grad():
            q_t = torch.tensor(user_vec_np, dtype=torch.float32, device=device)
            c_t = torch.tensor(cand_vecs_np, dtype=torch.float32, device=device)
            logits = self.router(q_t, c_t)

            k = min(top_k, logits.numel())
            topv, topi = torch.topk(logits, k=k, largest=True)

            temp = max(float(self.infer_tau), 1e-6)
            w = torch.softmax(topv / temp, dim=0)

        sel_ids = [cand_ids[int(i)] for i in topi.detach().cpu().numpy().tolist()]
        return sel_ids, w

    # ========================= Auto metrics (prints to stdout/log) =========================
    def _run_metrics_single(self):
        """
        Runs metrics_single.py using the SAME args (incl. run_tag) so it finds the file,
        and prints results to stdout (so nohup logs include metrics).
        """
        script = os.path.join(_proj_root(), "metrics_single.py")
        if not os.path.exists(script):
            print(f"[WARN] metrics_single.py not found at: {script} (skip auto-metrics)")
            return

        run_tag = _safe_tag(getattr(args, "run_tag", ""))
        cmd = [
            sys.executable, "-u", script,
            "--dataset", str(getattr(args, "dataset", "")),
            "--method", str(getattr(args, "method", "baseline")),
            "--similar_threthold", str(getattr(args, "similar_threthold", 5)),
            "--alpha1", str(getattr(args, "alpha1", 0.6)),
            "--alpha2", str(getattr(args, "alpha2", 0.6)),
            "--alpha3", str(getattr(args, "alpha3", 0.8)),
            "--alpha4", str(getattr(args, "alpha4", 0.8)),
            "--output_path", str(getattr(args, "output_path", "output")),
            "--proj_path", str(getattr(args, "proj_path", ".")),
        ]

        if run_tag:
            cmd += ["--run_tag", run_tag]

        if str(getattr(args, "method", "baseline")) == "kd":
            cmd += [
                "--kd_lambda", str(getattr(args, "kd_lambda", 0.0)),
                "--kd_tau", str(getattr(args, "kd_tau", 0.1)),
                "--kd_topk", str(getattr(args, "kd_topk", 100)),
                "--kd_every", str(getattr(args, "kd_every", 5)),
                "--kd_pool_mult", str(getattr(args, "kd_pool_mult", 3)),
                "--kd_warmup_epochs", str(getattr(args, "kd_warmup_epochs", 0)),
                "--kd_ramp_epochs", str(getattr(args, "kd_ramp_epochs", 0)),
            ]

        print("\n[AUTO-METRICS] Running metrics_single.py:")
        print("  " + " ".join(cmd))

        try:
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"[WARN] metrics_single.py exited with code={rc} (see logs above).")
        except Exception as e:
            print(f"[WARN] Failed to run metrics_single.py: {repr(e)}")

    # ========================= Recommend =========================
    def save_recResult(self):
        """
        Restored: save under output/<run_tag>/... so parallel experiments never overlap.
        Also restores older filename style and then auto-runs metrics_single.py (prints to logs).
        """
        self.model.eval()
        if self.router is not None:
            self.router.eval()
        if self.hin_user is not None:
            self.hin_user.eval()
        if self.hin_item is not None:
            self.hin_item.eval()
        if self.hin_mtag is not None:
            self.hin_mtag.eval()
        if self.hin_atag is not None:
            self.hin_atag.eval()

        with torch.no_grad():
            ep_ref = max(_to_int(getattr(args, "epoch", 0), 0), self.hin_warmup_epochs)
            (u1, u2), _ = self._get_adj_pair("user", epoch=ep_ref, training=False)
            (i1, i2), _ = self._get_adj_pair("item", epoch=ep_ref, training=False)
            (m1, m2), _ = self._get_adj_pair("mtag", epoch=ep_ref, training=False)
            (a1, a2), _ = self._get_adj_pair("atag", epoch=ep_ref, training=False)
            ua_embeddings, ia_embeddings, _, _ = self.model(u1, u2, i1, i2, m1, m2, a1, a2)

        users_to_test = list(getattr(data_generator, "test_set", {}).keys())

        run_tag = _safe_tag(getattr(args, "run_tag", ""))
        # Always isolate outputs by run_tag (parallel-safe)
        out_root = _output_root()
        out_dir = os.path.join(out_root, run_tag) if run_tag else out_root
        os.makedirs(out_dir, exist_ok=True)

        # match old metrics_single naming: sim as float (e.g., 5.0)
        sim_val = float(getattr(args, "similar_threthold", 5.0))
        sim_str = f"{sim_val:.1f}"

        method = str(getattr(args, "method", "baseline"))
        ds = str(getattr(args, "dataset", "fold_1"))

        # Old-style: recommend_<method>_<run_tag>_sim... (run_tag omitted if empty)
        base = f"recommend_{method}"
        if run_tag:
            base += f"_{run_tag}"

        base += f"_sim{sim_str}_a{args.alpha1}_{args.alpha2}_{args.alpha3}_{args.alpha4}"

        # IMPORTANT: include pool/w/r so metric.py's path builder matches.
        if method == "kd":
            base += (
                f"_kdL{_fmt_float_short(_to_float(getattr(args,'kd_lambda',0.0),0.0))}"
                f"_tau{_fmt_float_short(_to_float(getattr(args,'kd_tau',0.1),0.1))}"
                f"_topk{_to_int(getattr(args,'kd_topk',100),100)}"
                f"_every{_to_int(getattr(args,'kd_every',5),5)}"
                f"_pool{_to_int(getattr(args,'kd_pool_mult',3),3)}"
                f"_w{_to_int(getattr(args,'kd_warmup_epochs',0),0)}"
                f"_r{_to_int(getattr(args,'kd_ramp_epochs',0),0)}"
            )

        output_file = os.path.join(out_dir, f"{base}_{ds}.json")
        self.last_output_file = output_file

        vec_path = os.path.join(_proj_root(), "data", "vectors.h5")
        f_vec = h5py.File(vec_path, "r") if os.path.exists(vec_path) else None

        try:
            with open(output_file, mode="w", encoding="utf-8") as fp:
                for user in users_to_test:
                    if f_vec is not None and str(user) in f_vec:
                        user_vector = f_vec[str(user)][:]
                    else:
                        user_vector = self.train_vectors.get(int(user), None)
                        if user_vector is None:
                            continue

                    top_k = int(sim_val)

                    if self.method == "kd" and self.use_kd_router and self.router is not None:
                        sel_ids, w = self._router_select_neighbors(user_vector, top_k=top_k, user_id=int(user))
                        if not sel_ids:
                            continue
                        u_sim_emb = ua_embeddings[sel_ids]
                        average_embedding = torch.sum(u_sim_emb * w.unsqueeze(1), dim=0)
                    else:
                        sim_list = []
                        for other_user, ov in self.train_vectors.items():
                            if int(other_user) == int(user):
                                continue
                            sim_list.append((cosine_similarity_scipy(user_vector, ov), int(other_user)))

                        top_similar_users = sorted(sim_list, key=lambda x: x[0], reverse=True)[:top_k]
                        similar_users = [i[1] for i in top_similar_users]
                        similar = [i[0] for i in top_similar_users]
                        if len(similar_users) == 0:
                            continue

                        u_sim_emb = ua_embeddings[similar_users]
                        similar_t = torch.tensor(similar, dtype=torch.float32, device=device)
                        similar_t = similar_t / (similar_t.sum() + 1e-12)
                        average_embedding = torch.sum(u_sim_emb * similar_t.unsqueeze(1), dim=0)

                    rate_batch = torch.matmul(average_embedding, ia_embeddings.T).detach().cpu().numpy()
                    top_20_indices = np.argsort(rate_batch)[-20:][::-1]
                    fp.write(json.dumps({"mashup_id": int(user), "recommend_api": top_20_indices.tolist()}) + "\n")

        finally:
            if f_vec is not None:
                f_vec.close()

        print(f"\n[SAVE] Recommendation file written: {output_file}")

        # Auto-run metrics and print into stdout (so your nohup log contains results)
        self._run_metrics_single()

    # ========================= Sparse helper =========================
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data.astype(np.float32))
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def main():
    # Ensure run_tag is ALWAYS set (parallel-safe outputs)
    rt = _safe_tag(getattr(args, "run_tag", ""))
    if not rt:
        args.run_tag = _auto_run_tag()
        print(f"[RUN_TAG] No --run_tag provided. Auto run_tag set to: {args.run_tag}")

    config = {"n_users": data_generator.n_users, "n_items": data_generator.n_items}
    config_mashup = {"n_mashup": data_generator.n_mashup, "n_mashup_tag": data_generator.n_mashup_tag}
    config_api = {"n_api": data_generator.n_api, "n_api_tag": data_generator.n_api_tag}

    Engine = Model_Wrapper(
        data_config=config,
        config_mashup=config_mashup,
        config_api=config_api,
        pretrain_data=None,
    )

    if _to_int(getattr(args, "pretrain", 0), 0) == 1:
        print("pretrain path: ", Engine.weights_save_path)
        if os.path.exists(Engine.weights_save_path):
            Engine.load_model()
            Engine.save_recResult()
        else:
            print("Cannot load pretrained model. Start training from scratch")
    else:
        print("without pretraining")
        Engine.train()


if __name__ == "__main__":
    main()
