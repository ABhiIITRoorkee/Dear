# metrics_single.py
# ------------------------------------------------------------
# ROBUST (AUTO-FINDS YOUR main.py OUTPUT FILE, EVEN IF NAMING VARIES)
#
# It first tries the exact expected filename, then falls back to glob patterns that
# tolerate extra tokens (e.g., _pool3_w10_r30) and float formatting differences.
#
# Parallel-safe behavior:
# - Reads ONLY from output/<run_tag>/...
# - If multiple matches exist, selects the most recently modified file.
#
# Defensive behavior:
# - Empty-file guard (with short retry)
# - JSONL robust parsing (skip empty/malformed lines)
# - Clear diagnostics about which file is being evaluated
# ------------------------------------------------------------

import json
import math
import os
import glob
import time
from utility.parser import parse_args

args = parse_args()


def _safe_tag(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _fmt_float_short(v) -> str:
    """Match main.py-style compact float formatting (0.03, 0.1, 5, 1e-3)."""
    try:
        return f"{float(v):.6g}"
    except Exception:
        return str(v)


def _fmt_sim_1dec(v) -> str:
    """Your main.py commonly uses 1 decimal for sim (e.g., 5.0)."""
    try:
        return f"{float(v):.1f}"
    except Exception:
        return str(v)


def load_ground_truth_rs(rs_path: str):
    """Load fold ground-truth from RS.csv: mashup_id -> [api_ids]."""
    gt = {}
    with open(rs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            mashup_id = int(parts[0])
            apis = list(map(int, parts[1:])) if len(parts) > 1 else []
            gt[mashup_id] = apis
    return gt


def load_recommendations_jsonl(recom_path: str):
    """
    Load json-lines file: each line has {'mashup_id': ..., 'recommend_api': [...]}.

    Defensive behavior:
    - Skips empty lines
    - Skips malformed JSON lines with a warning instead of crashing
    - Skips lines missing mashup_id
    """
    rec = {}
    with open(recom_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Bad JSON at line {lineno} in {recom_path}. Skipping this line.")
                continue

            if "mashup_id" not in data:
                print(f"[WARN] Missing mashup_id at line {lineno} in {recom_path}. Skipping.")
                continue

            mashup_id = int(data["mashup_id"])
            recommend_api = data.get("recommend_api", [])
            rec[mashup_id] = recommend_api
    return rec


def _load_api_universe(fallback_rs_path: str):
    """
    For Coverage:
    - prefer Original Dataset/api.json
    - fallback: infer api IDs seen in RS.csv (prevents crash)
    """
    api_json_path = os.path.join("Original Dataset", "api.json")
    if os.path.exists(api_json_path):
        with open(api_json_path, "r", encoding="utf-8") as f:
            api_data = json.load(f)
        return {item["id"] for item in api_data if "id" in item}

    max_id = -1
    with open(fallback_rs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 1:
                continue
            ids = [int(x) for x in parts[1:]]
            if ids:
                max_id = max(max_id, max(ids))

    return set(range(max_id + 1)) if max_id >= 0 else set()


def _candidate_exact_paths(out_dir: str, run_tag: str, method: str, fold: str):
    """
    Build a couple of exact candidates because float formatting can differ
    across runs (e.g., sim=5 vs sim=5.0).
    """
    tag_part = f"_{run_tag}" if run_tag else ""

    sim_raw = _fmt_float_short(getattr(args, "similar_threthold", 5))
    sim_1d = _fmt_sim_1dec(getattr(args, "similar_threthold", 5))

    a1 = _fmt_float_short(getattr(args, "alpha1", 0.6))
    a2 = _fmt_float_short(getattr(args, "alpha2", 0.6))
    a3 = _fmt_float_short(getattr(args, "alpha3", 0.8))
    a4 = _fmt_float_short(getattr(args, "alpha4", 0.8))

    if method == "kd":
        kdL = _fmt_float_short(getattr(args, "kd_lambda", 0.0))
        tau = _fmt_float_short(getattr(args, "kd_tau", 0.1))
        topk = str(int(getattr(args, "kd_topk", 100)))
        every = str(int(getattr(args, "kd_every", 5)))

        p1 = os.path.join(
            out_dir,
            f"recommend_{method}{tag_part}_sim{sim_1d}_a{a1}_{a2}_{a3}_{a4}_"
            f"kdL{kdL}_tau{tau}_topk{topk}_every{every}_{fold}.json"
        )
        p2 = os.path.join(
            out_dir,
            f"recommend_{method}{tag_part}_sim{sim_raw}_a{a1}_{a2}_{a3}_{a4}_"
            f"kdL{kdL}_tau{tau}_topk{topk}_every{every}_{fold}.json"
        )
        return [p1, p2]

    # baseline naming (keep compatible)
    p1 = os.path.join(
        out_dir,
        f"recommend_{method}{tag_part}_sim{sim_1d}_a{a1}_{a2}_{a3}_{a4}_{fold}.json"
    )
    p2 = os.path.join(
        out_dir,
        f"recommend_{method}{tag_part}_sim{sim_raw}_a{a1}_{a2}_{a3}_{a4}_{fold}.json"
    )
    return [p1, p2]


def _candidate_glob_patterns(out_dir: str, run_tag: str, method: str, fold: str):
    """
    Glob patterns that tolerate:
    - extra tokens between every{...} and _{fold}.json (e.g., _pool3_w10_r30)
    - float formatting differences
    """
    tag_part = f"_{run_tag}" if run_tag else ""

    sim_1d = _fmt_sim_1dec(getattr(args, "similar_threthold", 5))
    sim_raw = _fmt_float_short(getattr(args, "similar_threthold", 5))

    a1 = _fmt_float_short(getattr(args, "alpha1", 0.6))
    a2 = _fmt_float_short(getattr(args, "alpha2", 0.6))
    a3 = _fmt_float_short(getattr(args, "alpha3", 0.8))
    a4 = _fmt_float_short(getattr(args, "alpha4", 0.8))

    patterns = []

    if method == "kd":
        kdL = _fmt_float_short(getattr(args, "kd_lambda", 0.0))
        tau = _fmt_float_short(getattr(args, "kd_tau", 0.1))
        topk = str(int(getattr(args, "kd_topk", 100)))
        every = str(int(getattr(args, "kd_every", 5)))

        # Allow anything (*) before _{fold}.json
        patterns.append(
            os.path.join(
                out_dir,
                f"recommend_{method}{tag_part}_sim{sim_1d}_a{a1}_{a2}_{a3}_{a4}_"
                f"kdL{kdL}_tau{tau}_topk{topk}_every{every}*_{fold}.json"
            )
        )
        patterns.append(
            os.path.join(
                out_dir,
                f"recommend_{method}{tag_part}_sim{sim_raw}_a{a1}_{a2}_{a3}_{a4}_"
                f"kdL{kdL}_tau{tau}_topk{topk}_every{every}*_{fold}.json"
            )
        )
        return patterns

    patterns.append(
        os.path.join(
            out_dir,
            f"recommend_{method}{tag_part}_sim{sim_1d}_a{a1}_{a2}_{a3}_{a4}*_{fold}.json"
        )
    )
    patterns.append(
        os.path.join(
            out_dir,
            f"recommend_{method}{tag_part}_sim{sim_raw}_a{a1}_{a2}_{a3}_{a4}*_{fold}.json"
        )
    )
    return patterns


def find_recom_file():
    method = getattr(args, "method", "baseline")
    fold = args.dataset  # e.g., fold_1
    run_tag = _safe_tag(getattr(args, "run_tag", ""))

    out_dir = os.path.join("output", run_tag) if run_tag else "output"
    if not os.path.exists(out_dir):
        return None, out_dir, run_tag, method, fold

    # 1) exact candidates first
    for p in _candidate_exact_paths(out_dir, run_tag, method, fold):
        if os.path.exists(p):
            return p, out_dir, run_tag, method, fold

    # 2) glob patterns (robust)
    matches = []
    for pat in _candidate_glob_patterns(out_dir, run_tag, method, fold):
        matches.extend(glob.glob(pat))

    # If still nothing, last-resort: any recommend file for this fold in this run_dir
    if not matches:
        matches = glob.glob(os.path.join(out_dir, f"recommend_{method}*_{fold}.json"))

    if not matches:
        return None, out_dir, run_tag, method, fold

    # pick the newest file (prevents overlap confusion)
    matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return matches[0], out_dir, run_tag, method, fold


def wait_for_nonempty(path: str, retries: int = 10, sleep_s: float = 0.25):
    """
    Small guard against race conditions (e.g., file created but not flushed yet).
    This is short and only runs when needed.
    """
    for _ in range(max(1, retries)):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return True
        time.sleep(sleep_s)
    return os.path.exists(path) and os.path.getsize(path) > 0


if __name__ == "__main__":
    Ns = [3, 5, 10, 20]

    fold = args.dataset
    rs_path = f"dataset/{fold}/RS.csv"
    if not os.path.exists(rs_path):
        raise FileNotFoundError(f"[metrics_single.py] RS.csv not found: {rs_path}")

    ground_truth = load_ground_truth_rs(rs_path)
    all_api_ids = _load_api_universe(rs_path)

    recom_path, out_dir, run_tag, method, fold = find_recom_file()

    if recom_path is None:
        raise FileNotFoundError(
            f"[metrics_single.py] Recommendation file not found in:\n  {out_dir}\n\n"
            f"Diagnostics:\n"
            f"  out_dir  : {out_dir}\n"
            f"  run_tag  : {run_tag if run_tag else '(empty)'}\n"
            f"  method   : {method}\n"
            f"  fold     : {fold}\n"
            f"  similar  : {getattr(args, 'similar_threthold', None)}\n"
            f"  alphas   : {getattr(args, 'alpha1', None)},{getattr(args, 'alpha2', None)},{getattr(args, 'alpha3', None)},{getattr(args, 'alpha4', None)}\n"
            f"  kd       : lambda={getattr(args, 'kd_lambda', None)}, tau={getattr(args, 'kd_tau', None)}, topk={getattr(args, 'kd_topk', None)}, every={getattr(args, 'kd_every', None)}\n\n"
            f"Hint: check output/<run_tag>/ for the actual saved recommendation filename."
        )

    if not wait_for_nonempty(recom_path, retries=10, sleep_s=0.25):
        raise RuntimeError(
            f"[metrics_single.py] Recommendation file exists but is EMPTY:\n  {recom_path}\n\n"
            f"Likely cause: metrics_single.py started before main.py finished writing/flushing the file."
        )

    recommendations = load_recommendations_jsonl(recom_path)

    missing = [mid for mid in ground_truth.keys() if mid not in recommendations]
    if missing:
        print(f"[WARN] Missing recommendations for {len(missing)} mashups in {fold} (showing first 5): {missing[:5]}")

    print(f"[INFO] Using recommendations from: {recom_path}")
    print(f"The performance of {fold} (method={method}, run_tag={run_tag if run_tag else 'none'}) are as follows:")

    for N in Ns:
        precision_list = []
        recall_list = []
        map_list = []
        ndcg_list = []
        recommended_api_set = set()

        for mashup_id, true_apis in ground_truth.items():
            recommended_apis = recommendations.get(mashup_id, [])[:N]
            if not true_apis:
                continue

            true_set = set(true_apis)
            hit_count = len(set(recommended_apis) & true_set)

            recommended_api_set.update(recommended_apis)

            precision_list.append(hit_count / N)
            recall_list.append(hit_count / len(true_apis))

            # MAP@N
            cor_list = [1.0 if recommended_apis[i] in true_set else 0.0 for i in range(len(recommended_apis))]
            sum_cor_list = sum(cor_list)
            if sum_cor_list == 0:
                map_score = 0.0
            else:
                summary = 0.0
                for i in range(len(cor_list)):
                    if cor_list[i] == 1.0:
                        summary += (sum(cor_list[:i + 1]) / (i + 1))
                map_score = summary / sum_cor_list
            map_list.append(map_score)

            # NDCG@N
            dcg = 0.0
            for i in range(N):
                if i < len(recommended_apis) and recommended_apis[i] in true_set:
                    dcg += 1 / math.log2(i + 2)

            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(true_apis), N)))
            ndcg = (dcg / idcg) if idcg != 0 else 0.0
            ndcg_list.append(ndcg)

        average_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0
        average_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
        average_map = sum(map_list) / len(map_list) if map_list else 0.0
        average_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0

        valid_recommended = recommended_api_set.intersection(all_api_ids) if all_api_ids else set(recommended_api_set)
        coverage = (len(valid_recommended) / len(all_api_ids)) if all_api_ids else 0.0

        print(
            f"N={N} -> Precision: {average_precision:.4f}, Recall: {average_recall:.4f}, "
            f"MAP: {average_map:.4f}, NDCG: {average_ndcg:.4f}, Cov: {coverage:.4f}"
        )
