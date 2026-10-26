# metric.py

import os
import json
import math
from datetime import datetime
from utility.parser import parse_args

args = parse_args()


def safe_jsonl_load(path: str):
    """Load JSONL safely (skip empty/malformed lines)."""
    rec = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Bad JSON in {path} at line {lineno}. Skipping.")
                continue
            if "mashup_id" not in data:
                print(f"[WARN] Missing mashup_id in {path} at line {lineno}. Skipping.")
                continue
            rec[int(data["mashup_id"])] = data.get("recommend_api", [])
    return rec


def _proj_root() -> str:
    p = str(getattr(args, "proj_path", "") or ".")
    return p


def _output_root() -> str:
    outp = str(getattr(args, "output_path", "") or "output")
    return os.path.join(_proj_root(), outp)


def _fmt_float_short(v) -> str:
    try:
        return f"{float(v):.6g}"
    except Exception:
        return str(v)


def _sim_str() -> str:
    # main.py writes sim with one decimal, e.g., "5.0"
    try:
        return f"{float(getattr(args, 'similar_threthold', 5.0)):.1f}"
    except Exception:
        return str(getattr(args, "similar_threthold", 5.0))


def build_rec_path(fold_name: str):
    """
    MUST match main.py (current):

    out_dir = <proj_path>/<output_path>/<run_tag>/   (if run_tag set)
    filename = recommend_{method}_{run_tag}_sim5.0_a..._(kdL.._tau.._topk.._every.._pool.._w.._r..)_<fold>.json
    """
    method = getattr(args, "method", "baseline")
    run_tag = (getattr(args, "run_tag", "") or "").strip()

    out_root = _output_root()
    out_dir = os.path.join(out_root, run_tag) if run_tag else out_root

    base = f"recommend_{method}"
    if run_tag:
        base += f"_{run_tag}"

    base += f"_sim{_sim_str()}_a{args.alpha1}_{args.alpha2}_{args.alpha3}_{args.alpha4}"

    if str(method) == "kd":
        base += (
            f"_kdL{_fmt_float_short(getattr(args,'kd_lambda',0.0))}"
            f"_tau{_fmt_float_short(getattr(args,'kd_tau',0.0))}"
            f"_topk{int(getattr(args,'kd_topk',0) or 0)}"
            f"_every{int(getattr(args,'kd_every',0) or 0)}"
            f"_pool{int(getattr(args,'kd_pool_mult',3) or 3)}"
            f"_w{int(getattr(args,'kd_warmup_epochs',0) or 0)}"
            f"_r{int(getattr(args,'kd_ramp_epochs',0) or 0)}"
        )

    filename = f"{base}_{fold_name}.json"
    return os.path.join(out_dir, filename), out_dir, run_tag


def _load_api_universe():
    """
    Prefer api.json if available. If missing, fallback to range(max_id+1) using RS.csv.
    """
    api_json_path = os.path.join(_proj_root(), "Original Dataset", "api.json")
    if os.path.exists(api_json_path):
        with open(api_json_path, "r", encoding="utf-8") as f:
            api_data = json.load(f)
        return {item["id"] for item in api_data}

    # fallback: infer from test files
    max_id = -1
    data_root = os.path.join(_proj_root(), str(getattr(args, "data_path", "dataset") or "dataset"))
    for fold_idx in range(1, 11):
        rs_path = os.path.join(data_root, f"fold_{fold_idx}", "RS.csv")
        if not os.path.exists(rs_path):
            continue
        with open(rs_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) <= 1:
                    continue
                ids = [int(x) for x in parts[1:]]
                if ids:
                    max_id = max(max_id, max(ids))
    if max_id < 0:
        return set()
    return set(range(max_id + 1))


def print_run_header():
    method = getattr(args, "method", "baseline")
    run_tag = (getattr(args, "run_tag", "") or "").strip()

    print("================================================================================")
    print("[RUN SUMMARY]")
    print("================================================================================")
    print(f"timestamp              : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"method                 : {method}")
    print(f"run_tag                : {run_tag if run_tag else '(none)'}")
    print(f"dataset                : 10-fold (fold_1..fold_10)")
    print("--------------------------------------------------------------------------------")
    print(f"similar_threthold      : {args.similar_threthold}")
    print(f"alpha1 alpha2 alpha3 alpha4 : {args.alpha1} {args.alpha2} {args.alpha3} {args.alpha4}")
    print("--------------------------------------------------------------------------------")
    print(f"use_kd_router          : {getattr(args,'use_kd_router',0)}")
    print(f"kd_lambda              : {getattr(args,'kd_lambda',0.0)}")
    print(f"kd_tau                 : {getattr(args,'kd_tau',0.0)}")
    print(f"kd_topk                : {getattr(args,'kd_topk',0)}")
    print(f"kd_every               : {getattr(args,'kd_every',0)}")
    print(f"router_hidden          : {getattr(args,'router_hidden',256)}")
    print("--------------------------------------------------------------------------------")
    print(f"kd_pool_mult           : {getattr(args,'kd_pool_mult',3)}")
    print(f"kd_warmup_epochs       : {getattr(args,'kd_warmup_epochs',0)}")
    print(f"kd_ramp_epochs         : {getattr(args,'kd_ramp_epochs',0)}")
    print(f"kd_teacher_mix         : {getattr(args,'kd_teacher_mix',1.0)}")
    print("--------------------------------------------------------------------------------")
    print(f"use_geom_kd            : {getattr(args,'use_geom_kd',0)}")
    print(f"geom_lambda            : {getattr(args,'geom_lambda',0.0)}")
    print(f"geom_batch             : {getattr(args,'geom_batch',0)}")
    print("================================================================================")
    print()


if __name__ == "__main__":
    print_run_header()

    Ns = [3, 5, 10, 20]
    method = getattr(args, "method", "baseline")
    run_tag = (getattr(args, "run_tag", "") or "").strip()

    all_api_ids = _load_api_universe()

    fold_results = {N: {"precision": [], "recall": [], "map": [], "ndcg": []} for N in Ns}
    total_recommended_api_set = {N: set() for N in Ns}
    used_folds = 0

    data_root = os.path.join(_proj_root(), str(getattr(args, "data_path", "dataset") or "dataset"))

    for fold_idx in range(1, 11):
        fold_name = f"fold_{fold_idx}"
        rs_path = os.path.join(data_root, fold_name, "RS.csv")
        rec_path, _, _ = build_rec_path(fold_name)

        if not os.path.exists(rs_path):
            print(f"[SKIP] {fold_name}: RS.csv not found: {rs_path}")
            continue

        if not os.path.exists(rec_path):
            print(f"[SKIP] {fold_name}: recommendation file not found: {rec_path}")
            continue

        if os.path.getsize(rec_path) == 0:
            print(f"[SKIP] {fold_name}: recommendation file is EMPTY: {rec_path}")
            continue

        used_folds += 1

        # ground truth
        ground_truth = {}
        with open(rs_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                mashup_id = int(parts[0])
                apis = list(map(int, parts[1:])) if len(parts) > 1 else []
                ground_truth[mashup_id] = apis

        # recommendations
        recommendations = safe_jsonl_load(rec_path)

        for N in Ns:
            precision_list, recall_list, map_list, ndcg_list = [], [], [], []
            recommended_api_set = set()

            for mashup_id, true_apis in ground_truth.items():
                if not true_apis:
                    continue

                recommended_apis = recommendations.get(mashup_id, [])[:N]
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

            fold_results[N]["precision"].append(sum(precision_list) / len(precision_list) if precision_list else 0.0)
            fold_results[N]["recall"].append(sum(recall_list) / len(recall_list) if recall_list else 0.0)
            fold_results[N]["map"].append(sum(map_list) / len(map_list) if map_list else 0.0)
            fold_results[N]["ndcg"].append(sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0)

            total_recommended_api_set[N].update(recommended_api_set)

    print("==== Overall average across all folds ====")
    print(f"[INFO] method={method}, folds_used={used_folds}/10, run_tag={run_tag if run_tag else 'none'}")

    for N in Ns:
        avg_prec = sum(fold_results[N]["precision"]) / len(fold_results[N]["precision"]) if fold_results[N]["precision"] else 0.0
        avg_rec  = sum(fold_results[N]["recall"]) / len(fold_results[N]["recall"]) if fold_results[N]["recall"] else 0.0
        avg_map  = sum(fold_results[N]["map"]) / len(fold_results[N]["map"]) if fold_results[N]["map"] else 0.0
        avg_ndcg = sum(fold_results[N]["ndcg"]) / len(fold_results[N]["ndcg"]) if fold_results[N]["ndcg"] else 0.0

        valid_recommended = total_recommended_api_set[N].intersection(all_api_ids) if all_api_ids else set(total_recommended_api_set[N])
        final_cov = (len(valid_recommended) / len(all_api_ids)) if all_api_ids else 0.0

        print(
            f"N={N} -> Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}, "
            f"MAP: {avg_map:.4f}, NDCG: {avg_ndcg:.4f}, Cov: {final_cov:.4f}"
        )
