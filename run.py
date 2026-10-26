# run.py

import os
import subprocess
import sys
from typing import List

from utility.parser import parse_args
from utility.parser_utils import print_args


def run_cmd(cmd: List[str], label: str, capture: bool = False) -> str:
    """
    Run a subprocess command.
    - capture=False: stream output live (best for nohup logs)
    - capture=True : capture stdout+stderr and return combined text
    """
    print(f"\n[RUN] {label}: {' '.join(cmd)}")

    if capture:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = ""
        if proc.stdout:
            out += proc.stdout
            print(proc.stdout)
        if proc.stderr:
            out += ("\n[STDERR]\n" + proc.stderr)
            print("[STDERR]")
            print(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed ({label}) with exit code {proc.returncode}")
        return out

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({label}) with exit code {proc.returncode}")
    return ""


def _filter_forward_args(argv: List[str]) -> List[str]:
    """
    Forward args exactly as user typed, but remove run.py-only controls and dataset.
    Handles both:
      --flag value
      --flag=value
    """
    run_only_flags = {"--fold", "--fold_start", "--fold_end", "--dataset"}

    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]

        # handle --flag=value
        if tok.startswith("--") and "=" in tok:
            k = tok.split("=", 1)[0]
            if k in run_only_flags:
                i += 1
                continue
            out.append(tok)
            i += 1
            continue

        # handle --flag value
        if tok in run_only_flags:
            i += 1
            if i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue

        out.append(tok)
        i += 1

    return out


if __name__ == "__main__":
    try:
        print("[CMD]", " ".join(sys.argv))

        args = parse_args()
        print_args(args, title="RUN.PY ARGUMENTS")

        py = sys.executable

        # Decide folds
        if getattr(args, "fold_start", None) is not None and getattr(args, "fold_end", None) is not None:
            fold_start = int(args.fold_start)
            fold_end = int(args.fold_end)
        else:
            fold_start = int(getattr(args, "fold", 1))
            fold_end = fold_start

        forwarded = _filter_forward_args(sys.argv[1:])

        print("\n" + "=" * 80)
        print(f"[INFO] Running folds {fold_start}..{fold_end} with method = {args.method}")
        print(f"[INFO] Forwarded args: {' '.join(forwarded) if forwarded else '(none)'}")
        print("=" * 80)

        # Run folds
        for i in range(fold_start, fold_end + 1):
            fold_name = f"fold_{i}"

            cmd = [py, "main.py", "--dataset", fold_name] + forwarded
            metric_cmd = [py, "metrics_single.py", "--dataset", fold_name] + forwarded

            run_cmd(cmd, label=f"TRAIN+RECOMMEND ({fold_name})", capture=False)
            run_cmd(metric_cmd, label=f"EVALUATE ({fold_name})", capture=False)

        # Aggregate results
        agg_cmd = [py, "metric.py"] + forwarded
        agg_out = run_cmd(agg_cmd, label="AGGREGATE (metric.py)", capture=True)

        # Write aggregation to results file
        run_tag = (getattr(args, "run_tag", "") or "").strip()
        os.makedirs("results", exist_ok=True)
        out_name = f"{run_tag}_10fold_avg.txt" if run_tag else f"{args.method}_10fold_avg.txt"
        out_path = os.path.join("results", out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(agg_out)

        print(f"\n[SAVE] Aggregated results written to: {os.path.abspath(out_path)}")

    except Exception as e:
        print(f"\n[ERROR] Experiment stopped: {e}")
        sys.exit(1)
