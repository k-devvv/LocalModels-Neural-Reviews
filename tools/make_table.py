"""Render a layer_stats JSON as a markdown per-layer table (norm gains + full-matrix RMS).

usage: python tools/make_table.py results/model.json [--stack blk] > table.md
"""
import argparse, json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("--stack", default="blk")
    x = ap.parse_args()
    d = json.load(open(x.json))
    rows, cols = defaultdict(dict), []
    for e in d["tensors"].values():
        if e["stack"] != x.stack:
            continue
        # norms: mean|g|; other vectors (ssm_a, dt bias): signed mean; matrices: full RMS
        v = e["rms"] if "rms" in e else (e["mean_abs"] if e["name"].endswith("norm") or "_norm" in e["name"] else e["mean"])
        rows[e["block"]][e["name"]] = v
        if e["name"] not in cols:
            cols.append(e["name"])
    print("| L | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for b in sorted(rows):
        print(f"| {b} | " + " | ".join(f"{rows[b][c]:.4g}" if c in rows[b] else "·" for c in cols) + " |")
    if d.get("gdn_decay"):
        print("\n| L | mean \\|A\\| | mean dt_bias | decay per head (mean) | min | max |\n|---|---|---|---|---|---|")
        for b, g in sorted(d["gdn_decay"].items(), key=lambda kv: int(kv[0])):
            print(f"| {b} | {g['absA_mean']:.3g} | {g['dt_bias_mean']:+.2f} | {g['decay_per_head_mean']:.3g} | {g['decay_min']:.3g} | {g['decay_max']:.3g} |")


if __name__ == "__main__":
    main()
