"""Weight-level difference between two GGUFs with the same skeleton.

Reports, per tensor present in both: relative Frobenius difference ||B-A||/||A|| and cosine,
plus per-norm ||g_b - g_a|| / ||g_a|| on the effective gain (the old 'drift' divided per element,
which blows up for tiny gains). Use --noise to estimate the
quantization noise floor: pass a second copy of A requantized to B's quant type.

usage: python tools/drift.py base.gguf tuned.gguf -o results/drift_base_tuned.json
"""
import argparse, json, sys
import numpy as np
from common import open_gguf, tensor_f32, split_name, is_norm, norm_shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("-o", "--out", default="-")
    x = ap.parse_args()
    ra, arch = open_gguf(x.a); rb, _ = open_gguf(x.b)
    tb = {t.name: t for t in rb.tensors}
    rows, blocks = {}, {}
    for t in ra.tensors:
        if t.name not in tb:
            continue
        A = tensor_f32(t).astype(np.float64); B = tensor_f32(tb[t.name]).astype(np.float64)
        if A.shape != B.shape:
            continue
        stack, idx, short = split_name(t.name)
        na = np.linalg.norm(A) or 1e-12
        d = {"rel_fro": float(np.linalg.norm(B - A) / na),
             "cos": float((A * B).sum() / (na * (np.linalg.norm(B) or 1e-12))),
             "qtypes": [t.tensor_type.name, tb[t.name].tensor_type.name]}
        if A.ndim == 1 and is_norm(short):
            sh = norm_shift(arch, short)     # compare the trained gain, not the stored 1 + w
            ga, gb = A - sh, B - sh
            d["norm_gain_rel_change"] = float(np.linalg.norm(gb - ga) / (np.linalg.norm(ga) or 1e-12))
        rows[t.name] = d
        if idx is not None:
            blocks.setdefault(f"{stack}.{idx}", []).append(d["rel_fro"])
    summary = {k: float(np.mean(v)) for k, v in blocks.items()}
    out = {"a": x.a, "b": x.b, "mean_rel_fro": float(np.mean([r["rel_fro"] for r in rows.values()])) if rows else None,
           "per_block_mean_rel_fro": summary, "tensors": rows}
    s = json.dumps(out, indent=1)
    (sys.stdout.write(s) if x.out == "-" else open(x.out, "w").write(s))


if __name__ == "__main__":
    main()
