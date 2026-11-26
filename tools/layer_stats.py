"""Per-tensor weight statistics for one GGUF file -> JSON.

Fixes vs. the original reviews: full-matrix RMS (not first 256 rows), norm-gain distribution
(mean, median, top channels) instead of mean|g| alone, signed values, and per-head
Gated DeltaNet decay |A| * softplus(dt_bias).

usage: python tools/layer_stats.py model.gguf -o results/model.json
"""
import argparse, json, sys
import numpy as np
from common import open_gguf, tensor_f32, split_name, is_norm, softplus, norm_shift, field_str, ONE_PLUS_W


def norm_stats(v):
    a = np.abs(v)
    top = np.argsort(-a)[:5]
    med = float(np.median(a)) or 1e-12
    return {"mean_abs": float(a.mean()), "median_abs": float(np.median(a)), "mean": float(v.mean()),
            "max_abs": float(a.max()), "top_channels": [[int(i), float(v[i]), float(a[i] / med)] for i in top]}


def matrix_stats(w):
    rms = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
    out = {"rms": rms, "shape": list(w.shape)}
    if w.ndim == 2:
        row = np.sqrt(np.mean(w.astype(np.float64) ** 2, axis=1))
        med = float(np.median(row)) or 1e-12
        top = np.argsort(-row)[:5]
        out["row_rms_top"] = [[int(i), float(row[i] / med)] for i in top]   # rows = output dims
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("-o", "--out", default="-")
    a = ap.parse_args()
    r, arch = open_gguf(a.gguf)
    res = {"file": a.gguf, "arch": arch, "norms_stored_as_1_plus_w": arch in ONE_PLUS_W, "tensors": {}, "gdn_decay": {}}
    ssm = {}
    kl = field_str(r, f"{arch}.attention.key_length")
    heads = {"attn_k": kl, "attn_v": field_str(r, f"{arch}.attention.value_length") or kl, "attn_q": kl}
    for t in r.tensors:
        stack, idx, short = split_name(t.name)
        w = tensor_f32(t)
        entry = {"stack": stack, "block": idx, "name": short, "qtype": t.tensor_type.name, "n": int(w.size)}
        if w.ndim == 1 or (is_norm(short) and w.ndim <= 2):
            entry.update(norm_stats(w.ravel()))
            if is_norm(short) and norm_shift(arch, short):
                entry["effective_gain_mean_abs"] = float(np.abs(w.ravel() - 1.0).mean())  # the trained w
        else:
            entry.update(matrix_stats(w.reshape(w.shape[0], -1) if w.ndim > 2 else w))
            hd = heads.get(short)
            if isinstance(hd, (list, tuple, np.ndarray)):   # per-layer arrays (e.g. Qwen3.5 hybrid)
                hd = hd[idx] if idx is not None and idx < len(hd) else next((x for x in hd if x), None)
            hd = int(hd) if hd else None
            if hd and w.ndim == 2 and w.shape[0] % hd == 0:   # per-head RMS (rows grouped by head)
                entry["head_rms"] = np.sqrt((w.reshape(w.shape[0] // hd, -1).astype(np.float64) ** 2).mean(1)).round(6).tolist()
        res["tensors"][t.name] = entry
        if short in ("ssm_a", "ssm_dt") and idx is not None:
            ssm.setdefault(idx, {})[short] = w.ravel()
    for idx, d in sorted(ssm.items()):
        if "ssm_a" in d and "ssm_dt" in d and d["ssm_a"].size == d["ssm_dt"].size:
            A = np.abs(d["ssm_a"])            # llama.cpp stores ssm_a = -exp(A_log)
            dec = A * softplus(d["ssm_dt"])  # per-head baseline decay rate (input term excluded)
            res["gdn_decay"][idx] = {"absA_mean": float(A.mean()), "dt_bias_mean": float(d["ssm_dt"].mean()),
                                     "decay_per_head_mean": float(dec.mean()), "decay_per_head_median": float(np.median(dec)),
                                     "decay_min": float(dec.min()), "decay_max": float(dec.max()),
                                     "half_life_tokens_median": float(np.log(2) / max(np.median(dec), 1e-12))}
    s = json.dumps(res, indent=1)
    (sys.stdout.write(s) if a.out == "-" else open(a.out, "w").write(s))


if __name__ == "__main__":
    main()
