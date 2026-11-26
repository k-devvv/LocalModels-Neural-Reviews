"""LoRA adapter statistics: ||dW||, effective rank for every module, optional ||dW||/||W_base||.

dW = B @ A * (alpha / r). Effective rank = exp(entropy of normalised singular values) (Roy & Vetterli 2007).
usage: python tools/lora_stats.py adapter_dir [--base base_model.safetensors-or-dir] -o results/lora.json
"""
import argparse, glob, json, os, re, sys
import numpy as np
from safetensors.numpy import load_file


def eff_rank(m):
    s = np.linalg.svd(m, compute_uv=False)
    p = s / (s.sum() or 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir"); ap.add_argument("--base"); ap.add_argument("-o", "--out", default="-")
    x = ap.parse_args()
    cfg = json.load(open(os.path.join(x.adapter_dir, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    w = {k: v.astype(np.float32) for k, v in load_file(os.path.join(x.adapter_dir, "adapter_model.safetensors")).items()}
    base = {}
    if x.base:
        files = glob.glob(os.path.join(x.base, "*.safetensors")) if os.path.isdir(x.base) else [x.base]
        for f in files:
            base.update(load_file(f))
    out = {}
    for k in w:
        if "lora_A" not in k:
            continue
        kb = k.replace("lora_A", "lora_B")
        dW = (w[kb] @ w[k]) * scale
        m = re.search(r"layers\.(\d+)\.(?:\w+\.)?(\w+_proj)", k)
        key = f"L{m[1]}.{m[2]}" if m else k
        d = {"norm": float(np.linalg.norm(dW)), "eff_rank": eff_rank(dW), "rms": float(np.sqrt((dW ** 2).mean()))}
        bk = re.sub(r"^base_model\.model\.", "", k).replace(".lora_A.weight", ".weight").replace(".lora_A.default.weight", ".weight")
        if bk in base:
            d["rel_to_base"] = d["norm"] / float(np.linalg.norm(base[bk].astype(np.float32)))
        out[key] = d
    s = json.dumps({"scale": scale, "r": cfg["r"], "modules": out}, indent=1)
    (sys.stdout.write(s) if x.out == "-" else open(x.out, "w").write(s))


if __name__ == "__main__":
    main()
