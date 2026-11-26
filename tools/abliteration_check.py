"""Recover an abliteration (refusal-direction orthogonalisation) from the weights alone.

Arditi et al. (arXiv:2406.11717) remove a direction r from every matrix that writes to the residual
stream (token embedding, attention output, MLP down-projection): r^T W = 0. So r is a shared
near-null left direction of all writers. We accumulate G = sum_W W W^T (d x d) and take its smallest
eigenvectors. A clear gap (smallest eigenvalue far below the next) means an orthogonalised direction exists;
the per-matrix residual ||r^T W|| / ||W|| then shows which matrices were edited.

usage: python tools/abliteration_check.py model.gguf -o results/abliteration.json
"""
import argparse, json, sys
import numpy as np
from common import open_gguf, tensor_f32, split_name

WRITERS = ("attn_output", "ffn_down")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gguf"); ap.add_argument("-o", "--out", default="-")
    x = ap.parse_args()
    r, _ = open_gguf(x.gguf)
    mats = {}
    for t in r.tensors:
        _, idx, short = split_name(t.name)
        if short in WRITERS or t.name.startswith("token_embd"):
            W = tensor_f32(t).astype(np.float64)
            # writers: numpy rows = residual dims (d); token_embd: (vocab, d) -> transpose
            mats[t.name] = W.T if t.name.startswith("token_embd") else W
    d = next(iter(mats.values())).shape[0]
    G = np.zeros((d, d))
    for W in mats.values():
        Wn = W / (np.linalg.norm(W) or 1)          # equal weight per matrix
        G += Wn @ Wn.T
    ev, evec = np.linalg.eigh(G)                    # ascending
    rhat = evec[:, 0]
    per = {k: float(np.linalg.norm(rhat @ W) / (np.linalg.norm(W) / np.sqrt(d))) for k, W in mats.items()}
    out = {"d": d, "smallest_eigs": ev[:5].tolist(), "gap_ratio": float(ev[1] / max(ev[0], 1e-30)),
           "median_eig": float(np.median(ev)),
           "residual_along_rhat_vs_random": per,   # ~1 = untouched, ~0 = orthogonalised against rhat
           "rhat_top_dims": np.argsort(-np.abs(rhat))[:10].tolist()}
    s = json.dumps(out, indent=1)
    (sys.stdout.write(s) if x.out == "-" else open(x.out, "w").write(s))


if __name__ == "__main__":
    main()
