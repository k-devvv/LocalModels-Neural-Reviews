"""Activation probe: what the weight-only reviews could not measure.

Per layer, averaged over prompts and token positions:
  resid_norm   ||h_l||                      residual-stream size entering layer l
  write_ratio  ||h_{l+1} - h_l|| / ||h_l||   how much layer l actually changes the stream ("loudness")
  angular_dist arccos(cos(h_l, h_{l+1}))/pi  layer similarity used for pruning in arXiv:2403.17887
  attn_entropy mean entropy of attention rows, / log(n) (1.0 = uniform), where the model exposes attentions
  top_dims     residual dims with the largest mean |h| (massive activations, Sun et al. arXiv:2402.17762)
Uses Hugging Face transformers. GGUF files load via gguf_file= for supported architectures.

usage: python tools/act_probe.py <hf_id_or_dir> [--gguf file.gguf] [--prompts prompts.txt] -o results/act_model.json
"""
import argparse, json, math, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "The capital of France is", "def fibonacci(n):\n    ", "Translate to German: The weather is nice today.",
    "Q: What is 17 * 23?\nA:", "Once upon a time, in a small village,", "import numpy as np\nx = np.zeros((3, 3))\n",
    "The mitochondria is the powerhouse of the", "Write a haiku about autumn leaves.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--gguf"); ap.add_argument("--prompts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-attn", action="store_true", help="skip attention entropy (saves memory)")
    ap.add_argument("-o", "--out", default="-")
    x = ap.parse_args()
    prompts = open(x.prompts).read().split("\n\n") if x.prompts else DEFAULT_PROMPTS
    kw = {"gguf_file": x.gguf} if x.gguf else {}
    tok = AutoTokenizer.from_pretrained(x.model, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        x.model, torch_dtype=torch.float16 if x.device == "cuda" else torch.float32,
        attn_implementation="eager", **kw).to(x.device).eval()
    acc = None
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(x.device)
        with torch.no_grad():
            o = model(**ids, output_hidden_states=True, output_attentions=not x.no_attn)
        hs = [h[0].float() for h in o.hidden_states]           # L+1 tensors, (seq, d)
        n = [h.norm(dim=-1) for h in hs]
        rec = {"resid_norm": [v.mean().item() for v in n[:-1]],
               "write_ratio": [((hs[i + 1] - hs[i]).norm(dim=-1) / (n[i] + 1e-6)).mean().item() for i in range(len(hs) - 1)],
               # angular distance between consecutive layers, as in Gromov et al. arXiv:2403.17887 (low = layer barely rotates the stream)
               "angular_dist": [(torch.arccos(torch.nn.functional.cosine_similarity(hs[i], hs[i + 1], dim=-1).clamp(-1, 1)) / math.pi).mean().item() for i in range(len(hs) - 1)]}
        if not x.no_attn and o.attentions:
            ent = []
            for a in o.attentions:
                if a is None:
                    ent.append(None); continue
                a = a[0].float().clamp_min(1e-12)                # (heads, q, k)
                L = a.shape[-1]
                rows = -(a * a.log()).sum(-1)                   # (heads, q)
                norm = torch.log(torch.arange(1, L + 1, device=a.device).float()).clamp_min(1e-6)
                ent.append((rows[:, 1:] / norm[1:]).mean().item() if L > 1 else None)
            rec["attn_entropy"] = ent
        top = hs[len(hs) // 2].abs().mean(0)
        rec["top_dims_mid"] = torch.topk(top, 5).indices.tolist()
        if acc is None:
            acc = {k: ([0.0 if not isinstance(v, int) else v for v in vs] if k != "top_dims_mid" else []) for k, vs in rec.items()}
            acc["_n"] = 0
        for k, vs in rec.items():
            if k == "top_dims_mid":
                acc[k].append(vs)
            else:
                acc[k] = [(a or 0) + (b or 0) for a, b in zip(acc[k], vs)]
        acc["_n"] += 1
    n = acc.pop("_n")
    res = {k: (v if k == "top_dims_mid" else [round(t / n, 5) for t in v]) for k, v in acc.items()}
    res.update({"model": x.model, "gguf": x.gguf, "n_prompts": n})
    s = json.dumps(res, indent=1)
    (sys.stdout.write(s) if x.out == "-" else open(x.out, "w").write(s))


if __name__ == "__main__":
    main()
