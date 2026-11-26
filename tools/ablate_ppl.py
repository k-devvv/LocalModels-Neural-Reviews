"""Ablation test for 'prunable' / 'fragile' / 'leave alone' claims: delta perplexity when one layer
is skipped (output := input) or one residual dim is mean-ablated. Gromov et al. arXiv:2403.17887 use the
same skip test.

usage: python tools/ablate_ppl.py <hf_id_or_dir> [--gguf f.gguf] --text eval.txt [--dims 0 397] -o results/ablate.json
"""
import argparse, json, math, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def layers_of(model):
    for path in ("model.layers", "model.language_model.layers", "transformer.h"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise SystemExit("could not find decoder layers")


def ppl(model, ids, chunk=512):
    nll, n = 0.0, 0
    for i in range(0, ids.shape[1] - 1, chunk):
        x = ids[:, i:i + chunk + 1]
        with torch.no_grad():
            out = model(x[:, :-1], labels=x[:, :-1])
        k = x.shape[1] - 2
        nll += out.loss.item() * k; n += k
    return math.exp(nll / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--gguf"); ap.add_argument("--text", required=True)
    ap.add_argument("--max-tokens", type=int, default=20000); ap.add_argument("--dims", type=int, nargs="*", default=[])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("-o", "--out", default="-")
    x = ap.parse_args()
    kw = {"gguf_file": x.gguf} if x.gguf else {}
    tok = AutoTokenizer.from_pretrained(x.model, **kw)
    model = AutoModelForCausalLM.from_pretrained(x.model, torch_dtype=torch.float16 if x.device == "cuda" else torch.float32, **kw).to(x.device).eval()
    ids = tok(open(x.text).read(), return_tensors="pt").input_ids[:, : x.max_tokens].to(x.device)
    base = ppl(model, ids)
    res = {"base_ppl": base, "skip_layer": {}, "mean_ablate_dim": {}}
    for i, layer in enumerate(layers_of(model)):
        h = layer.register_forward_hook(lambda m, inp, out: ((inp[0],) + tuple(out[1:])) if isinstance(out, tuple) else inp[0])
        res["skip_layer"][i] = ppl(model, ids) / base
        h.remove()
    for dim in x.dims:
        def hook(m, inp, out, dim=dim):
            hs = out[0] if isinstance(out, tuple) else out
            hs[..., dim] = hs[..., dim].mean()
            return out
        hs_ = [l.register_forward_hook(hook) for l in layers_of(model)]
        res["mean_ablate_dim"][dim] = ppl(model, ids) / base
        for h in hs_: h.remove()
    s = json.dumps(res, indent=1)
    (sys.stdout.write(s) if x.out == "-" else open(x.out, "w").write(s))


if __name__ == "__main__":
    main()
