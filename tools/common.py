"""Shared helpers: open a GGUF, dequantize tensors, classify them."""
import re
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

# Arches whose llama.cpp converter stores RMSNorm weights as (1 + w) (Gemma 1-3 norm_shift, Qwen3-Next/3.5
# modify_tensors; Qwen's ssm_norm is not shifted). Gemma 4's norm_shift is 0, so it is not listed.
ONE_PLUS_W = {"gemma", "gemma2", "gemma3", "gemma3n", "qwen3next", "qwen35"}


def norm_shift(arch, short):
    return 1.0 if arch in ONE_PLUS_W and short != "ssm_norm" else 0.0
BLOCK_RE = re.compile(r"^(?P<stack>(?:[a-z]+\.)*?blk)\.(?P<idx>\d+)\.(?P<name>.+?)(?:\.weight|\.bias)?$")


def open_gguf(path):
    r = GGUFReader(path)
    arch = field_str(r, "general.architecture")
    return r, arch


def field_str(r, key):
    f = r.fields.get(key)
    if f is None:
        return None
    v = f.parts[f.data[0]]
    return bytes(v).decode("utf-8", "replace") if v.dtype == np.uint8 else v.tolist()


def tensor_f32(t):
    """Dequantize a ReaderTensor to float32 numpy, shape (rows=ne1.., cols=ne0)."""
    a = dequantize(t.data, t.tensor_type)
    return np.asarray(a, dtype=np.float32).reshape([int(x) for x in reversed(t.shape.tolist())])


def split_name(name):
    m = BLOCK_RE.match(name)
    if not m:
        return None, None, name
    return m["stack"], int(m["idx"]), m["name"]


def is_norm(short):
    return short.endswith("norm") or "_norm" in short or short.startswith("ln")


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
