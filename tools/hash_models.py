"""Print sha256 + size for model files, for MODELS.md. usage: python tools/hash_models.py file1.gguf ..."""
import hashlib, os, sys
for p in sys.argv[1:]:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    print(f"| {os.path.basename(p)} | {os.path.getsize(p) / 1e9:.2f} GB | `{h.hexdigest()}` |")
