# Analysis tools

Scripts that produce (and will regenerate) the numbers in the reviews. Every script writes JSON to `results/`.

```bash
pip install -r tools/requirements.txt
python tools/hash_models.py path/to/*.gguf                    # sha256 + size rows for MODELS.md
python tools/layer_stats.py model.gguf -o results/model.json  # weight stats, all tensors
python tools/make_table.py results/model.json > table.md      # per-layer markdown table
python tools/drift.py base.gguf tuned.gguf -o results/drift_base_tuned.json
python tools/abliteration_check.py abliterated.gguf -o results/abliteration.json
python tools/lora_stats.py adapter_dir --base base_dir -o results/lora.json
python tools/act_probe.py <hf_id_or_dir> --gguf model.gguf -o results/act_model.json     # needs torch
python tools/ablate_ppl.py <hf_id_or_dir> --gguf model.gguf --text eval.txt --dims 0 -o results/ablate_model.json
```

| Script | Measures | Fixes this gap in the original method |
|---|---|---|
| `layer_stats.py` | Full-matrix RMS, top output rows; norm gain mean, median and top channels (signed); per-head Gated DeltaNet decay \|A\|·softplus(dt_bias) | "First 256 rows" covered only 2 of 8 KV heads in some matrices; mean\|gain\| hid outlier channels; dt bias was ignored |
| `drift.py` | Per-tensor ‖B−A‖/‖A‖ and cosine on dequantized weights | Old drift compared summary stats (norm gains, RMS), which LoRA-merged fine-tunes barely change |
| `abliteration_check.py` | Shared near-null direction of all residual writers (embedding, attn_output, ffn_down) | Llama review said the abliteration edit could not be measured without the base model |
| `lora_stats.py` | ‖ΔW‖, effective rank for every module, ‖ΔW‖/‖W_base‖ | Old totals favoured large MLP matrices; rank was only checked on `down_proj` |
| `act_probe.py` | Residual norm, write ratio ‖h₍l+1₎−h_l‖/‖h_l‖, angular distance, attention entropy, top residual dims | "Loudness" was read from pre-norm gains, which post-norms and QK-norms largely cancel |
| `ablate_ppl.py` | ΔPPL when one layer is skipped or one residual dim is mean-ablated | "Prunable", "fragile" and "leave alone" claims were untested |

Conventions to know when reading GGUF values (llama.cpp converter):
- Qwen3.5 / Qwen3-Next (except `ssm_norm`) and Gemma 1–3 norms are stored as 1 + w, so a "gain near 0 = off" reading does not apply to them; safetensors files store w. Gemma 4 is stored unshifted. `layer_stats.py` reports `effective_gain_mean_abs` (the trained w) for shifted norms.
- `ssm_a` is stored as −exp(A_log); `ssm_dt.bias` is the signed dt bias.
- GGUF matrices dequantize to (rows = output features, cols = input features).

Measure the quantization noise floor before calling a drift "training": run `drift.py` between a model's own Q8_0 and Q4_K_M files.

The scripts were tested on synthetic GGUF files built with `gguf.GGUFWriter`. They have not yet been run on the review models; see the open items in the PR.
