# Llama-3.2-3B Abliterated: a look inside the layers

[← all reviews](../README.md)

*Source: the local Ollama blob for `Llama-3.2-3B Abliterated` (`Q8_0` GGUF, base `meta-llama/Llama-3.2-3B-Instruct`), read directly with the gguf library. Every number below is measured from the weights. No prompts were run and nothing was trained.*

---

## 1. Summary

This is Meta's Llama-3.2-3B-Instruct with **abliteration** applied: a refusal direction was found in the residual stream and projected out of the weights, so the model refuses less. The architecture is untouched: a plain 28-layer dense transformer with grouped-query attention and a SwiGLU MLP.

The whole file is **Q8_0**, so the numbers below are close to the original bf16 weights. The layer profile is the classic Llama shape: very quiet first layers, gains that climb steadily with depth, and a small drop in the last two layers.

> **What this review cannot show.** Abliteration changes the weights by a tiny rank-1 edit. It was not measured here. It can be, from this file alone: abliteration orthogonalises every matrix that writes to the residual stream against one direction (arXiv:2406.11717), so that direction shows up as a shared near-null direction of the embedding, `attn_output` and `ffn_down` weights. `tools/abliteration_check.py` finds it; diffing against the original Instruct weights with `tools/drift.py` would confirm it. Everything below describes the model as it is, not what was removed.

---

## 2. The model in one picture

```
 tokens ──► tokenizer (128,256 vocab)
              │
              ▼
 ┌──────────────────────────────────────────┐
 │ embedding 3072-d  (Q8_0, tied to head)   │
 └──────────────────────────────────────────┘
              │
              ▼  × 28 identical layers
 ┌──────────────────────────────────────────┐
 │ attn_norm → attention (24q / 8kv, d=128) │
 │ ffn_norm  → SwiGLU MLP 3072 → 8192 → 3072│  ~101M per block
 └──────────────────────────────────────────┘
              │
              ▼
 ┌──────────────────────────────────────────┐
 │ output_norm → head (same matrix as the   │
 │ embedding)                               │
 └──────────────────────────────────────────┘
```

| Fact | Value |
|---|---|
| Parameters | 3.21B |
| Layers | 28, all dense attention |
| Width / MLP | 3072 / 8192 |
| Heads | 24 query / 8 KV, head dim 128 |
| Context | 131,072 (RoPE base 500k, with Llama-3 rope_freqs scaling) |
| Vocab | 128,256 |
| Head | tied to the embedding |
| Quant | Q8_0 throughout (norms F32) |
| Default sampling | temperature 0.6, top_p 0.9 |
| License | Llama 3.2 Community License |

---

## 3. What the layers do

> **Caveat.** These are weight statistics, not behaviour tests. A norm gain says how loudly a layer reads from the residual stream; it does not say what the layer "knows". Read zones as hints, not proof.

| Zone | Layers | Attention gain | MLP gain | What it means |
|---|---|---|---|---|
| Entry | 0–1 | 0.09–0.14 | ~0.14 | Very small gains. Layer 0's q and k weights are the largest in the model (q 0.038, k 0.044), but its attn_output is the smallest (0.012). Typical "token identity and position" layers. |
| Early | 2–9 | 0.32 → ~0.38 | rising | Gains more than double from layer 1 to layer 2 (0.14 → 0.32), then hover around 0.30–0.38. |
| Middle | 10–20 | ~0.36–0.46 | ~0.27–0.37 | Smooth growth with no spikes. The bulk of the processing. |
| Late | 21–27 | peak 0.493 (layer 25) | peak 0.464 (layer 26) | The loudest layers. attn_output RMS reaches 0.020 at layers 26–27. Gains drop slightly in the last 1–2 layers (0.454 / 0.422 attention, 0.431 MLP). |

**Observations**

- **A slow ramp.** attn_norm goes 0.089 → 0.138 → 0.321 over the first three layers and then rises gently to its peak at layer 25. ffn_norm follows the same curve from 0.139 to 0.464.
- **Layer 0 attention is special.** Big q and k weights with a tiny output projection: it builds sharp attention patterns but writes little back. Not checked against other Llama models here.
- **The MLP weights are flat.** ffn_up stays at 0.017–0.018 (0.020 at layer 27) and ffn_down at about 0.018 everywhere. Depth changes come from the norms, not the matrices.
- **The output norm is strong.** output_norm averages 1.63 (max 2.36), much larger than any layer gain. It plausibly rescales the final state for the tied head (not compared with other models here).

### Every layer

Gains are the mean absolute value of each norm. Matrices are the RMS of the first 256 rows. Every matrix is Q8_0.

| L | type | M params | attn_n | q | k | v | o | ffn_n | f_gate | f_up | f_down |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | dense | 101 | 0.089 | 0.038 (Q8_0) | 0.044 (Q8_0) | 0.012 (Q8_0) | 0.012 (Q8_0) | 0.139 | 0.019 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 1 | dense | 101 | 0.138 | 0.029 (Q8_0) | 0.040 (Q8_0) | 0.013 (Q8_0) | 0.014 (Q8_0) | 0.171 | 0.019 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 2 | dense | 101 | 0.321 | 0.026 (Q8_0) | 0.033 (Q8_0) | 0.013 (Q8_0) | 0.013 (Q8_0) | 0.206 | 0.019 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 3 | dense | 101 | 0.336 | 0.025 (Q8_0) | 0.035 (Q8_0) | 0.014 (Q8_0) | 0.015 (Q8_0) | 0.224 | 0.020 (Q8_0) | 0.017 (Q8_0) | 0.017 (Q8_0) |
| 4 | dense | 101 | 0.297 | 0.026 (Q8_0) | 0.032 (Q8_0) | 0.016 (Q8_0) | 0.015 (Q8_0) | 0.240 | 0.021 (Q8_0) | 0.017 (Q8_0) | 0.017 (Q8_0) |
| 5 | dense | 101 | 0.341 | 0.028 (Q8_0) | 0.034 (Q8_0) | 0.012 (Q8_0) | 0.014 (Q8_0) | 0.254 | 0.020 (Q8_0) | 0.017 (Q8_0) | 0.017 (Q8_0) |
| 6 | dense | 101 | 0.340 | 0.029 (Q8_0) | 0.032 (Q8_0) | 0.014 (Q8_0) | 0.014 (Q8_0) | 0.259 | 0.020 (Q8_0) | 0.017 (Q8_0) | 0.017 (Q8_0) |
| 7 | dense | 101 | 0.343 | 0.021 (Q8_0) | 0.029 (Q8_0) | 0.012 (Q8_0) | 0.014 (Q8_0) | 0.259 | 0.020 (Q8_0) | 0.017 (Q8_0) | 0.017 (Q8_0) |
| 8 | dense | 101 | 0.365 | 0.025 (Q8_0) | 0.030 (Q8_0) | 0.012 (Q8_0) | 0.014 (Q8_0) | 0.263 | 0.020 (Q8_0) | 0.017 (Q8_0) | 0.017 (Q8_0) |
| 9 | dense | 101 | 0.376 | 0.024 (Q8_0) | 0.030 (Q8_0) | 0.016 (Q8_0) | 0.015 (Q8_0) | 0.262 | 0.020 (Q8_0) | 0.018 (Q8_0) | 0.017 (Q8_0) |
| 10 | dense | 101 | 0.383 | 0.026 (Q8_0) | 0.034 (Q8_0) | 0.011 (Q8_0) | 0.014 (Q8_0) | 0.267 | 0.019 (Q8_0) | 0.018 (Q8_0) | 0.017 (Q8_0) |
| 11 | dense | 101 | 0.364 | 0.023 (Q8_0) | 0.033 (Q8_0) | 0.014 (Q8_0) | 0.015 (Q8_0) | 0.271 | 0.019 (Q8_0) | 0.018 (Q8_0) | 0.017 (Q8_0) |
| 12 | dense | 101 | 0.417 | 0.022 (Q8_0) | 0.033 (Q8_0) | 0.012 (Q8_0) | 0.014 (Q8_0) | 0.280 | 0.019 (Q8_0) | 0.018 (Q8_0) | 0.017 (Q8_0) |
| 13 | dense | 101 | 0.415 | 0.024 (Q8_0) | 0.033 (Q8_0) | 0.014 (Q8_0) | 0.015 (Q8_0) | 0.297 | 0.019 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 14 | dense | 101 | 0.416 | 0.027 (Q8_0) | 0.030 (Q8_0) | 0.012 (Q8_0) | 0.015 (Q8_0) | 0.310 | 0.019 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 15 | dense | 101 | 0.428 | 0.025 (Q8_0) | 0.030 (Q8_0) | 0.014 (Q8_0) | 0.015 (Q8_0) | 0.322 | 0.020 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 16 | dense | 101 | 0.434 | 0.025 (Q8_0) | 0.031 (Q8_0) | 0.014 (Q8_0) | 0.015 (Q8_0) | 0.334 | 0.020 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 17 | dense | 101 | 0.436 | 0.025 (Q8_0) | 0.026 (Q8_0) | 0.014 (Q8_0) | 0.016 (Q8_0) | 0.343 | 0.021 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 18 | dense | 101 | 0.446 | 0.024 (Q8_0) | 0.029 (Q8_0) | 0.014 (Q8_0) | 0.015 (Q8_0) | 0.352 | 0.021 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 19 | dense | 101 | 0.440 | 0.025 (Q8_0) | 0.028 (Q8_0) | 0.013 (Q8_0) | 0.016 (Q8_0) | 0.368 | 0.021 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 20 | dense | 101 | 0.458 | 0.024 (Q8_0) | 0.028 (Q8_0) | 0.017 (Q8_0) | 0.016 (Q8_0) | 0.372 | 0.020 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 21 | dense | 101 | 0.458 | 0.024 (Q8_0) | 0.032 (Q8_0) | 0.020 (Q8_0) | 0.017 (Q8_0) | 0.382 | 0.021 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 22 | dense | 101 | 0.463 | 0.023 (Q8_0) | 0.024 (Q8_0) | 0.017 (Q8_0) | 0.017 (Q8_0) | 0.395 | 0.020 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 23 | dense | 101 | 0.446 | 0.021 (Q8_0) | 0.028 (Q8_0) | 0.018 (Q8_0) | 0.017 (Q8_0) | 0.411 | 0.021 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 24 | dense | 101 | 0.480 | 0.024 (Q8_0) | 0.026 (Q8_0) | 0.019 (Q8_0) | 0.018 (Q8_0) | 0.431 | 0.022 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 25 | dense | 101 | 0.493 | 0.023 (Q8_0) | 0.028 (Q8_0) | 0.021 (Q8_0) | 0.018 (Q8_0) | 0.453 | 0.021 (Q8_0) | 0.018 (Q8_0) | 0.018 (Q8_0) |
| 26 | dense | 101 | 0.454 | 0.021 (Q8_0) | 0.028 (Q8_0) | 0.024 (Q8_0) | 0.020 (Q8_0) | 0.464 | 0.021 (Q8_0) | 0.018 (Q8_0) | 0.017 (Q8_0) |
| 27 | dense | 101 | 0.422 | 0.021 (Q8_0) | 0.026 (Q8_0) | 0.021 (Q8_0) | 0.020 (Q8_0) | 0.431 | 0.022 (Q8_0) | 0.020 (Q8_0) | 0.017 (Q8_0) |

### Where the compression spent its bits

Nowhere special: every matrix is Q8_0 and the norms are F32. At 8 bits the quantization error is very small, so these statistics are a faithful view of the weights as published.

---

## 4. What's strong

- **Near-lossless quant.** Q8_0 everywhere keeps quality very close to the original model.
- **Simple and well understood.** A plain Llama block works with every tool: LoRA trainers, steering libraries, llama.cpp, MLX.
- **Small and fast.** About 3.4 GB at Q8_0, so it fits fully on a 6 GB GPU with room for context.
- **Fewer refusals.** That is the point of abliteration, and it is useful for creative writing or red-team testing.

## 5. Weak spots

- **Fewer guardrails.** Abliteration removes refusals, not only bad ones. Do not put it in front of untrusted users without your own filter.
- **Possible small quality loss.** Abliterated models sometimes lose a little reasoning or instruction-following. This review cannot measure that.
- **Older and smaller than Qwen3.5.** The 128K context, smaller vocabulary and 3B size put it behind the Qwen3.5-4B on most tasks.
- **Bigger file than its size suggests.** This 3.2B model at Q8_0 is about 3.4 GB, similar to or larger than a 4B Q4_K_M file (about 2.7–3.4 GB), in exchange for near-lossless precision.

## 6. Surgical map

| Goal | Best place | Why |
|---|---|---|
| Style / tone LoRA | attention + MLP, layers 18–27 | The highest gains: small changes show up strongly. |
| Restore some refusals | residual stream around layers 10–16 | Published abliteration work usually finds refusal directions in middle layers (not measured here); 10–16 is a starting guess. |
| Steering vectors | residual stream, layers 8–20 (sweep range) | Whether an added vector is "drowned out" depends on the residual-stream norm, not the norm gain; ActAdd tunes both layer and coefficient (arXiv:2308.10248). Measure \|\|h_l\|\| with `tools/act_probe.py`. |
| Domain knowledge LoRA | MLP, layers 6–22 | MLP layers are commonly believed to store most facts (interpretation, not measured). |
| Leave alone | layers 0–1, output_norm, embedding | The entry layers and the tied head/embedding matrix affect every token. |

---

## 7. Verdict

A clean, standard Llama-3.2-3B at near-full precision with the refusals removed. The layer profile is textbook and healthy. Use it when you need an uncensored small model that every tool understands; for general quality, Qwen3.5-4B is stronger.
