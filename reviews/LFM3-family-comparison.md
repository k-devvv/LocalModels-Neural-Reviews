# LFM3 family: how the six models relate

![skeleton](../assets/skeleton-lfm3-family.svg)

[← all reviews](../README.md)

Six local models share **one 24-block LLM skeleton** (13 short-conv + 11 gated-attention blocks, width 2,048, 1.73B params). They differ in training and in the "senses" bolted on.

## Lineage (LLM drift = mean relative change of norm gains; 0 = identical)

This table averages over all norm tensors; the per-model pages average per block first, so their averages differ slightly (e.g. VL 0.265 here vs 0.261 there).

| | LFM3 | d1 | d1-Omni | Audio | Omni | VL |
|---|---|---|---|---|---|---|
| **LFM3-1.7B** | – | 0.186 | 0.087 | 0.138 | 0.138 | 0.265 |
| **d1-2B** | | – | 0.252 | | | 0.101 |
| **d1-Omni-2B** | | | – | 0.062 | 0.061 | |
| **LFM3-Audio-2B** | | | | – | **0.005** | 0.300 |

```
 LFM3-1.7B ── d1-Omni-2B ── LFM3-Audio-2B ≈ LFM3-Omni-2B   (one cluster)
     ╲
      ╲…… d1-2B ──── LFM3-VL-2B                            (other cluster)
```

Drift is symmetric, so these lines show similarity, not which model was trained from which. **Audio and Omni have practically the same LLM** (0.005). d1-Omni sits between the base and them by similarity. d1 and VL form a separate pair.

## Shared parts

| Part | Used by | Notes |
|---|---|---|
| Vision encoder A (small weights, adapter norm 0.45) | VL, d1 | near-identical (median 0.4% relative difference; 28/133 tensors byte-identical) |
| Vision encoder B (larger weights, adapter norm 1.0) | Omni, d1-Omni | close (~6% relative; Q4_K vs Q8_0, so mostly quantization noise) |
| Audio encoder (17 blocks) | Audio, Omni, d1-Omni | same design, different weights (~9–12% relative; 0/170 byte-identical Audio vs Omni) |
| Audio generator (code 3 + wave 8 blocks) | Audio, Omni | mostly identical (97/143 tensors byte-identical) |

## Common traits

| Trait | lfm3 | d1 | d1omni | audio | omni | vl |
|---|---|---|---|---|---|---|
| L18 attention q-norm (≈1 is normal) | 0.087 | 0.069 | 0.101 | 0.113 | 0.113 | 0.061 |
| Output norm | 1.928 | 2.389 | 2.71 | 2.912 | 2.904 | 2.587 |
| Channel 397 tops post-attention norm (layers) | 15/24 | 14 | 18 | 18 | 18 | 15 |
| Causal | yes | **no** | **no** | yes | yes | yes |
| Quant | Q4_K_M | Q4_K_M | **Q8_0** | Q4_K_M | Q4_K_M | Q4_K_M |
| Projector | – | 189M | 305M | 285M | 475M | 189M |

- **L18 attention has tiny q/k gains in every model.** It was already small in the base and no fine-tune raised it. By weights it is likely the best candidate for pruning (untested).
- **Every sibling has a larger output norm than LFM3-1.7B** (1.93 → 2.39–2.91).

## Which one to use

| Need | Pick |
|---|---|
| Plain text, longest context | LFM3-1.7B |
| Images → text | LFM3-VL-2B |
| Voice chat | LFM3-Audio-2B |
| Everything | LFM3-Omni-2B |
| Judge / score / choose | d1-2B (text+image) or d1-Omni-2B (+audio) |

Per-model reviews with every layer: see the other files in this folder.
