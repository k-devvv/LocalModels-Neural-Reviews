# Spark-X2.6-4B: a look inside the layers

[← all reviews](../README.md)

*Read from the local files: the GGUF Q4_K_M base build (3.54 GB) and its "abliterated" sibling (2.60 GB). Every number here was measured from the weights. No prompts were run and nothing was trained.*

---

## 1. Summary

Spark-X2.6-4B is a 36-block `spark2_5` model with a 2560-wide residual stream, a gated attention output (an `attn_gate` tensor next to a 4096 to 2560 `attn_output`) and a 131072-token vocabulary. The abliterated build has the largest drift from its base of any pair in this batch, with a mean relative change of 0.160 per tensor. That number is not a clean measure of the refusal edit. The two files use different quant recipes, and 73 of 290 tensors changed quant type. Even the 144 tensors that kept the same quant type still differ by 0.285 on average, so the abliterated file is also a substantive re-export, not just a small directional edit.

## 2. The model in one picture

```
tokens -> embed (131072 x 2560)
   |
   blocks 0..35, each:
     attn_norm -> attention (4096-wide heads) -> attn_gate -> attn_output (4096 -> 2560)
     ffn_norm  -> FFN -> ffn_down (-> 2560)
   |
   output_norm -> output (Q8_0 in base)

refusal-like direction r_hat: weak (gap ratio 1.09 base, 1.15 abliterated)
it lines up most with attn_output in blocks 10..15
```

## 3. What the layers do

| Signal | Early (blk 0) | Middle (blk 17) | Late (blk 35) |
|---|---|---|---|
| attn_norm mean gain | 0.258 | 0.288 | 0.395 |
| attn_norm max gain | 0.629 | 0.338 | 0.484 |
| ffn_down rms, base | 0.0186 | 0.0225 | 0.0142 |
| ffn_down rms, abliterated | 0.0265 | 0.0272 | 0.0295 |
| Block drift, base vs abliterated | 0.27 (blk 2) | 0.10 (blk 16) | 0.27 (blk 35) |

The attn_norm gains are identical in both files, so the edit left the norms alone. Gain climbs with depth after a spiky first block.

Quant split. Base: Q4_K 108, F32 73, Q8_0 37, BF16 37, Q6_K 36. Abliterated: Q4_K 180, F32 73, Q6_K 37. The base keeps 74 tensors at 8 or 16 bits that the abliterated file squeezes to Q4_K, which explains most of the 0.94 GB size gap.

## 4. What's strong

- The base build is generously quantized. A Q8_0 output head and 37 BF16 tensors make it closer to a mixed Q5 or Q6 file than a typical Q4_K_M.
- The residual-along-r̂ map is coherent. The five lowest values in both files are all attn_output in blocks 10 to 15 (about 0.27 against a typical 0.83). The model has one clear place where a refusal-like direction would be written.
- Norm gains are smooth and modest after block 0, with no runaway late-layer scaling.

## 5. Weak spots

- The eigengap is weak in both builds (1.091 base, 1.145 abliterated). The candidate direction barely separates from the next one, so treat r̂ as a hint, not a proof of a single refusal axis.
- The base has very large ffn_down row outliers in late blocks (row to rms ratio 594 in blk 35, 553 in blk 30, 543 in blk 26). Those rows are fragile under heavy quantization.
- The abliterated build flattens those outliers (its worst is 140 in blk 24 attn_output) and its ffn_down rms rises steadily with depth to 0.0295. That is a different weight distribution, not a light edit. The biggest block drifts (blk 30 0.294, blk 26 0.269, blk 35 0.265) sit exactly on the base outlier blocks.
- The abliterated file only narrows the residual along r̂ by a tiny amount (blk 15 0.266 to 0.263). On these numbers the direction is still present, so the large drift came mostly from something other than removing it.

## 6. Surgical map

| Target | Where | Why |
|---|---|---|
| Refusal-direction projection | attn_output, blocks 10 to 15 | Lowest residual along r̂ in both builds |
| Keep at high precision | ffn_down, blocks 26, 30, 34, 35 | Base row outliers above 500x rms |
| Leave untouched | all norms | Identical in both files, already well behaved |
| Re-check before trusting drift | any tensor with a quant-type change | 73 of 290 tensors differ in quant type |

## 7. Verdict

The base Spark-X2.6-4B Q4_K_M is a well-built, generously quantized file with a readable middle-block structure. The abliterated sibling should not be read as "base minus refusal". It is a re-quantized and re-exported model whose late blocks were reshaped, while the refusal-like direction it targets barely moved. To judge the abliteration on its own, compare it against a base file made with the same quant recipe.
