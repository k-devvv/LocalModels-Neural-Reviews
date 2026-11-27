# Raon-1-nano: a look inside the layers

[← all reviews](../README.md)

*Read from the local files: the GGUF Q4_K_M base build (2.08 GB) and its "Abliterated" sibling (2.08 GB). Every number here was measured from the weights. No prompts were run and nothing was trained.*

---

## 1. Summary

Raon-1-nano is a small decoder with 24 blocks and a 1536-wide residual stream, stored under its own `raon1nano` architecture. It is not a plain transformer. Each block carries extra `hc_*` mixing tensors (`hc_attn_chart`, `hc_ffn_chart`, `hc_attn_fn`, `hc_ffn_fn`) and an `inp_gate`, which points to a hyper-connection style design where several residual streams are blended at each block.

The main projection weights are calm and grow slowly with depth. The loud numbers in this model all live in the small `hc_*` tensors, not in attention or the MLP. The "Abliterated" build is a light edit focused on blocks 13 to 18, and it does not show a strongly removed refusal direction.

## 2. The model in one picture

```
 tokens ─► embed (1536)
            │
   ┌────────┴─────────────────────────────────────────┐
   │  24 blocks, each:                                 │
   │    hc_attn_chart / hc_attn_fn  ─► mix streams     │
   │    attn_norm ─► attention ─► attn_output          │
   │    hc_ffn_chart / hc_ffn_fn    ─► mix streams     │
   │    inp_gate ─► ffn_up/gate ─► ffn_down            │
   └────────┬─────────────────────────────────────────┘
            │   attn_norm mean: 0.70 ─► 0.32 (blk 8) ─► 0.92 (blk 20)
            ▼
         output
```

## 3. What the layers do

**Every layer.** 497 tensors in total. The norms are stored as plain weights, not as `1 + w`.

| Signal | Early (0-4) | Middle (8-12) | Late (16-23) |
|---|---|---|---|
| attn_norm mean | 0.695 (0), 0.38 (4) | 0.322 (8), 0.428 (12) | 0.707 (16), 0.924 (20) |
| ffn_down rms | ~0.088 | rising | ~0.097 |
| attn_output rms | ~0.081 | rising | ~0.094 |

The attention norm dips hard in the middle of the model and comes back up near the end. That means the middle blocks feed a weaker signal into attention, so they lean more on the residual path and on the `hc_*` mixing.

**Quant split.** F32 236, F16 85, Q4_K 85, Q6_K 55, Q8_0 36. A large share of the tensors are kept at F32 and F16. These are mostly the small norm and `hc_*` tensors, so they cost little space.

## 4. What's strong

- The attention and MLP projections are smooth. ffn_down rms moves only from 0.0876 to 0.0969 across 24 blocks, and attn_output from 0.0806 to 0.0943. Nothing jumps.
- The small control tensors are kept at high precision, so the quantizer does not touch the part of the model with the extreme values.
- The edit in the Abliterated build is small (mean drift 0.00517), so the sibling should behave almost the same as the base on ordinary text.

## 5. Weak spots

- **Extreme `hc_*` values.** blk.3 `hc_attn_chart` reaches a max_abs of 36,940.9. blk.18 reaches 4,152, and blk.23 `hc_ffn_chart` and blk.0 `hc_attn_chart` both reach 2,946.6. Channel 276 carries most of this. These tensors are tiny and almost zero elsewhere, so the ratios look enormous.
- **Degenerate row ratios.** blk.0 `hc_attn_fn` has a row/rms ratio of about 2.4e12, and blk.23 `hc_ffn_fn` about 2.2e8. These come from near-zero rms rather than a true outlier, but any tool that rescales these tensors will break them.
- **Gate outliers.** blk.21 `inp_gate` has a row at 378× rms, and blk.22 at 299×. These are the loudest rows in an ordinary-sized tensor.

## 6. Surgical map

| Target | Where | Why |
|---|---|---|
| Leave alone | all `hc_*` tensors, channel 276 | Huge values on tiny tensors. Edits or requantization here are risky. |
| Leave alone | blk.21-22 `inp_gate` | Strongest real row outliers in the model. |
| Refusal edits | blk.15, 20, 21, 23 | Lowest residual along r̂ (0.37-0.42) in ffn_down and attn_output. |
| Edited in the sibling | blk.13-18 | Drift peaks here: blk.16 0.0139, blk.18 0.0128, blk.17 0.0128, blk.15 0.0127, blk.13 0.0122. |
| Untouched in the sibling | blk.2, 8, 22 | Drift about 0.0005. |

## 7. Verdict

Raon-1-nano is a clean small model with an unusual stream-mixing design. Its extreme values are confined to the `hc_*` control tensors, which are stored at F32 or F16, so the Q4_K_M build keeps them intact.

The "Abliterated" build is not classic abliteration. The eigengap ratio is 1.055 for the base and 1.054 for the sibling, so neither has a clear single refusal direction. The lowest residuals along r̂ (0.37 to 0.42, across 49 tensors) are nearly identical in both files. The weight change is light (mean relative drift 0.00517) and centered on blocks 13 to 18. Read it as a small targeted fine-tune of the middle blocks. Expect behavior close to the base, and test refusals before you rely on it.
