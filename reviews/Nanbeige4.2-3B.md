# Nanbeige4.2-3B: a look inside the layers

[← all reviews](../README.md)

*Read from the local files: the GGUF Q4_K_M base build (2.68 GB) and its "heretic" sibling (2.57 GB). Every number here was measured from the weights. No prompts were run and nothing was trained.*

---

## 1. Summary

Nanbeige4.2-3B is a compact decoder with 22 blocks and a 3072-wide residual stream, stored under its own `nanbeige` architecture with 201 tensors. It is a plain transformer layout with no extra mixing tensors.

The main signal in this model is a steady climb in the attention norm gain from the first block to the last. The loudest values sit in the final block, where the norms and the MLP both peak. The "heretic" build changes every block a little, but it does not remove a refusal direction, and part of its measured drift comes from a different quant layout rather than from an edit.

## 2. The model in one picture

```
 tokens ─► embed (3072)
            │
   ┌────────┴──────────────────────────────┐
   │  22 blocks, each:                      │
   │    attn_norm ─► attention ─► attn_out  │
   │    ffn_norm  ─► ffn_up/gate ─► down    │
   └────────┬──────────────────────────────┘
            │   attn_norm mean: 0.40 (blk 0) ─► 0.70 (blk 12) ─► 1.28 (blk 21)
            ▼
     output_norm ─► output
```

## 3. What the layers do

**Every layer.** 201 tensors in total. The norms are stored as plain weights, not as `1 + w`.

| Signal | Early (0-6) | Middle (9-15) | Late (18-21) |
|---|---|---|---|
| attn_norm mean | 0.399 (0), 0.462 (3), 0.535 (6) | 0.564 (9), 0.702 (12), 0.835 (15) | 0.882 (18), 1.275 (21) |
| ffn_down rms | 0.0272 to 0.0291 | 0.0289 to 0.0268 | 0.029 (18), 0.0242 (21) |
| attn_output rms | 0.0219 to 0.0249 | 0.0235 to 0.0241 | 0.0295 (18), 0.0284 (21) |

The attention norm gain more than triples across depth. Early blocks feed a small signal into attention and late blocks feed a large one, so the last few blocks have the most influence on the output. The projection weights stay nearly flat, so the growth is carried by the norms.

**Quant split.** The base uses Q4_K 81, Q5_K 44, Q6_K 31 and F32 45. The heretic build uses Q4_K 135, Q6_K 21 and F32 45, with no Q5_K. The two files were quantized with different recipes, which is why the heretic file is about 0.11 GB smaller.

## 4. What's strong

- The projection weights are very even. ffn_down rms stays between 0.024 and 0.029 across all 22 blocks, and attn_output between 0.022 and 0.030.
- The norm channel peaks are mild. The strongest single channel is blk.0 attn_norm ch1094 at 5.4× the tensor mean, followed by blk.11 attn_norm ch1777 at 5.0×. Nothing approaches the extreme channels seen in larger models.
- The largest absolute values are small. The top three are blk.21 ffn_norm at 5.8, output_norm at 5.4 and blk.21 attn_norm at 4.8.

## 5. Weak spots

- **A loud last block.** blk.21 holds the strongest row outliers in the model: ffn_down at 217× rms, attn_output at 195× and ffn_up at 193×. It also has the highest norm values. Edits or heavy requantization here will show up directly in the output.
- **An early outlier.** blk.1 ffn_up has a row at 213× rms, the second loudest in the model, together with blk.1 ffn_norm ch1252 at 4.9×.
- **Requant noise in the sibling.** Because the heretic file moved most Q5_K tensors to Q4_K, its drift numbers mix real edits with quantization error. They overstate how much the weights were changed.

## 6. Surgical map

| Target | Where | Why |
|---|---|---|
| Leave alone | blk.21 ffn_down, attn_output, ffn_up | Strongest row outliers (193× to 217×) and the highest norms. |
| Leave alone | blk.1 ffn_up | Row outlier at 213× rms. |
| Refusal edits | blk.13 and blk.11 attn_output, blk.6 and blk.11 ffn_down | Lowest residual along r̂ (0.195 to 0.256). |
| Edited in the sibling | blk.18, 6, 17, 5, 14 | Highest block drift: 0.0601, 0.0585, 0.0558, 0.0557, 0.0550. |
| Least changed in the sibling | blk.10, 0, 1 | Drift 0.0408, 0.0391, 0.0389. |

## 7. Verdict

Nanbeige4.2-3B is a tidy small model with even projection weights and a norm gain that rises steadily with depth. Its only loud region is the final block, plus one early row in blk.1.

The "heretic" build is not classic abliteration. The eigengap ratio is 1.147 for the base and 1.141 for the sibling, so neither has a clear single refusal direction. The lowest residuals along r̂ are nearly the same in both files (0.195 against 0.197, and 0.218 against 0.219). The mean relative drift is 0.0483 and it is spread over every block, with no block left untouched. Counting only the 147 tensors stored at the same quant type in both files, the mean drift drops to 0.0309, with a maximum of 0.1417, and some tensors such as `output.weight` are identical. Read it as a broad, light fine-tune of the whole model, made to look larger by the change in quant recipe. Test refusals before you rely on it.
