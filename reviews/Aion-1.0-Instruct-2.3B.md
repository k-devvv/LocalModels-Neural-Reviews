# Aion-1.0-Instruct-2.3B: a look inside the layers

[← all reviews](../README.md)

*Read from the local files (GGUF Q4_K_M "AD" build, 1.58 GB, and its "Uncensored-AD" sibling). Every number here was measured from the weights. No prompts were run and nothing was trained.*

---

## 1. Summary

Aion is a 2.3B decoder with 28 blocks and a 2048-wide residual stream. The weights are tidy. The norms sit almost exactly at 1.0, the projection sizes grow gently with depth, and the only loud rows are in the first two or three blocks, which is normal for a small model.

The "Uncensored" build is a very light edit. Blocks 0 to 8 are byte-for-byte the same, and blocks 9 to 27 moved by under 1% on average. The abliteration check finds no clean refusal direction in either file, and the two files give almost the same answer, so the uncensoring was not done by classic abliteration. It looks more like a small fine-tune of the upper two thirds.

## 2. The model in one picture

```
 tokens ─► token_embd (Q6_K, 2048 wide)
             │
   blk 0-2   │  loud rows in attn_output / ffn_down   ◄─ "write" heads that set up the stream
   blk 3-8   │  quiet, steady                         ◄─ identical in both builds
   blk 9-13  │  lowest energy along r̂ (≈0.35×)        ◄─ where the uncensor edit starts
   blk 14-27 │  rms climbs slowly 0.032 → 0.039       ◄─ edited lightly (≈0.9% each)
             ▼
          output norm ─► lm head
```

## 3. What the layers do

### Every layer

| Region | What the numbers say |
|---|---|
| Norms | All 113 F32 tensors are norms or biases. attn_norm means stay at about 1.0 in every block. They are stored as plain weights, not 1+w. |
| Blocks 0-2 | The largest single rows in the model: blk.0 attn_output has a row 314× its own rms, blk.0 ffn_down 137×, blk.1 and blk.2 attn_output about 110×. These are the usual "massive activation" writers. |
| Blocks 3-27 | ffn_down rms is flat at 0.031 to 0.034. attn_output rms rises from 0.020 (blk.0) to 0.039 (blk.24), so late attention writes a bit harder. |
| Channel 1351 | Shows up as the top channel in many attn_norm tensors, but its ratio is 1.0, so it is a tie, not an outlier. |

### Where the compression spent its bits

The 310 tensors split as F32 113, Q8_0 56, Q4_K 79, Q5_K 33, Q6_K 29. Q8_0 goes mostly to attn_v and other small, sensitive projections, Q6_K to the embedding and some ffn_down, and the rest is Q4_K or Q5_K. That is a standard Q4_K_M layout with no surprises.

## 4. What's strong

- Clean norms and no runaway channels, so the model should quantize and merge well.
- Only the first three blocks hold large rows, which keeps outlier handling simple.
- The uncensored sibling leaves blocks 0 to 8 untouched, so the early stream is identical.

## 5. Weak spots

- The refusal signal is diffuse. gap_ratio is 1.066 in both builds, which is a weak eigengap, so a single-direction abliteration would have little to grab.
- The large rows in blk.0 to blk.2 make those blocks the most fragile under further quantization.

## 6. Surgical map

| Target | Measurement | Note |
|---|---|---|
| Refusal direction | gap_ratio 1.066 (AD and Uncensored) | Weak, no clean single direction |
| Lowest residual along r̂ | ≈0.33-0.37× random at blk.9-13 attn_output | Same in both builds |
| Early blocks | token_embd 0.96×, blk.0 attn_output 1.10×, blk.1-5 fall from 0.64× to 0.46× | |
| Drift, AD → Uncensored | mean rel. Frobenius 0.00525 | blk.0-8 exactly 0, max blk.11 at 0.00898 |

## 7. Verdict

A well-behaved small model with a conventional quant layout. The Uncensored-AD build is a gentle edit of blocks 9 to 27 and leaves the refusal geometry almost where it was, so expect its behavior to differ only modestly from the base. If you want to abliterate it yourself, aim at blocks 9 to 13, but the weak eigengap means results may be uneven.
