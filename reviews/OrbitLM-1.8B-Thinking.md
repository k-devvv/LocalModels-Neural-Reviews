# OrbitLM-1.8B-Thinking: a look inside the layers

[← all reviews](../README.md)

*Read from the local safetensors file (single `model.safetensors`). Every number here was measured from the weights. This is a single-model review: there is no abliterated sibling, so there are no refusal-direction or drift numbers. No prompts were run and nothing was trained.*

---

## 1. Summary

OrbitLM-1.8B-Thinking is a 24-layer `OuroForCausalLM` model with a 2048-wide residual stream, a 5632-wide FFN and a 151936-token vocabulary. It has 269 tensors. Two things set it apart from a plain decoder. Every layer carries two pairs of norms (a "sandwich" layout), and the model has an `early_exit_gate`, a single learned linear probe on the hidden state. Together they point to a looped or early-exit design, where the same stack can be run more than once or stopped early. The weights are tidy. Row outliers are mild, and the only striking signal is a sharp rise in the second set of norm gains in the last layer.

## 2. The model in one picture

```
tokens -> embed (151936 x 2048, rms 0.048)
   |
   layers 0..23, each:
     input_layernorm -> input_layernorm_2 -> attention (q,k,v,o all 2048 x 2048, no GQA)
     post_attention_layernorm -> post_attention_layernorm_2 -> MLP (2048 -> 5632 -> 2048)
   |
   model.norm (mean gain 1.107)
   |-> early_exit_gate (1 x 2048, bias -0.377)   "stop here?"
   |-> lm_head (151936 x 2048, untied, rms 0.138)
```

## 3. What the layers do

| Signal | Early (L0) | Middle (L12) | Late (L23) |
|---|---|---|---|
| input_layernorm mean gain | 1.016 | 1.225 | 1.312 |
| input_layernorm_2 mean gain | 0.823 | 0.910 | 1.776 |
| post_attention_layernorm mean gain | 0.838 | 0.665 | 0.996 |
| post_attention_layernorm_2 mean gain | 0.697 | 0.936 | 2.176 |
| o_proj rms | 0.0147 | 0.0146 | 0.0127 |
| down_proj rms | 0.0160 | 0.0158 | 0.0137 |

The first norm of each pair moves gently with depth. The second norm of each pair is where the action is: it roughly doubles in the final layer. Projection rms is flat through the stack and dips slightly at the end, so the late-layer scaling lives in the norms, not the weights.

## 4. What's strong

- Clean weight distribution. The worst row to rms ratio anywhere is 14.4 (L1 o_proj). The GGUF models in this batch show ratios in the hundreds, so this file should quantize without special care.
- The output head is untied and well scaled. lm_head rms is 0.138 against an embedding rms of 0.048, and its top row ratio is only 5.6.
- Full multi-head attention (q, k, v and o are all 2048 x 2048). There is no grouped-query compression, which is generous for a 1.8B model.
- The early-exit gate is tiny and easy to inspect: one weight row (rms 0.047) and a bias of -0.377, which leans toward "keep going" by default.

## 5. Weak spots

- The late norm gains spike. input_layernorm_2 reaches 1.78 and post_attention_layernorm_2 reaches 2.18 at L23, against values near 0.9 in the middle. That last layer amplifies its inputs more than any other, so it is the most sensitive place for quantization error or edits.
- The mild outliers cluster in o_proj of the first few layers (L1 14.4, L0 13.4, L2 12.4, L4 11.7). That is where precision matters most if the file is quantized.
- The architecture is custom. Standard GGUF converters and tools such as make_table.py in this repo do not handle it out of the box, so local use needs care.
- No abliterated build exists, so this review cannot say where a refusal-like direction lives.

## 6. Surgical map

| Target | Where | Why |
|---|---|---|
| Keep at high precision | o_proj, layers 0 to 4 | Highest row outliers in the model |
| Keep at high precision | all norms in layer 23 | Second-norm gains of 1.8 to 2.2 |
| Inspect for looping behaviour | early_exit_gate | Decides when the stack stops; bias leans toward continuing |
| Safe to quantize harder | middle-layer down_proj and o_proj | Flat rms, low outliers |

## 7. Verdict

OrbitLM-1.8B-Thinking is a well-conditioned small model with an unusual shape: sandwich norms and an early-exit gate around a full-attention stack. Its weights are among the cleanest in this batch. The one place to handle carefully is the final layer, where the second set of norms doubles its gain. Without an abliterated sibling, this is a structure review only.
