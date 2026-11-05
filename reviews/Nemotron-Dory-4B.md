# Nemotron-Labs Dory 4B: a look inside the layers

*Source: local GGUF (Q5_K_M), read with the `gguf` library. 3,975M params.*

---

## 1. Summary

Dory is a **looped (recurrent-depth) transformer**: some of its layers are run **several times in a row** on the same token. The file holds 36 logical layers, but the model thinks through **192 blocks of compute**. It is an RL-tuned checkpoint with a **262k context**.

It also has **no normalisation layers at all**. Instead every block has learned scalars (`residual_alpha`, gate/qk/up scales) that set how loud it is.

---

## 2. The model in one picture

```
 tokens ─► EMBEDDING (BF16, separate from output head)
            │
 INPUT STACK      blocks 0–15   (8 layers)   run ×1
            │
 RECURRENT STACK  blocks 16–55  (20 layers)  run ×4  ◄─┐ same weights, 4 passes
            │                                          ─┘
 OUTPUT STACK     blocks 56–71  (8 layers)   run ×1
            │
 OUTPUT SCALE (mean 2.02) ─► OUTPUT HEAD (BF16) ─► next token

 effective depth = 16 + 40×4 + 16 = 192 blocks
```

| Item | Value |
|---|---|
| Params | 3,975M (Q5_K_M; embedding + head kept in BF16) |
| Blocks in file | 72 = 36 layers × (attention block + MLP block) |
| Width | 2,560 |
| MLP neurons | 7,168 |
| Attention | 16 query / 4 key-value heads, head size 256 |
| Window | sliding window 1,024; every 4th layer global |
| Norms | **none**; learned per-block scales instead |
| Context | 262,144 |

---

## 3. What the layers do

> **Caveat:** numbers are measured from the weights. The "what it means" readings are informed interpretation, not proven by activation probing.

| Zone | Layers (blocks) | What the numbers show | Likely job |
|---|---|---|---|
| **Input** | 0–7 (0–15) | residual α falls 0.180 → 0.148 → 0.123; MLP gate scale highest (1.99 → 1.1) | encode the prompt once, loudly |
| **Loop, early** | 8–11 (16–23) | gate scale drops 1.44 → 0.84 | the loop body starts quieter so 4 passes don't blow up |
| **Loop, middle** | 12–22 (24–45) | residual α lowest band (~0.07–0.09); gate scale ~0.6–0.8 | small careful updates per pass: the "thinking" iterations |
| **Loop, late** | 22–27 (45–55) | gate scale back up to ~1.0 | each pass ends with a stronger write |
| **Output** | 28–35 (56–71) | **block 56: lowest residual (0.030) but highest gate & qk scales**; block 70 qk 0.058; block 71 up 0.966; gate scale ends 0.62 | turn the looped state into a token |

- Matrix RMS is **~0.0198 in every block**: all the per-layer differences live in the scalars, not in the weights.
- **Block 56** is the hand-off from the loop: it barely adds to the stream but attends very sharply.

### Every layer

| Layer | blocks | zone | window | res_α attn | res_α ffn | attn gate_s | qk_s | ffn gate_s | ffn up_s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0+1 | input (×1) | SWA 1024 | 0.180 | 0.178 | 0.031 | 0.029 | 1.99 | 1.56 |
| 1 | 2+3 | input (×1) | SWA 1024 | 0.148 | 0.148 | 0.027 | 0.030 | 1.95 | 1.62 |
| 2 | 4+5 | input (×1) | SWA 1024 | 0.123 | 0.127 | 0.027 | 0.030 | 1.81 | 1.47 |
| 3 | 6+7 | input (×1) | global | 0.119 | 0.165 | 0.026 | 0.031 | 1.42 | 1.59 |
| 4 | 8+9 | input (×1) | SWA 1024 | 0.137 | 0.159 | 0.025 | 0.031 | 1.56 | 1.60 |
| 5 | 10+11 | input (×1) | SWA 1024 | 0.114 | 0.154 | 0.023 | 0.032 | 1.33 | 1.59 |
| 6 | 12+13 | input (×1) | SWA 1024 | 0.118 | 0.152 | 0.021 | 0.030 | 1.46 | 1.57 |
| 7 | 14+15 | input (×1) | global | 0.159 | 0.181 | 0.021 | 0.029 | 1.11 | 1.49 |
| 8 | 16+17 | recurrent (×4) | SWA 1024 | 0.109 | 0.171 | 0.034 | 0.029 | 1.44 | 1.61 |
| 9 | 18+19 | recurrent (×4) | SWA 1024 | 0.113 | 0.161 | 0.021 | 0.031 | 1.24 | 1.56 |
| 10 | 20+21 | recurrent (×4) | SWA 1024 | 0.106 | 0.145 | 0.021 | 0.027 | 0.84 | 1.51 |
| 11 | 22+23 | recurrent (×4) | global | 0.202 | 0.126 | 0.018 | 0.030 | 0.84 | 1.53 |
| 12 | 24+25 | recurrent (×4) | SWA 1024 | 0.073 | 0.113 | 0.023 | 0.030 | 0.79 | 1.52 |
| 13 | 26+27 | recurrent (×4) | SWA 1024 | 0.070 | 0.109 | 0.022 | 0.030 | 0.80 | 1.47 |
| 14 | 28+29 | recurrent (×4) | SWA 1024 | 0.089 | 0.095 | 0.021 | 0.030 | 0.76 | 1.42 |
| 15 | 30+31 | recurrent (×4) | global | 0.112 | 0.108 | 0.018 | 0.029 | 0.69 | 1.47 |
| 16 | 32+33 | recurrent (×4) | SWA 1024 | 0.066 | 0.116 | 0.022 | 0.028 | 0.61 | 1.44 |
| 17 | 34+35 | recurrent (×4) | SWA 1024 | 0.078 | 0.126 | 0.022 | 0.030 | 0.60 | 1.41 |
| 18 | 36+37 | recurrent (×4) | SWA 1024 | 0.096 | 0.129 | 0.019 | 0.029 | 0.69 | 1.43 |
| 19 | 38+39 | recurrent (×4) | global | 0.113 | 0.125 | 0.018 | 0.034 | 0.63 | 1.39 |
| 20 | 40+41 | recurrent (×4) | SWA 1024 | 0.101 | 0.123 | 0.021 | 0.030 | 0.77 | 1.39 |
| 21 | 42+43 | recurrent (×4) | SWA 1024 | 0.096 | 0.124 | 0.022 | 0.032 | 0.76 | 1.40 |
| 22 | 44+45 | recurrent (×4) | SWA 1024 | 0.084 | 0.126 | 0.023 | 0.033 | 0.99 | 1.42 |
| 23 | 46+47 | recurrent (×4) | global | 0.137 | 0.123 | 0.019 | 0.029 | 0.97 | 1.43 |
| 24 | 48+49 | recurrent (×4) | SWA 1024 | 0.128 | 0.124 | 0.022 | 0.030 | 1.03 | 1.41 |
| 25 | 50+51 | recurrent (×4) | SWA 1024 | 0.086 | 0.131 | 0.023 | 0.031 | 0.87 | 1.36 |
| 26 | 52+53 | recurrent (×4) | SWA 1024 | 0.107 | 0.143 | 0.022 | 0.032 | 0.94 | 1.38 |
| 27 | 54+55 | recurrent (×4) | global | 0.124 | 0.124 | 0.017 | 0.028 | 0.86 | 1.35 |
| 28 | 56+57 | output (×1) | SWA 1024 | 0.030 | 0.122 | 0.049 | 0.044 | 0.86 | 1.37 |
| 29 | 58+59 | output (×1) | SWA 1024 | 0.098 | 0.118 | 0.024 | 0.031 | 0.87 | 1.37 |
| 30 | 60+61 | output (×1) | SWA 1024 | 0.093 | 0.120 | 0.023 | 0.028 | 0.95 | 1.36 |
| 31 | 62+63 | output (×1) | global | 0.088 | 0.121 | 0.022 | 0.029 | 0.86 | 1.40 |
| 32 | 64+65 | output (×1) | SWA 1024 | 0.128 | 0.121 | 0.028 | 0.033 | 0.68 | 1.58 |
| 33 | 66+67 | output (×1) | SWA 1024 | 0.093 | 0.109 | 0.028 | 0.032 | 0.67 | 1.77 |
| 34 | 68+69 | output (×1) | SWA 1024 | 0.090 | 0.097 | 0.029 | 0.036 | 0.67 | 1.72 |
| 35 | 70+71 | output (×1) | global | 0.088 | 0.119 | 0.035 | 0.058 | 0.62 | 0.97 |

---

## 4. What's strong

1. **Deep thinking for its size:** 192 blocks of compute from 4B params. Reasoning per byte is very high.
2. **Long context:** 262k, with mostly sliding-window attention (cheap KV cache).
3. **Clean scale design:** with no norms, every layer's volume is one readable number. That makes it easy to inspect and steer.
4. **Precise ends:** embedding and output head kept in BF16.

## 5. Weak spots

1. **4× loop = ~4× slower** per token than a plain 4B model; on 6 GB, expect slow generation.
2. **Edits in the loop are applied 4 times.** A small change in layers 8–27 is multiplied.
3. **No norms** means it is sensitive to quantization and to fine-tuning learning rates.
4. **Large BF16 embedding/head** eat VRAM.

---

## 6. Surgical map

| Goal | Best place | Why |
|---|---|---|
| Fact edit | **input stack MLPs (layers 3–7)** | run once; edit is not repeated |
| Steering vector | **output of the input stack (after block 15)** | one injection, then the loop refines it |
| LoRA | **input + output stacks first**; loop only with a low rank / low LR | loop updates compound ×4 |
| Leave alone | **block 56** and **blocks 70–71** | loop hand-off and final token shaping |

---

## 7. Verdict

Dory is the **most unusual model in the lab**: a 4B looped, norm-free transformer that thinks like a much deeper model. Strong for reasoning, slow per token. Edit outside the loop.
