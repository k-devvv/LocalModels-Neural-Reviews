# LFM3-Omni-2B: a look inside the layers

![skeleton](../assets/skeleton-lfm3-family.svg)

[← all reviews](../README.md)

*Source: local GGUF files read with the `gguf` library. LLM file quant: Q6_K embeddings, 1,728,801,600 params.*

---

## 1. Summary

**Everything in one:** the LFM3-Audio brain and voice, plus a vision encoder. Text, images and sound in; text and speech out.

---

## 2. The LLM in one picture

```
 tokens (vocab 128,000) ─► EMBEDDING 2048-wide (tied with output head)
          │
 ┌──────── 24 BLOCKS, two kinds, interleaved ────────────────────────────┐
 │ SHORT-CONV block  (13 of them: L0,1,3,5,…,23)   64.0M params each     │
 │   norm ─► short convolution "look at the last few tokens" ─► post-norm│
 │   norm ─► MLP 7,680 neurons                              ─► post-norm│
 │ GATED-ATTENTION block (11 of them: L2,4,…,22)    57.7M params each    │
 │   norm ─► attention 16 query / 4 key-value heads, gated  ─► post-norm│
 │   norm ─► MLP 7,680 neurons                              ─► post-norm│
 └───────────────────────────────────────────────────────────────────────┘
          │  attention L4,8,12,16,20 = sliding window 256; others global
          ▼
 OUTPUT NORM ─► OUTPUT HEAD (same matrix as embedding) ─► next token
```

| Item | Value |
|---|---|
| LLM parameters | 1,728,801,600 (same skeleton in all six LFM3 models) |
| Blocks | 24 = 13 short-conv + 11 gated-attention |
| Width | 2,048 |
| MLP neurons | 7,680 per block, 184,320 total |
| Attention | 16 query heads, 4 key/value heads, head size 128, q/k norms, output gate |
| Sliding window | 256 tokens on attention L4, 8, 12, 16, 20; L2, 6, 10, 14, 18, 22 see everything |
| RoPE base | 1,000,000 (global) / 10,000 (window) |
| Norms | pre-norm **and** post-norm ("sandwich") around every mixer and MLP |
| Embedding | 128,000 tokens, tied with output head |

---

## 3. What the layers do

> **Caveat:** numbers are measured from the weights (norm = mean |gain|, matrix = RMS of the first 256 rows after de-quantizing). The "what it means" readings are informed interpretation, not proven by activation probing.

| Zone | Blocks | What the numbers show | Likely job |
|---|---|---|---|
| **Entry** | 0–1 | conv mixer is strongest here (conv RMS highest at L0); MLP gain high at L1 | turn raw tokens into local word/phrase features |
| **Early mix** | 2–7 | first global attention at L2; post-norms still small | first look across the whole context |
| **Core** | 8–17 | flattest stretch; MLP gain lowest at L9 (0.138) | steady meaning-building; best place for edits |
| **Dead-ish attention** | 18 | q/k norm only 0.113/0.120 vs ~1.0+ elsewhere | attention scores here are nearly flat, so L18 attends almost uniformly; its MLP still works |
| **Exit** | 19–23 | post-norms climb to their max; MLP gain peaks (whole model) at L2, L1, L4, L22 | commit to the answer and shape the next token |

Other measured patterns:
- **Conv blocks** pre-norm gain 0.139–0.204; **attention blocks** pre-norm only 0.044–0.076. Attention gets a quieter input than conv.
- **Post-MLP norm** grows from 0.030 at L0 to 0.227 at L23: late blocks write louder into the stream.
- **q-norm** across attention blocks: 1.27, 1.18, 1.40, 1.15, 1.41, 1.28, 1.47, 1.19, 0.11, 1.26, 1.09 (L2→L22).
- **Output norm** 2.904.

### How far it moved from LFM3-1.7B

Mean relative change of every norm gain per block (0 = identical):

| L | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| drift | 0.12 | 0.20 | 0.17 | 0.21 | 0.18 | 0.16 | 0.13 | 0.15 | 0.13 | 0.12 | 0.13 | 0.13 | 0.15 | 0.10 | 0.09 | 0.11 | 0.09 | 0.11 | 0.15 | 0.12 | 0.08 | 0.08 | 0.09 | 0.23 |

Average **0.134**. Most changed: L23, L3, L1. Least changed: L20, L21, L16.

### Every LLM layer

`·` = tensor not present in that block type. Norms are gains; matrices are RMS.

#### omni · `blk` (24 blocks)

| L | type | M params | attn_n | q | k | v | o | q_n | k_n | gate_a | ffn_n | f_gate | f_up | f_down | post_a | post_f | conv | conv_in | conv_out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | conv | 64.0 | 0.186 | · | · | · | · | · | · | · | 0.232 | 0.0219 | 0.0198 | 0.0206 | 0.030 | 0.030 | 0.0403 | 0.0213 | 0.0234 |
| 1 | conv | 64.0 | 0.192 | · | · | · | · | · | · | · | 0.468 | 0.0241 | 0.0197 | 0.0207 | 0.028 | 0.036 | 0.0368 | 0.0215 | 0.0246 |
| 2 | attn | 57.7 | 0.054 | 0.0164 | 0.0178 | 0.0199 | 0.0207 | 1.270 | 1.213 | 0.0698 | 0.481 | 0.0247 | 0.0200 | 0.0197 | 0.018 | 0.030 | · | · | · |
| 3 | conv | 64.0 | 0.168 | · | · | · | · | · | · | · | 0.320 | 0.0229 | 0.0198 | 0.0203 | 0.033 | 0.030 | 0.0364 | 0.0214 | 0.0232 |
| 4 | attn | 57.7 | 0.068 | 0.0189 | 0.0191 | 0.0204 | 0.0197 | 1.185 | 1.138 | 0.0703 | 0.400 | 0.0236 | 0.0200 | 0.0197 | 0.023 | 0.029 | · | · | · |
| 5 | conv | 64.0 | 0.150 | · | · | · | · | · | · | · | 0.281 | 0.0223 | 0.0199 | 0.0197 | 0.044 | 0.031 | 0.0361 | 0.0213 | 0.0207 |
| 6 | attn | 57.7 | 0.058 | 0.0191 | 0.0174 | 0.0199 | 0.0206 | 1.397 | 1.353 | 0.0692 | 0.231 | 0.0216 | 0.0199 | 0.0198 | 0.027 | 0.029 | · | · | · |
| 7 | conv | 64.0 | 0.139 | · | · | · | · | · | · | · | 0.210 | 0.0218 | 0.0198 | 0.0195 | 0.066 | 0.032 | 0.0347 | 0.0211 | 0.0268 |
| 8 | attn | 57.7 | 0.059 | 0.0201 | 0.0181 | 0.0199 | 0.0203 | 1.152 | 1.118 | 0.0730 | 0.236 | 0.0220 | 0.0198 | 0.0195 | 0.029 | 0.031 | · | · | · |
| 9 | conv | 64.0 | 0.158 | · | · | · | · | · | · | · | 0.138 | 0.0206 | 0.0194 | 0.0193 | 0.055 | 0.034 | 0.0362 | 0.0210 | 0.0220 |
| 10 | attn | 57.7 | 0.055 | 0.0197 | 0.0179 | 0.0191 | 0.0200 | 1.408 | 1.380 | 0.0687 | 0.196 | 0.0215 | 0.0198 | 0.0195 | 0.042 | 0.039 | · | · | · |
| 11 | conv | 64.0 | 0.172 | · | · | · | · | · | · | · | 0.195 | 0.0211 | 0.0196 | 0.0194 | 0.054 | 0.046 | 0.0361 | 0.0210 | 0.0210 |
| 12 | attn | 57.7 | 0.064 | 0.0194 | 0.0189 | 0.0196 | 0.0200 | 1.275 | 1.229 | 0.0715 | 0.210 | 0.0215 | 0.0198 | 0.0195 | 0.055 | 0.047 | · | · | · |
| 13 | conv | 64.0 | 0.184 | · | · | · | · | · | · | · | 0.180 | 0.0214 | 0.0196 | 0.0197 | 0.058 | 0.056 | 0.0376 | 0.0211 | 0.0202 |
| 14 | attn | 57.7 | 0.060 | 0.0205 | 0.0176 | 0.0174 | 0.0194 | 1.465 | 1.430 | 0.0862 | 0.185 | 0.0216 | 0.0200 | 0.0196 | 0.081 | 0.072 | · | · | · |
| 15 | conv | 64.0 | 0.181 | · | · | · | · | · | · | · | 0.303 | 0.0221 | 0.0200 | 0.0198 | 0.071 | 0.065 | 0.0336 | 0.0213 | 0.0218 |
| 16 | attn | 57.7 | 0.076 | 0.0214 | 0.0179 | 0.0189 | 0.0193 | 1.192 | 1.159 | 0.0746 | 0.225 | 0.0216 | 0.0201 | 0.0199 | 0.059 | 0.069 | · | · | · |
| 17 | conv | 64.0 | 0.193 | · | · | · | · | · | · | · | 0.237 | 0.0218 | 0.0201 | 0.0199 | 0.083 | 0.078 | 0.0321 | 0.0213 | 0.0217 |
| 18 | attn | 57.7 | 0.044 | 0.0207 | 0.0112 | 0.0189 | 0.0189 | 0.113 | 0.120 | 0.0539 | 0.283 | 0.0220 | 0.0201 | 0.0199 | 0.025 | 0.085 | · | · | · |
| 19 | conv | 64.0 | 0.197 | · | · | · | · | · | · | · | 0.305 | 0.0221 | 0.0199 | 0.0198 | 0.099 | 0.101 | 0.0334 | 0.0213 | 0.0209 |
| 20 | attn | 57.7 | 0.068 | 0.0225 | 0.0190 | 0.0194 | 0.0201 | 1.261 | 1.228 | 0.0982 | 0.219 | 0.0215 | 0.0199 | 0.0198 | 0.094 | 0.105 | · | · | · |
| 21 | conv | 64.0 | 0.202 | · | · | · | · | · | · | · | 0.228 | 0.0214 | 0.0198 | 0.0197 | 0.120 | 0.131 | 0.0339 | 0.0212 | 0.0212 |
| 22 | attn | 57.7 | 0.047 | 0.0196 | 0.0164 | 0.0188 | 0.0196 | 1.091 | 1.024 | 0.0933 | 0.369 | 0.0224 | 0.0197 | 0.0196 | 0.174 | 0.147 | · | · | · |
| 23 | conv | 64.0 | 0.204 | · | · | · | · | · | · | · | 0.331 | 0.0223 | 0.0196 | 0.0190 | 0.156 | 0.227 | 0.0385 | 0.0218 | 0.0208 |

---

## 4. The extra senses (projector file `mmproj`, 474.5M params)

### Audio encoder (`a.blk`, 17 blocks) — identical in Audio, Omni and d1-Omni

Sound → `a.conv1d` subsampling (256 channels) → `a.pre_encode.out` [4096→768] → 17 blocks of width 768:
- **10 conv blocks** (6.5M each): depthwise conv `c_dw` + two pointwise convs `c_pw1`/`c_pw2` (Conformer style);
- **7 attention blocks** (5.9M each) at L2, 5, 8, 11, 14, 15, 16. The last three are back-to-back: the top of the encoder is attention-heavy, to tie sounds into words.

Norms climb with depth: conv `ln1` 0.34→0.74, attention `ln1` 0.03→0.41, `ln2` 0.23→0.71. Then `a.post_ln` (0.108) → `mm.a.mlp` (768→2048→2048) into the LLM.

### Audio generator (speech out) — identical in Audio and Omni

Two stages turn the LLM's hidden state into sound:
1. **Code transformer** (`a.gen.code.blk`, 3 gated-attention blocks, 11.3M each, width 1,024). `proj_in` [2048→12288] feeds it; `out_embd` [2048×24,204] = **12 codebooks × 2,017 codes**. `head_norm` ≈1.05, plus a `mask_embd` (masked/parallel code prediction).
2. **Waveform decoder** (`a.gen.wav.tfm.blk`, 8 blocks, width 512: 5 conv at 4.6M, 3 attention at 4.2M). `in_proj` [256→512], `out_proj` [512→1282], quantizer codebook 4×2,016. Its post-MLP norm rises 0.58 → 1.50 at L7, and conv `c_dw` peaks at L5 (0.229): the last blocks do the loudest shaping of the waveform.

### Vision encoder (`v.blk`, 14 blocks)

Images are cut into 16×16 patches (two F16 patch-embedding tensors, 3→1024), then pass 14 encoder blocks of width 1,024 (MLP 2,560):
- **8 conv blocks** (`c_in` / `c_sp` / `c_out`: in-projection, spatial depthwise conv, out-projection);
- **6 attention blocks** at L2, 4, 6, 8, 10, 12.

Measured highlights:
- **L12 attention** has the largest value/output weights of the encoder: it gathers the final image summary.
- **L13 MLP down-projection collapses** (much smaller than every other block): an output bottleneck before the adapter.
- **Adapter:** `mm.input_norm` → `mm.0` [4096×4096] → `mm.2` [4096→2048]. It folds patches into the LLM's 2,048-wide stream.

### Every projector layer

#### omni_proj · `a.blk` (17 blocks)

| L | type | M params | q | k | v | o | q_n | k_n | f_gate | f_up | f_down | c_dw | c_pw1 | c_pw2 | ln1 | ln2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | conv | 6.5 | · | · | · | · | · | · | 0.0291 | 0.0292 | 0.0290 | 0.0438 | 0.0310 | 0.0316 | 0.343 | 0.229 |
| 1 | conv | 6.5 | · | · | · | · | · | · | 0.0293 | 0.0286 | 0.0291 | 0.0386 | 0.0304 | 0.0300 | 0.402 | 0.206 |
| 2 | attn | 5.9 | 0.0286 | 0.0284 | 0.0284 | 0.0282 | 0.953 | 0.951 | 0.0292 | 0.0285 | 0.0289 | · | · | · | 0.033 | 0.231 |
| 3 | conv | 6.5 | · | · | · | · | · | · | 0.0294 | 0.0290 | 0.0294 | 0.0437 | 0.0311 | 0.0304 | 0.543 | 0.258 |
| 4 | conv | 6.5 | · | · | · | · | · | · | 0.0298 | 0.0291 | 0.0296 | 0.0432 | 0.0310 | 0.0306 | 0.554 | 0.257 |
| 5 | attn | 5.9 | 0.0297 | 0.0292 | 0.0286 | 0.0290 | 0.940 | 0.926 | 0.0298 | 0.0290 | 0.0292 | · | · | · | 0.073 | 0.263 |
| 6 | conv | 6.5 | · | · | · | · | · | · | 0.0298 | 0.0292 | 0.0294 | 0.0407 | 0.0309 | 0.0308 | 0.568 | 0.282 |
| 7 | conv | 6.5 | · | · | · | · | · | · | 0.0296 | 0.0290 | 0.0298 | 0.0417 | 0.0310 | 0.0310 | 0.559 | 0.287 |
| 8 | attn | 5.9 | 0.0294 | 0.0283 | 0.0311 | 0.0294 | 1.022 | 1.021 | 0.0298 | 0.0287 | 0.0292 | · | · | · | 0.107 | 0.299 |
| 9 | conv | 6.5 | · | · | · | · | · | · | 0.0301 | 0.0292 | 0.0296 | 0.0417 | 0.0310 | 0.0309 | 0.584 | 0.343 |
| 10 | conv | 6.5 | · | · | · | · | · | · | 0.0302 | 0.0295 | 0.0301 | 0.0437 | 0.0311 | 0.0314 | 0.638 | 0.383 |
| 11 | attn | 5.9 | 0.0296 | 0.0280 | 0.0302 | 0.0302 | 0.999 | 0.982 | 0.0300 | 0.0289 | 0.0293 | · | · | · | 0.194 | 0.410 |
| 12 | conv | 6.5 | · | · | · | · | · | · | 0.0309 | 0.0296 | 0.0301 | 0.0455 | 0.0314 | 0.0313 | 0.705 | 0.482 |
| 13 | conv | 6.5 | · | · | · | · | · | · | 0.0311 | 0.0300 | 0.0305 | 0.0466 | 0.0317 | 0.0313 | 0.742 | 0.561 |
| 14 | attn | 5.9 | 0.0285 | 0.0283 | 0.0315 | 0.0309 | 1.027 | 1.020 | 0.0312 | 0.0299 | 0.0303 | · | · | · | 0.242 | 0.633 |
| 15 | attn | 5.9 | 0.0294 | 0.0278 | 0.0309 | 0.0309 | 1.167 | 1.172 | 0.0313 | 0.0302 | 0.0306 | · | · | · | 0.313 | 0.671 |
| 16 | attn | 5.9 | 0.0307 | 0.0288 | 0.0316 | 0.0319 | 1.109 | 1.101 | 0.0322 | 0.0307 | 0.0302 | · | · | · | 0.414 | 0.706 |

#### omni_proj · `a.gen.code.blk` (3 blocks)

| L | type | M params | q | k | v | o | q_n | k_n | gate_a | f_gate | f_up | f_down | ln1 | ln2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | attn | 11.3 | 0.0290 | 0.0289 | 0.0264 | 0.0278 | 0.984 | 0.986 | 0.0216 | 0.0286 | 0.0268 | 0.0266 | 1.043 | 0.982 |
| 1 | attn | 11.3 | 0.0271 | 0.0225 | 0.0295 | 0.0268 | 0.962 | 0.966 | 0.0168 | 0.0264 | 0.0275 | 0.0270 | 1.104 | 1.105 |
| 2 | attn | 11.3 | 0.0269 | 0.0214 | 0.0341 | 0.0290 | 0.960 | 0.964 | 0.0230 | 0.0282 | 0.0320 | 0.0288 | 1.157 | 1.270 |

#### omni_proj · `a.gen.wav.tfm.blk` (8 blocks)

| L | type | M params | q | k | v | o | q_n | k_n | gate_a | f_gate | f_up | f_down | c_dw | c_pw1 | c_pw2 | post_a | post_f | ln1 | ln2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | conv | 4.6 | · | · | · | · | · | · | · | 0.0849 | 0.0871 | 0.0784 | 0.1629 | 0.0854 | 0.0797 | 0.409 | 0.580 | 0.674 | 0.614 |
| 1 | conv | 4.6 | · | · | · | · | · | · | · | 0.0868 | 0.0900 | 0.0841 | 0.1898 | 0.0937 | 0.0870 | 0.579 | 0.534 | 0.693 | 0.678 |
| 2 | attn | 4.2 | 0.0856 | 0.0847 | 0.0808 | 0.0818 | 1.017 | 1.008 | 0.0540 | 0.0924 | 0.0955 | 0.0836 | · | · | · | 0.528 | 0.580 | 0.712 | 0.740 |
| 3 | conv | 4.6 | · | · | · | · | · | · | · | 0.0922 | 0.0934 | 0.0851 | 0.2084 | 0.0950 | 0.0887 | 0.553 | 0.636 | 0.718 | 0.726 |
| 4 | attn | 4.2 | 0.0896 | 0.0887 | 0.0828 | 0.0815 | 1.184 | 1.176 | 0.0578 | 0.0928 | 0.0944 | 0.0863 | · | · | · | 0.631 | 0.731 | 0.719 | 0.727 |
| 5 | conv | 4.6 | · | · | · | · | · | · | · | 0.0936 | 0.0965 | 0.0863 | 0.2287 | 0.0968 | 0.0906 | 0.769 | 1.140 | 0.678 | 0.715 |
| 6 | attn | 4.2 | 0.0908 | 0.0881 | 0.0872 | 0.0767 | 0.883 | 0.872 | 0.0517 | 0.0926 | 0.0969 | 0.0824 | · | · | · | 0.849 | 1.445 | 0.680 | 0.806 |
| 7 | conv | 4.6 | · | · | · | · | · | · | · | 0.0844 | 0.0838 | 0.0699 | 0.1838 | 0.0899 | 0.0775 | 1.281 | 1.496 | 0.699 | 0.743 |

#### omni_proj · `v.blk` (14 blocks)

| L | type | M params | q | k | v | o | q_n | k_n | f_gate | f_up | f_down | c_in | c_sp | c_out | ln1 | ln2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | conv | 12.1 | · | · | · | · | · | · | 0.0611 | 0.0624 | 0.0496 | 0.0450 | 0.0558 | 0.0492 | 0.062 | 0.026 |
| 1 | conv | 12.1 | · | · | · | · | · | · | 0.0616 | 0.0607 | 0.0584 | 0.0446 | 0.0550 | 0.0561 | 0.100 | 0.039 |
| 2 | attn | 11.0 | 0.0629 | 0.0594 | 0.0581 | 0.0565 | 1.025 | 1.024 | 0.0619 | 0.0623 | 0.0583 | · | · | · | 0.007 | 0.046 |
| 3 | conv | 12.1 | · | · | · | · | · | · | 0.0620 | 0.0605 | 0.0605 | 0.0554 | 0.0487 | 0.0563 | 0.128 | 0.057 |
| 4 | attn | 11.0 | 0.0623 | 0.0616 | 0.0631 | 0.0602 | 1.130 | 1.155 | 0.0656 | 0.0604 | 0.0593 | · | · | · | 0.018 | 0.068 |
| 5 | conv | 12.1 | · | · | · | · | · | · | 0.0627 | 0.0587 | 0.0597 | 0.0576 | 0.0529 | 0.0566 | 0.163 | 0.083 |
| 6 | attn | 11.0 | 0.0670 | 0.0645 | 0.0689 | 0.0665 | 1.155 | 1.148 | 0.0629 | 0.0600 | 0.0603 | · | · | · | 0.040 | 0.101 |
| 7 | conv | 12.1 | · | · | · | · | · | · | 0.0642 | 0.0587 | 0.0594 | 0.0608 | 0.0559 | 0.0591 | 0.214 | 0.122 |
| 8 | attn | 11.0 | 0.0626 | 0.0627 | 0.0720 | 0.0678 | 1.201 | 1.204 | 0.0677 | 0.0615 | 0.0613 | · | · | · | 0.081 | 0.153 |
| 9 | conv | 12.1 | · | · | · | · | · | · | 0.0671 | 0.0611 | 0.0624 | 0.0636 | 0.0634 | 0.0613 | 0.300 | 0.178 |
| 10 | attn | 11.0 | 0.0651 | 0.0630 | 0.0761 | 0.0760 | 1.148 | 1.147 | 0.0715 | 0.0655 | 0.0678 | · | · | · | 0.109 | 0.192 |
| 11 | conv | 12.1 | · | · | · | · | · | · | 0.0656 | 0.0701 | 0.0770 | 0.0658 | 0.0755 | 0.0650 | 0.308 | 0.194 |
| 12 | attn | 11.0 | 0.0617 | 0.0598 | 0.1123 | 0.0906 | 1.038 | 1.037 | 0.0747 | 0.0771 | 0.0830 | · | · | · | 0.137 | 0.190 |
| 13 | conv | 12.1 | · | · | · | · | · | · | 0.0800 | 0.0873 | 0.0312 | 0.0414 | 0.0352 | 0.0390 | 0.193 | 0.267 |

---

## 5. What's strong

1. **Most complete model in the lab:** 3 input senses, 2 output modes.
2. **Reuses parts exactly:** audio encoder and generator are identical to LFM3-Audio, so the LLM did not have to re-learn voice.
3. **Stronger vision encoder** than LFM3-VL (weights ~2× larger RMS, adapter norm 1.0 vs 0.45).

## 6. Weak spots

1. **Largest projector file** (474.5M params): ~28% of the model is not the LLM.
2. **Same audio-side drift as Audio:** entry and exit re-tuned, text-only skill may drop.
3. **Q4_K_M everywhere**, including the vision and voice towers.

---

## 7. Surgical map

| Goal | Best place | Why |
|---|---|---|
| Fact / habit edit (ROME-style) | **MLP of L8–17** | flattest, lowest-gain core; edits stay contained |
| Steering vector | **residual after L10–16** | middle of the stack, before the exit zone amplifies |
| LoRA | **attention q/v/o of L2–16 + MLP L8–17** | attention blocks are few (11), so this is cheap |
| Revive / prune | **L18 attention** | near-flat attention; candidate to prune or to re-train with a small LoRA |
| Leave alone | **L0–1 and L21–23** | entry builds token features; exit has the highest gains and drift |
| Projector | **adapter (`mm.*`) only** | the encoders are shared with sibling models; tune the adapter, not the tower |

---

## 8. Verdict

LFM3-Omni-2B is a **1.73B-param LFM3 hybrid** (13 conv + 11 attention blocks) that fits easily on a 6 GB GPU. Its layers follow the family plan: busy entry, quiet core, near-off L18 attention, loud exit. Edit in the core, protect the ends.
