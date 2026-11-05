# TwIL-8B: a look inside the layers

*Source: the local file `TwIL-8B-Q4_K_M-balanced.gguf`, read directly with the `gguf` library. Norm values are exact. Weight sizes are the RMS of the first 256 rows of each matrix after un-quantizing.*

---

## 1. Summary

TwIL-8B is an **8.8-billion-parameter model built on IBM's Granite design**. Its internal name is "Granite8B L2Sv2 S30". It was fine-tuned and then compressed to 4-bit ("Q4_K_M balanced"). The compression was calibrated on TwIL's own domain data (`twil-domains` imatrix), so the bits were spent where TwIL's tasks need them.

It is a **standard, clean transformer**:
- no mixture-of-experts;
- no unusual blocks;
- 40 identical layers stacked on top of each other.

That makes it a good model to study and to edit surgically.

---

## 2. The model in one picture

```
 text ─► TOKENIZER (100,352 word-pieces)
          │
          ▼
 EMBEDDING  411M params   turns each token into a list of 4,096 numbers
          │
          ▼
 ┌─────────────────── LAYER × 40 (each ~199M params) ───────────────────┐
 │  RMSNorm  ─► ATTENTION  "look at other tokens"        ~42M  (21%)    │
 │               32 question heads, 8 shared key/value heads             │
 │  RMSNorm  ─► MLP        "think about what was found"  ~157M (79%)    │
 │               12,800 neurons                                          │
 └───────────────────────────────────────────────────────────────────────┘
          │
          ▼
 OUTPUT HEAD  411M params  scores all 100,352 tokens ─► next word
```

| Item | Value |
|---|---|
| Total parameters | 8,791,592,960 |
| Layers | 40 |
| Width (numbers per token) | 4,096 |
| MLP neurons | 12,800 per layer, **512,000 total** |
| Attention heads | 32 per layer, **1,280 total** (8 key/value groups per layer, which saves memory) |
| Max context | 131,072 tokens (RoPE base 10,000,000, built for long text) |
| Default sampling | temperature 1.0, top_p 0.95 |

**Simple analogy.** Each layer is one round of a meeting:
1. **Attention** is everyone looking around the table: "which earlier words matter for this one?"
2. **The MLP** is each person thinking alone, using what they know.

This repeats 40 times. Each round refines the answer a little more.

---

## 3. What the layers do

> **Caveat:** the numbers below are measured from the weights. The "what it means" column is an informed reading of those numbers plus what is known about similar transformers. It is not proven for TwIL. Proof needs activation probing (see section 6).

Each layer has two **norm gains**: one before attention, one before the MLP. They act as volume knobs on how strongly that layer pushes its result forward. Here is how they change with depth.

| Zone | Layers | Attention gain | MLP gain | What it means |
|---|---|---|---|---|
| **Entry** | 0 | 0.37 | 0.59 | Very quiet. Mostly passes the raw token meaning through. |
| **Reading** | 1–9 | 1.2 → 2.8 | 0.8 → 1.1 | Climbs fast. Builds syntax, local patterns and "what kind of text is this". |
| **Thinking core** | 10–30 | 2.7 → 3.8 | 1.2 → 1.55 | Flat and steady. Logic, code structure, facts. All 21 layers look nearly alike. |
| **Decision** | 31–39 | 3.8 → **6.4** | 1.6 → **4.0** | Gains shoot up, with the MLP gain ×2.5 in 6 layers. The model commits to an answer and shapes the exact next token. |

Attention also changes shape near the end:
- **Key size falls by half** (0.018 → 0.008).
- **Value size doubles** (0.010 → 0.020).

So late layers care less about *where* to look and more about *what* to copy in. They are gathering final details for the answer.

The **MLP weights stay very even everywhere**: down-projection RMS is 0.0031–0.0033 in every layer. Where a layer's influence differs, it comes from the norm gains, not from the MLP weights.

### Where the compression spent its bits

"Balanced" Q4_K_M gives some parts 6-bit precision (Q6_K) instead of 4-bit:
- **First 5 layers (0–4) and last 6 layers (34–39):** value and MLP-down weights are 6-bit.
- **Middle layers:** every third layer gets 6-bit, the rest are 4-bit.
- **Output head:** 6-bit.

The quantizer protected the entry and the decision zone. That matches the measurements above: those zones are the most sensitive.

---

## 4. What's strong

1. **Strong coding fix on a real-style task.** It scored **12/12** on our hidden-test SWE task: a 6-file shop repo with 3 bugs. It fixed rounding, BOGO and negative totals, and changed nothing that was already correct.
2. **Reasoning mode.** It thinks in `<think>` before answering, which helps with multi-step bugs.
3. **Fits small hardware.** It runs on a 6 GB RTX 4050 at 16k context. The 8 shared key/value groups keep memory use about 4× smaller than full attention would.
4. **Long-context design.** 128k native context with a large RoPE base. It can read big files in principle, but VRAM is the limit.
5. **Smart compression.** It was calibrated on its own domain data, and extra precision goes to the sensitive first and last layers.
6. **Clean, standard design.** It loads in llama.cpp, Ollama and LM Studio. That makes it easy to analyse, edit, or LoRA-tune.
7. **Large vocabulary (100k).** Code, symbols and many languages fit into fewer tokens, so it reads more per token.

## 5. Weak spots

1. **Thinking uses a lot of tokens.** On long prompts it can run out of budget before it writes the answer. Our SWE test had to give it a large thinking budget.
2. **4-bit is a loss.** The Q4 file is slightly less precise than the original weights. It also can't be trained directly, because training needs BF16 weights.
3. **Speed on 6 GB.** 16k context is about the comfortable limit. The full 128k won't fit on this GPU.
4. **The decision layers (34–39) are fragile.** Their gains are 2–4× higher than the rest. A small change there makes a big change in the output.

---

## 6. Surgical map: where to edit what

| Goal | Best place | Why |
|---|---|---|
| Edit a fact or API habit (ROME/MEMIT) | **MLP of layers 12–24** | Steady "thinking core". Facts live in MLP neurons, and edits there stay contained. |
| Push a behaviour, e.g. "write shorter thinking" (steering vector) | **Layers 14–22** | Middle layers carry style and intent. Steering there is strong but stable. |
| Small targeted LoRA for coding | **attn q/v + MLP in layers 10–30** | Most of the model's reasoning happens here. The cost is about 0.1–0.5% of parameters. |
| Leave alone | **Layer 0–2 and 34–39** | Entry layers hold basic token meaning. Decision layers have high gains, so edits there are amplified and break output easily. |

To find **which of the 12,800 neurons in one layer** handle coding, the next step is activation probing:
1. Run coding prompts through the model.
2. Record which neurons fire.
3. Compare against non-coding prompts.

That shows which neurons are worth scaling or editing.

---

## 7. Verdict

TwIL-8B is a **compact, well-built Granite-based reasoning coder**. It is sized and compressed with care for local use, and it already solves small multi-file bug-fix tasks fully.

Its layers follow a clear plan: a quiet entry, a reading zone, a steady 21-layer thinking core, and a loud 6-layer decision zone. That makes it **a good candidate for surgical editing**: work in the middle, protect the ends.
