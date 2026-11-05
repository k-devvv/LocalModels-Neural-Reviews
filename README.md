# LocalModels Neural Reviews

Layer-by-layer reviews of local LLMs. Each review reads the model file directly (GGUF or safetensors), measures the norm gains and weight sizes of **every layer of every stack** (LLM, vision encoder, audio encoder, audio generator), and explains in plain language what each layer zone does, what the model is good at, and where it is safe to make surgical edits.

## Reviews

| Model | Base / type | Params | Quant | Review |
|---|---|---|---|---|
| TwIL-8B | IBM Granite 8B, dense, 40 layers | 8.79B | Q4_K_M (balanced) | [TwIL-8B.md](reviews/TwIL-8B.md) |
| Nemotron-Labs Dory 4B | looped, norm-free, 36 layers ×4 loop | 3.98B | Q5_K_M | [Nemotron-Dory-4B.md](reviews/Nemotron-Dory-4B.md) |
| LFM3-1.7B | LFM3 hybrid conv+attention, text | 1.73B | Q4_K_M | [LFM3-1.7B.md](reviews/LFM3-1.7B.md) |
| LFM3-VL-2B | LFM3 + vision | 1.73B + 189M | Q4_K_M / Q8_0 | [LFM3-VL-2B.md](reviews/LFM3-VL-2B.md) |
| LFM3-Audio-2B | LFM3 + hearing + speech out | 1.73B + 285M | Q4_K_M | [LFM3-Audio-2B.md](reviews/LFM3-Audio-2B.md) |
| LFM3-Omni-2B | LFM3 + vision + hearing + speech | 1.73B + 475M | Q4_K_M | [LFM3-Omni-2B.md](reviews/LFM3-Omni-2B.md) |
| d1-2B | LFM3 non-causal judge + vision | 1.73B + 189M | Q4_K_M / Q8_0 | [d1-2B.md](reviews/d1-2B.md) |
| d1-Omni-2B | LFM3 non-causal judge + vision + hearing | 1.73B + 305M | Q8_0 | [d1-Omni-2B.md](reviews/d1-Omni-2B.md) |
| Coding QLoRA adapter | LoRA r16 on Qwen3-1.7B | 17.4M | fp32 | [Coding-QLoRA-adapter.md](reviews/Coding-QLoRA-adapter.md) |

**Family overview:** [LFM3-family-comparison.md](reviews/LFM3-family-comparison.md): how the six LFM3 models share one LLM and which encoders are identical.

## Method

- Norm vectors: mean |gain|. Weight matrices: RMS of the first 256 rows after de-quantizing.
- LoRA: Frobenius norm of ΔW = B·A·(α/r) and effective rank per module.
- Zone readings are interpretation of these numbers, not activation probing.

Test machine: RTX 4050 6 GB, 16 GB RAM, Windows 11.
