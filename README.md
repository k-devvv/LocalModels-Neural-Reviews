# LocalModels Neural Reviews

Layer-by-layer reviews of local LLMs. Each review reads the model's GGUF file directly, measures the norm gains and weight sizes of every layer, and explains in plain language what each layer zone does, what the model is good at, and where it is safe to make surgical edits.

## Reviews

| Model | Base | Quant | Review |
|---|---|---|---|
| TwIL-8B | IBM Granite 8B | Q4_K_M (balanced) | [reviews/TwIL-8B.md](reviews/TwIL-8B.md) |

Test machine: RTX 4050 6 GB, 16 GB RAM, Windows 11.
