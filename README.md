# 🧠 PaliGemma — Vision-Language Model from Scratch

> Complete from-scratch PyTorch implementation of Google's **PaliGemma** Vision-Language Model. No Hugging Face. No pre-built modules. Every tensor, every layer, every attention head — written from the ground up.

<p>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
</p>

---

## 🎯 What This Is

A full implementation of PaliGemma — Google's Vision-Language Model that combines a **SigLIP vision encoder** with a **Gemma language decoder** through a learned multimodal projector. This implementation covers the complete architecture from patch embeddings to autoregressive text generation.

This project was built to deeply understand how modern VLMs work at every layer — not just how to call an API, but how the model actually processes images and generates text.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Image + Text Prompt                │
└───────────┬─────────────────────────────────────┬───────────┘
            │                                     │
            ▼                                     ▼
┌───────────────────────┐              ┌──────────────────────┐
│   SigLIP Vision       │              │   Text Tokenizer     │
│   Transformer         │              │                      │
│                       │              │   Token Embeddings   │
│   • 12 Layers         │              │   at index 256000    │
│   • 16×16 Patches     │              │                      │
│   • 196 Visual Tokens │              └──────────┬───────────┘
│   • 768-dim output    │                         │
└───────────┬───────────┘                         │
            │                                     │
            ▼                                     │
┌───────────────────────┐                         │
│  Multimodal Projector │                         │
│  768 → 2048 dim       │                         │
│  Linear Projection    │                         │
└───────────┬───────────┘                         │
            │                                     │
            ▼                                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Gemma Language Decoder                       │
│                                                              │
│   • RMSNorm (pre-normalization)                              │
│   • Rotary Position Embeddings (RoPE)                        │
│   • Grouped Query Attention (KV heads)                       │
│   • KV-Cache for efficient generation                        │
│   • Causal Masking                                           │
│   • 8192-token context window                                │
│                                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: Generated Text Response                 │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Components Implemented

| Component | Details |
|-----------|---------|
| **SigLIP Vision Encoder** | 12-layer vision transformer, 16×16 patch embeddings, 196 visual tokens |
| **Multimodal Projector** | Linear projection from 768-dim vision space to 2048-dim language space |
| **Image-Token Merging** | Visual tokens merged into language sequence at embedding index 256000 |
| **RMSNorm** | Root Mean Square Layer Normalization (pre-norm architecture) |
| **RoPE** | Rotary Position Embeddings for relative position encoding |
| **Grouped KV Attention** | Efficient attention with separate key-value head groups |
| **KV-Cache** | Cached key-value pairs for efficient autoregressive generation |
| **Causal Masking** | Proper causal attention masks across multimodal token sequences |
| **Context Support** | Full 8192-token context window |

---

## 📁 Project Structure

```
Pytorch_PaliGemma/
├── modeling_gemma.py         # Gemma language model (decoder)
├── modeling_siglip.py        # SigLIP vision encoder
├── modeling_paligemma.py     # Full PaliGemma (vision + language)
├── processing_paligemma.py   # Image & text preprocessing
├── utils.py                  # Helper functions
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.9
torch >= 2.0
```

### Installation

```bash
git clone https://github.com/udayraj1238/Pytorch_PaliGemma.git
cd Pytorch_PaliGemma
pip install -r requirements.txt
```

### Inference

```python
from modeling_paligemma import PaliGemmaForConditionalGeneration
from processing_paligemma import PaliGemmaProcessor

model = PaliGemmaForConditionalGeneration.from_pretrained("path/to/weights")
processor = PaliGemmaProcessor(tokenizer, image_processor)

inputs = processor(text="Describe this image", images=image)
output = model.generate(**inputs)
```

---

## 🧪 What I Learned Building This

- **Image-token merging** isn't concatenation — the projector learns a mapping between fundamentally different representation spaces
- **RoPE** makes position information part of the attention computation itself, not just the input
- **KV-Cache** is where inference speed bottlenecks actually live — building it by hand reveals exactly why
- **Grouped KV attention** reduces memory proportional to the group ratio while preserving most of the representational capacity

---

## 📚 References

- [PaliGemma Paper](https://arxiv.org/abs/2407.07726) — Google DeepMind
- [SigLIP Paper](https://arxiv.org/abs/2303.15343) — Sigmoid Loss for Language Image Pre-Training
- [Gemma Paper](https://arxiv.org/abs/2403.08295) — Google DeepMind

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contact

**Uday Raj** — [LinkedIn](https://www.linkedin.com/in/uday6002/) · [Portfolio](https://udayraj1238.vercel.app) · [Email](mailto:rajuday6002@gmail.com)
