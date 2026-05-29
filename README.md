<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7C3AED,50:8B5CF6,100:A78BFA&height=220&section=header&text=PaliGemma%20VLM&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Vision-Language%20Model%20from%20Scratch&descSize=18&descAlignY=55&descAlign=50" width="100%"/>

<!-- Typing SVG -->
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=A78BFA&center=true&vCenter=true&width=700&lines=No+Hugging+Face.+No+Pre-built+Modules.;Every+Tensor%2C+Every+Layer%2C+Every+Attention+Head;SigLIP+%2B+Gemma+%2B+Multimodal+Projector;Built+From+the+Ground+Up+in+PyTorch" alt="Typing SVG" /></a>

<br/>

<!-- Tech Badges -->
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />

<br/><br/>

<!-- Stats -->
<img src="https://img.shields.io/badge/12%20Layer-Vision%20Transformer-7C3AED?style=flat-square&labelColor=1a1a2e" />
<img src="https://img.shields.io/badge/196-Visual%20Tokens-8B5CF6?style=flat-square&labelColor=1a1a2e" />
<img src="https://img.shields.io/badge/8192-Token%20Context-A78BFA?style=flat-square&labelColor=1a1a2e" />
<img src="https://img.shields.io/badge/768→2048-Projection-C4B5FD?style=flat-square&labelColor=1a1a2e" />

</div>

---

<div align="center">
<h2>🎯 What This Is</h2>
<p><i>A complete from-scratch PyTorch implementation of Google's PaliGemma Vision-Language Model.</i></p>
</div>

> **Why from scratch?** Because calling `model = AutoModel.from_pretrained()` teaches you nothing about *how* the model actually processes images and generates text. Building every component by hand — from patch embeddings to KV-cache — is the only way to truly understand modern VLMs.

---

<div align="center">
<h2>🏗️ Architecture</h2>
</div>

```mermaid
graph TD
    A["🖼️ Input Image"] --> B["SigLIP Vision Encoder"]
    T["📝 Text Prompt"] --> C["Text Tokenizer"]
    
    subgraph Vision["🔵 SigLIP Vision Transformer"]
        B --> B1["16×16 Patch Embedding"]
        B1 --> B2["12 Transformer Layers"]
        B2 --> B3["196 Visual Tokens (768-dim)"]
    end
    
    subgraph Proj["🟣 Multimodal Bridge"]
        B3 --> P1["Linear Projection"]
        P1 --> P2["768 → 2048 dim"]
    end
    
    C --> C1["Token Embeddings"]
    
    subgraph LLM["🔴 Gemma Language Decoder"]
        P2 --> D1["Merged Sequence @ idx 256000"]
        C1 --> D1
        D1 --> D2["RMSNorm + RoPE"]
        D2 --> D3["Grouped KV Attention"]
        D3 --> D4["KV-Cache"]
        D4 --> D5["Causal Masking"]
        D5 --> D6["8192-token Context"]
    end
    
    D6 --> OUT["💬 Generated Text"]
    
    style A fill:#3B82F6,stroke:#3B82F6,color:#fff
    style T fill:#F59E0B,stroke:#F59E0B,color:#fff
    style B fill:#7C3AED,stroke:#7C3AED,color:#fff
    style B1 fill:#8B5CF6,stroke:#8B5CF6,color:#fff
    style B2 fill:#8B5CF6,stroke:#8B5CF6,color:#fff
    style B3 fill:#8B5CF6,stroke:#8B5CF6,color:#fff
    style P1 fill:#A78BFA,stroke:#A78BFA,color:#fff
    style P2 fill:#A78BFA,stroke:#A78BFA,color:#fff
    style C fill:#F59E0B,stroke:#F59E0B,color:#fff
    style C1 fill:#F59E0B,stroke:#F59E0B,color:#fff
    style D1 fill:#EF4444,stroke:#EF4444,color:#fff
    style D2 fill:#EF4444,stroke:#EF4444,color:#fff
    style D3 fill:#EF4444,stroke:#EF4444,color:#fff
    style D4 fill:#EF4444,stroke:#EF4444,color:#fff
    style D5 fill:#EF4444,stroke:#EF4444,color:#fff
    style D6 fill:#EF4444,stroke:#EF4444,color:#fff
    style OUT fill:#10B981,stroke:#10B981,color:#fff
```

---

<div align="center">
<h2>🧩 Components Built from Scratch</h2>
</div>

<table>
<tr>
<td align="center" width="33%">
<img src="https://img.shields.io/badge/🔵-SigLIP_Vision-7C3AED?style=for-the-badge&labelColor=1a1a2e" /><br/><br/>
<b>12-Layer Vision Transformer</b><br/>
16×16 patch embeddings<br/>
196 visual tokens<br/>
768-dim output space
</td>
<td align="center" width="33%">
<img src="https://img.shields.io/badge/🟣-Multimodal_Projector-8B5CF6?style=for-the-badge&labelColor=1a1a2e" /><br/><br/>
<b>Linear Projection Layer</b><br/>
768 → 2048 dimensions<br/>
Bridges vision & language<br/>
Merges at index 256000
</td>
<td align="center" width="33%">
<img src="https://img.shields.io/badge/🔴-Gemma_Decoder-EF4444?style=for-the-badge&labelColor=1a1a2e" /><br/><br/>
<b>Full Language Decoder</b><br/>
RMSNorm + RoPE<br/>
Grouped KV Attention<br/>
8192-token context
</td>
</tr>
</table>

<br/>

<details>
<summary><b>📋 Full Component Checklist (click to expand)</b></summary>
<br/>

| Component | Status | Details |
|:----------|:------:|:--------|
| SigLIP Vision Encoder | ✅ | 12 transformer layers, 768-dim embeddings |
| 16×16 Patch Embeddings | ✅ | Image → 196 visual tokens |
| Multimodal Projector | ✅ | Linear 768→2048 bridge layer |
| Image-Token Merging | ✅ | Visual tokens inserted at index 256000 |
| RMSNorm | ✅ | Root Mean Square pre-normalization |
| Rotary Position Embeddings (RoPE) | ✅ | Relative position encoding in attention |
| Grouped Query Attention | ✅ | Efficient KV head grouping |
| KV-Cache | ✅ | Cached key-value for fast generation |
| Causal Masking | ✅ | Proper multimodal causal masks |
| 8192-Token Context | ✅ | Full context window support |

</details>

---

<div align="center">
<h2>📁 Project Structure</h2>
</div>

```
Pytorch_PaliGemma/
├── 🔵 modeling_siglip.py         # SigLIP Vision Transformer (encoder)
├── 🔴 modelling_gemma.py         # Gemma Language Model (decoder)
├── 🟣 processing_paligemma.py    # Image & text preprocessing
└── 📄 README.md
```

---

<div align="center">
<h2>🚀 Quick Start</h2>
</div>

```bash
git clone https://github.com/udayraj1238/Pytorch_PaliGemma.git
cd Pytorch_PaliGemma
pip install torch torchvision pillow
```

```python
from modeling_siglip import SigLIPVisionModel
from modelling_gemma import GemmaForCausalLM

# Vision encoder
vision_model = SigLIPVisionModel(config)
visual_tokens = vision_model(pixel_values)  # [B, 196, 768]

# Project to language space
projected = projector(visual_tokens)  # [B, 196, 2048]

# Merge with text tokens and generate
output = gemma_model.generate(merged_input)
```

---

<div align="center">
<h2>💡 What I Learned</h2>
</div>

<table>
<tr>
<td>🔗</td>
<td><b>Image-token merging</b> is not concatenation — the projector learns a mapping between fundamentally different representation spaces</td>
</tr>
<tr>
<td>🔄</td>
<td><b>RoPE</b> makes position information part of the attention computation itself, not just the input</td>
</tr>
<tr>
<td>⚡</td>
<td><b>KV-Cache</b> is where inference bottlenecks live — building it by hand reveals exactly why</td>
</tr>
<tr>
<td>🧮</td>
<td><b>Grouped KV attention</b> reduces memory proportional to the group ratio while preserving representational capacity</td>
</tr>
</table>

---

<div align="center">
<h2>📚 References</h2>

[PaliGemma Paper](https://arxiv.org/abs/2407.07726) • [SigLIP Paper](https://arxiv.org/abs/2303.15343) • [Gemma Paper](https://arxiv.org/abs/2403.08295)

<br/>

<h2>🤝 Contact</h2>
<a href="https://www.linkedin.com/in/uday6002/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" /></a>
<a href="https://udayraj1238.vercel.app"><img src="https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=vercel&logoColor=white" /></a>
<a href="mailto:rajuday6002@gmail.com"><img src="https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white" /></a>

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7C3AED,50:8B5CF6,100:A78BFA&height=120&section=footer" width="100%"/>
