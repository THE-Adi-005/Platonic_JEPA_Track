# AI Agent Context & Instructions: Platonic Attribute Grounding (PAG-JEPA)

> **READ THIS FIRST IF YOU ARE AN AI ASSISTANT (Claude, ChatGPT, Gemini, Cursor, Copilot, Antigravity, etc.)**  
> This file is the primary context injection header for this repository. Read this document completely before generating code, proposing modifications, or advising the user.

---

## 1. Quick Identity & Context Primer

- **Project Lead / Author:** Adithya R. (GitHub: `THE-Adi-005`)
- **Working Paper Title:** *Platonic Attribute Grounding: Proving and Exploiting Cross-Modal Alignment in Video Without Autoregressive Hallucination*
- **Primary Theoretical Basis:** **Randall Balestriero & Yann LeCun (October 2026, arXiv:2610.39302)**, *"Proving and Improving the Platonic Representation Hypothesis with JEPAs"*.
- **Target Venues:** CVPR / NeurIPS / ICLR (Main Technical Track).
- **Core Hardware Constraint:** **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM GDDR6, ~4.8 GB usable)**. All models, batch sizes, and pipelines MUST strictly run inside this memory boundary (< 2.5 GB peak VRAM).

---

## 2. What Is This Repository? (The 2-Minute Summary)

Standard Vision-Language Models (VLMs) like Qwen2.5-VL, InternVL, and LLaVA hallucinate human attributes in video (e.g. guessing someone is a *"software engineer"* or *"anxious"*) because their next-token prediction language decoders rely on text co-occurrence priors rather than visual evidence.

Balestriero & LeCun (2026) proved mathematically that:
1. Multi-modal Joint-Embedding Predictive Architectures (JEPAs) converge to the **exact same representation up to an orthogonal rotation $R \in \mathcal{O}(d)$** if and only if positive pairs are **"Good Views"** (independent observations of the exact same scene with bounded misalignment $\epsilon$).
2. Next-token autoregression is fundamentally a JEPA with **"Bad Views"** (predicting sequential text tokens from documents), explaining why language decoders require billions of parameters and still hallucinate.
3. Good-view JEPAs require **42× fewer training samples** than next-token autoregression.

**This repository implements PAG-JEPA:**
- Replaces heavy autoregressive decoders with a **compact 35M parameter non-autoregressive JEPA**.
- Enforces an orthogonal rotation between video representations (`DINOv2-small`, $d=384$) and structured attribute representations (`BGE-small` / Distil-JEPA, $d=384$) via **Orthogonal Procrustes / Stiefel Manifold parameterization**.
- Uses a **calibrated rejection threshold $\tau_{\text{grounding}}$**: if an attribute is unobservable in the video pixels, the model outputs *"Not Observable"* rather than hallucinating a guess.

---

## 3. Relationship to the Companion Dataset Project

This repository (`Platonic_JEPA_Track`) is **Paper 2 (Model Track)** in a two-paper research initiative.  
It is designed to run either **standalone** (with pre-computed tensors or synthetic validation sets) or **connected** to the companion benchmark repository:

| Entity | Role in Research | Location / Source |
|---|---|---|
| **Companion Paper 1 (`Dataset_Track`)** | *"Say Only What You Can See"* — CVPR Datasets & Benchmarks Track. Established the 56,187 video crawl, standard taxonomies (ISCO-08, Plutchik), and 3-pass blind human verification benchmark proving 48.6% of web claims are ungrounded ("Bad Views"). | `../Dataset_Track/` or GitHub repo |
| **Companion Raw Video Data (`Dataset`)** | 56,187 raw `.mp4` video clips across 7 archive directories (`part-1-...-001` to `007`) + `data_part1.json`. | `../Dataset/` |
| **This Repo (`Platonic_JEPA_Track`)** | **Paper 2 (Model Track)** — CVPR / NeurIPS Main Track. The algorithmic innovation: uses Paper 1's cleaned "Good Views" to train a non-autoregressive JEPA, validating the Platonic Representation Theorem on video. | Current Repository |

*(For full details on directory linkages, see `COMPANION_PROJECT_ECOSYSTEM.md`).*

---

## 4. Architectural Rules for AI Agents

When writing or editing code in this repository, you **MUST** strictly obey the following rules:

1. **NO Heavy Autoregressive Decoders:**
   - Do NOT import, load, or fine-tune multi-billion parameter autoregressive language models (e.g. do not load full Qwen-7B or Llama-8B for generation). The entire scientific premise of this paper is that **non-autoregressive JEPAs eliminate token decoders**.
2. **Strict 6 GB VRAM Boundary:**
   - Keep batch sizes at 32 or 64.
   - Use frozen backbones (`DINOv2-small` for video, `BGE-small` or `sentence-transformers` for text).
   - Only train the lightweight temporal pooling and projection adapters (~1.5M parameters).
3. **Orthogonal Geometry Constraint:**
   - The projector $g_\theta(z_v)$ must approximate an orthogonal transformation $R \in \mathcal{O}(d)$ such that $R^T R = I_d$. Use the Kabsch/Procrustes closed-form solution ($Z_v^T Z_a = U \Sigma V^T \implies R^* = U V^T$) or Stiefel manifold Riemannian gradient updates.
4. **Anti-Collapse Regularization:**
   - Always train using VICReg (Invariance + Variance hinge $\gamma=1.0$ + Covariance decorrelation) or Barlow Twins / Spectral SigReg to prevent representation collapse.
5. **Calibrated Codebook Gating:**
   - Attribute prediction must use cosine similarity against the pre-computed Platonic taxonomy codebook. If $\max_k \text{sim}(\hat{z}_a, z_k) < \tau_{\text{grounding}}$, output class $-1$ ("Not Observable").

---

## 5. Key Novel Concepts to Know

If the user mentions these terms, here is their precise meaning in this project:

- **Good Views:** Video clips paired with human-verified, observable attribute claims (bounded misalignment $\epsilon < 0.15$).
- **Bad Views:** Raw web video clips paired with uncleaned captions containing unobservable claims (e.g. unseen professions, contradicted emotions; $\epsilon \to \infty$).
- **APAD (Attribute-Stratified Platonic Alignment Distance):** The Procrustes alignment error $\| Z_v R - Z_a \|_F / \sqrt{d}$ calculated separately for each attribute dimension (Gender, Age, Emotion, Profession, Style). Used to prove that *geometry predicts hallucination*.
- **Singular Value Spectrum (Figure 1):** The singular values $\Sigma$ of the cross-covariance matrix $Z_v^T Z_a$. Under Good Views, $\Sigma$ is flat and uniform; under Bad Views, $\Sigma$ collapses exponentially.
- **AGS (Attribute Grounding Score):** Decomposed precision and recall across attribute claims:
  $$\text{AGS-P} = \frac{\text{Grounded Predicted Claims}}{\text{Total Predicted Claims}}, \quad \text{AGS-R} = \frac{\text{Grounded Predicted Claims}}{\text{Total True Observable Claims}}$$
- **CHAIR-Video:** Caption Hallucination Assessment ($CHAIR_s$ = percentage of samples with $\ge 1$ hallucination).

---

## 6. Common Tasks & Execution Commands

When the user asks you to execute tasks, refer to these scripts:

```bash
# 1. Run the Singular Value Spectrum experiment (Paper Fig 1):
python scripts/run_apad_spectrum.py --config configs/config.yaml

# 2. Train the PAG-JEPA projection adapter on RTX 4050:
python scripts/train_jepa.py --config configs/config.yaml --epochs 20 --batch_size 64

# 3. Evaluate APAD and hallucination rates:
python -m src.metrics.apad --config configs/config.yaml
```

---

## 7. Recommended Response Style for This Project

- Be concise, direct, mathematically rigorous, and code-accurate.
- Prioritize memory-efficient PyTorch implementations.
- Always reference the connection to Balestriero & LeCun (2026) when explaining theoretical reasoning.
