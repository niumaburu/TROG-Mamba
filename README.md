
# TROG-Mamba 🚀  
### Structure-Aware State Evolution Modeling for Graph-based Long-Term Time Series Forecasting

---

## 🌟 Visual Abstract (Model Overview)

<p align="center">
<img src="assets/model.png" width="85%">
</p>

**Figure 1. Overall architecture of TROG-Mamba.**  
The framework integrates a Bi-directional Temporal Mamba Backbone (Bi-TSE), a Sparse Graph Residual Constructor (SGRC), and a Signed Orthogonal Residual Regulator (SORR).  
Graph structure is treated as *residual correction* rather than a parallel propagation stream.

---

## 🧠 Core Idea

Traditional methods suffer from:
- ❌ GNN: over-smoothing & over-squashing  
- ❌ Transformer: quadratic complexity  
- ❌ Vanilla Mamba: ignores graph structure  

👉 TROG-Mamba resolves this by:

> **"Using graph structure as controlled residual correction over temporal state evolution."**

---

## 🏗️ Model Architecture

```
Input Multivariate Time Series
        │
        ▼
RevIN Normalization
        │
        ▼
Embedding Layer
(Value + Variable + Temporal)
        │
        ▼
────────────────────────────
Bi-TSE Backbone (Mamba)
────────────────────────────
Forward State Scan  →→→
Backward State Scan ←←←
        │
        ▼
Gated Fusion
        │
        ▼
────────────────────────────
SGRC Module
Sparse Graph Residual Construction
────────────────────────────
• Top-k Graph Sparsification
• Multi-hop Diffusion
• Selective Neighborhood Aggregation
        │
        ▼
────────────────────────────
SORR Module
Signed Orthogonal Regulation
────────────────────────────
• Tanh Signed Gate (-1 ~ 1)
• Orthogonal Residual Compensation
• Adaptive Scaling
        │
        ▼
Feedforward Enhancement
        │
        ▼
Forecasting Head
        │
        ▼
Final Prediction
```

---

## 🔥 Key Highlights

### ⚡ 1. Bi-TSE Backbone (Bidirectional Mamba)
- Linear complexity modeling
- Forward + backward temporal reasoning
- Better long-range dependency capture

### 🧩 2. SGRC (Sparse Graph Residual Constructor)
- Adaptive graph sparsification
- Multi-hop topology diffusion
- Temporal-state-guided neighbor filtering

### 🛡️ 3. SORR (Signed Orthogonal Residual Regulator)
- Signed correction (positive / negative / suppressive)
- Orthogonal feature decomposition
- Prevents redundant accumulation

---

## 📊 Why It Works

TROG-Mamba succeeds because it:

✔ Separates temporal modeling and structural correction  
✔ Avoids dense graph propagation  
✔ Injects graph information only at key stages  
✔ Controls noise via signed + orthogonal residuals  

---

## 📈 Performance Summary

- 🚀 Avg MAE improvement: **21.29%**
- 🚀 Avg MSE improvement: **28.22%**

Benchmarked on:
- Traffic
- Electricity
- PeMS03 / PeMS04 / PeMS07 / PeMS08

---

## 📌 Installation

```bash
git clone https://github.com/niumaburu/TROG-Mamba
cd TROG-Mamba
pip install -r requirements.txt
```

---

## 🏃 Training

```bash
python run.py \
  --model TROG-Mamba \
  --dataset Traffic \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 32 \
  --lr 3e-4
```

---

## 📦 Citation

```bibtex
@article{trogmamba2026,
  title={TROG-Mamba: Structure-Aware State Evolution Modeling for Graph-based Long-Term Time Series Forecasting},
  author={Your Name},
  journal={ESWA},
  year={2026}
}
```

---



## ⚠️ Note

This repository is the official implementation of TROG-Mamba.  
More updates and pretrained models will be released soon.

