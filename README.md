# Mixture of Experts — From Scratch

A ground-up PyTorch implementation of Mixture of Experts (MoE) — built progressively from a toy 2D problem to MNIST digit classification.

---

## How it works

Instead of one big network, MoE trains several small **expert networks** and a **gating network** that routes each input to the best experts.

```
Input --> Gating Network --> picks top-k experts
                                      |
                         [E0] [E1] [E2] [E3]
                                      |
                              Weighted sum --> Prediction
```

Only top-k experts activate per input — more parameters, same compute.

---

## Structure

```
Mixture-of-Experts-MoE-/
├── moe.py     
└── README.md
```

---

## Results

**MNIST** — ~97% accuracy in 20 epochs on a free Colab T4.

<!-- Replace loss_curve.png with your actual image filename -->
![Loss Curve](/assets/moe_1.png)

---

## Key hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `n_experts` | 4 | Number of experts |
| `top_k` | 2 | Experts activated per input |
| `load_balancing_coeff` | 0.01 | Penalises uneven expert usage |

Set `load_balancing_coeff = 0` to watch expert collapse happen.

---

## Setup

```bash
pip install torch torchvision matplotlib seaborn scikit-learn
```

Runs on free Colab T4. No paid tier needed.

---

## What's next

- Replace FFN blocks in a small transformer with MoE layers
- Load and inspect routing in Mixtral 8x7B
- Expert pruning — deactivate low-usage experts and measure degradation
