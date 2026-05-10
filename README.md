# Project 17: Build a Kernel That Knows the Physics

> **A kernel is not just a mathematical gadget. It is a hypothesis about what should count as similarity.**

This project investigates physics-informed kernel design for Gaussian Process / Kernel Ridge Regression on a ternary alloy composition problem. The central question is not just whether a custom kernel improves prediction, but *why* — and whether that reason is principled.

---

## Problem Statement

We model a synthetic alloy property `y` that depends on ternary element compositions `(c1, c2, c3)` constrained to the **simplex** (`c1 + c2 + c3 = 1`, all `ci ≥ 0`):

$$y = 2.0c_1 - 1.3c_2 + 0.8c_3 + 2.2c_1c_2 - 1.8c_2c_3 + \epsilon$$

The ground truth has two key physical features:
- **Pairwise element interactions** — the property depends on `c1*c2` and `c2*c3`, not just individual compositions.
- **A sign flip** — the interaction term `c2(2.2c1 - 1.8c3)` changes sign at the boundary `c3/c1 = 11/9`, meaning the effect of increasing `c2` on `y` reverses on either side of this line.

---

## Project Structure

```
Topic17.ipynb
├── 1. Data Generation          # Synthetic ternary alloy data on the simplex
├── 2. Kernel Definitions       # Five kernels with increasing physical awareness
├── 3. Baseline Comparison      # Global RMSE comparison across kernels
├── 4. Polynomial Degree Study  # Why degree 3–4 beats degree 2 on the simplex
├── 5. Regional Error Analysis  # Splitting test error by physical regime
└── Analysis (Q1–Q6)            # Written discussion of results and lessons
```

---

## Kernels Implemented

| Kernel | Similarity Concept | Physically Informed? |
|---|---|---|
| **Linear** | Similar composition ratios `c1:c2:c3` | No |
| **RBF** | Close in Euclidean distance | No |
| **Polynomial (deg 2)** | Similar compositions and pairwise interactions `ci*cj` | ✅ Yes (principled) |
| **Sign Flip (sum)** | RBF + explicit sign flip regime term | Partially |
| **Sign Flip (product)** | RBF × sign flip term | Partially (breaks PSD) |
| **Exact** | Encodes the exact terms of `y` directly | ⚠️ Reference only (cheating) |

All kernels are validated for **positive semi-definiteness** (PSD) before use — a necessary condition for valid kernel ridge regression. The Sign Flip product variant fails this check.

---

## Key Results

### Global RMSE Ranking

The polynomial and exact kernels outperform RBF and linear, confirming that encoding interaction structure helps.

### Regional Analysis (the critical finding)

Splitting the test set by the sign-flip boundary (`c3/c1 ≈ 11/9`) reveals what global RMSE hides:

| Kernel | Near Boundary RMSE | Away from Boundary RMSE | Failure Mode |
|---|---|---|---|
| **Polynomial** | 0.3210 | **0.2560** | None — consistent everywhere |
| **RBF** | 0.3128 | 0.3346 | Uniform, easy to diagnose |
| **Sign Flip** | ~0.305 | ~0.358 | Hidden — good locally, bad globally |

The Sign Flip kernel looks comparable to RBF by global RMSE (~0.333 vs ~0.333), but it secretly fails away from the boundary. Without regional analysis, this failure would be invisible.

---

## Core Lessons

1. **Physical correctness ≠ generalisation.** The Sign Flip kernel encoded the right physics (the boundary at `c3/c1 = 11/9` is real), but hardcoded specific numerical values rather than the structural class of interactions. Result: overfitting to the transition region.

2. **Polynomial kernel wins for principled reasons.** It encodes *pairwise element interactions as a class* (`ci*cj` for all pairs) without needing to know which pairs matter or what the coefficients are. This is the right level of physical abstraction.

3. **Simplex geometry explains the degree puzzle.** The minimum RMSE occurs at degree 3–4, not degree 2. Because `c1 + c2 + c3 = 1`, some degree-2 features are linearly dependent (e.g., `c2*c3 = c2 - c1*c2 - c2²`). Higher degrees introduce genuinely new terms that resolve this redundancy.

4. **Judge kernels by three criteria:**
   - **Global RMSE** — overall predictive performance
   - **Regional RMSE** — does it fail in physically meaningful subregions?
   - **Interpretation** — does its notion of similarity match the underlying physics?

---

## Dependencies

```
numpy
pandas
matplotlib
```

Standard Python scientific stack — no external ML libraries required (kernel ridge regression is implemented from scratch using `np.linalg.solve`).

---

## Usage

Open and run `Topic17.ipynb` in a Jupyter environment with Python 3.8+:

```bash
jupyter notebook Topic17.ipynb
```

All cells are self-contained and run top-to-bottom. The random seed is fixed (`np.random.default_rng(17)`) for reproducibility.

---

## Background

This project is part of a course on machine learning for materials science, exploring how domain knowledge can be embedded into the kernel function of a Gaussian process or kernel ridge regression model. The central thesis: a kernel that reflects the physical structure of the problem will generalise more robustly than one that merely fits the training data well.
