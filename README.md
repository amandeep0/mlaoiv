# MLAOIV: Machine Learning Approximate Optimal Instrumental Variables

This repository provides examples demonstrating the use of Machine Learning methods to construct Approximate Optimal Instrumental Variables (MLAOIV) for causal inference. It covers both:

1. **Linear IV-2SLS** — Simple instrumental variables regression with many/weak instruments
2. **General GMM** — Nonlinear models (e.g., BLP demand) via bi-level optimization that minimizes parameter variance subject to GMM moment conditions

## Overview

In many IV/GMM settings, we have access to many potential instruments but face efficiency loss from weak instruments or high dimensionality. MLAOIV uses machine learning to construct optimal instruments, improving estimation efficiency.

### Key Idea (Linear Case)

Instead of using all instruments $Z$ directly in 2SLS, we:
1. Use ML to predict the endogenous variable: $\hat{y}_1 = \hat{E}[y_1 | Z]$
2. Use cross-validation to avoid overfitting
3. Use $\hat{y}_1$ as the instrument in 2SLS

### Key Idea (General GMM)

For nonlinear GMM, we solve a bi-level optimization:
- **Outer**: Minimize variance of $\hat{\theta}$
- **Inner**: Subject to GMM first-order conditions

A neural network learns the optimal instrument transformation $H(Z)$.

## Examples

| Notebook | Description |
|----------|-------------|
| `01_many_weak_iv.ipynb` | Basic MLAOIV with many weak instruments (all instruments relevant with small weights) |
| `02_many_weak_instruments.ipynb` | Comprehensive comparison of ML methods (Lasso, Ridge, ElasticNet, KernelRidge, MLP) |
| `03_sparse_iv.ipynb` | Sparse instruments setting (only few instruments are strong) |
| `04_blp_gmm_optimal_iv.ipynb` | **General GMM framework**: BLP demand estimation with learned optimal instruments |

## General GMM Framework (Bi-Level Optimization)

The `04_blp_gmm_optimal_iv.ipynb` notebook demonstrates the general version of MLAOIV for nonlinear GMM estimation, such as BLP demand models.

### The Problem

In GMM estimation, we want to find parameters $\theta$ that satisfy moment conditions:
$$E[g(Z, \theta)] = 0$$

The GMM estimator minimizes:
$$\hat{\theta} = \arg\min_\theta \, g(\theta)' W \, g(\theta)$$

### Bi-Level Optimization

MLAOIV solves a **bi-level optimization** problem:

**Outer problem (minimize variance):**
$$\min_H \text{Var}(\hat{\theta})$$

**Inner problem (GMM estimation):**
$$\text{subject to: } \hat{\theta} = \arg\min_\theta \, g(\theta)' W(H) \, g(\theta)$$

where $H$ is a neural network that transforms raw instruments $Z$ into optimal instruments.

### Key Insight

The asymptotic variance of GMM is:
$$V = (G'WG)^{-1} (G'W S_0 W G) (G'WG)^{-1}$$

By learning the instrument transformation $H(Z)$, we can minimize the variance of $\hat{\theta}$ while ensuring the GMM first-order conditions are satisfied.

## Installation

```bash
pip install numpy pandas scikit-learn linearmodels joblib
```

## Quick Start

```python
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import RidgeCV
from linearmodels.iv import IV2SLS
import pandas as pd

# Generate data
n, d = 1000, 500
Z = np.random.normal(0, 1, (n, d))
u, e = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n).T
y1 = 0.1 * Z.sum(axis=1) + u  # Endogenous variable
y2 = 0.75 * y1 + e            # Outcome

# Compute MLAOIV
ridge = RidgeCV(cv=4, alphas=[1000, 100, 10, 1, 0.1])
mlaoiv = cross_val_predict(ridge, Z, y1, cv=3)

# IV-2SLS estimation
df = pd.DataFrame({'y2': y2, 'y1': y1, 'mlaoiv': mlaoiv})
model = IV2SLS.from_formula("y2 ~ 1 + [y1 ~ mlaoiv]", data=df).fit()
print(model.summary)
```

## Settings Covered

### Many Weak Instruments
- All $d$ instruments are relevant with equal small weights
- $\pi = (1, 1, ..., 1)$ scaled appropriately
- Ridge regression often works well

### Sparse Instruments  
- Only $s \ll d$ instruments are relevant
- $\pi = (1, ..., 1, 0, ..., 0)$ with $s$ ones
- Lasso excels due to variable selection

## ML Methods Compared

| Method | Best For |
|--------|----------|
| RidgeCV | Many weak instruments |
| LassoCV | Sparse instruments |
| ElasticNetCV | Mixed settings |
| KernelRidge | Non-linear relationships |
| MLPRegressor | Complex patterns |

## Citation

If you use this code, please cite:

**Singh, A., Hosanagar, K., & Gandhi, A. (2019). Machine Learning Instrument Variables for Causal Inference.**

Paper: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3352957](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3352957)

```bibtex
@article{singh2019mlaoiv,
  title={Machine Learning Instrument Variables for Causal Inference},
  author={Singh, Amandeep and Hosanagar, Kartik and Gandhi, Amit},
  journal={SSRN Working Paper},
  year={2019},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3352957}
}
```

## License

MIT License
