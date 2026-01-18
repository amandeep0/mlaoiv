# MLAOIV: Machine Learning Approximate Optimal Instrumental Variables

This repository provides examples demonstrating the use of Machine Learning methods to construct Approximate Optimal Instrumental Variables (MLAOIV) for IV-2SLS estimation.

## Overview

In many IV settings, we have access to many potential instruments. MLAOIV uses machine learning to construct the optimal instrument as $\hat{E}[y_1 | Z]$ using cross-validation, which can improve efficiency over standard IV methods.

### Key Idea

Instead of using all instruments $Z$ directly in 2SLS, we:
1. Use ML to predict the endogenous variable: $\hat{y}_1 = \hat{E}[y_1 | Z]$
2. Use cross-validation to avoid overfitting
3. Use $\hat{y}_1$ as the instrument in 2SLS

## Examples

| Notebook | Description |
|----------|-------------|
| `01_many_weak_iv.ipynb` | Basic MLAOIV with many weak instruments (all instruments relevant with small weights) |
| `02_many_weak_instruments.ipynb` | Comprehensive comparison of ML methods (Lasso, Ridge, ElasticNet, KernelRidge, MLP) |
| `03_sparse_iv.ipynb` | Sparse instruments setting (only few instruments are strong) |

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

```bibtex
@article{mlaoiv,
  title={Machine Learning Approximate Optimal Instrumental Variables},
  author={},
  journal={},
  year={}
}
```

## License

MIT License
