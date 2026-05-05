# End-to-End Markowitz Portfolio Optimization

Reproduction and empirical study of **"Machine Learning Meets Markowitz"** (Wang, Gao, Harvey, Liu & Tao, 2026), which proposes an end-to-end (E2E) framework that unifies return prediction and portfolio optimization into a single differentiable pipeline.

## Paper Summary

The traditional "predict-then-optimize" approach trains a return forecast model using statistical objectives (e.g., MSE) that treat all cross-sectional prediction errors equally, then feeds the forecasts into a Markowitz optimizer. This two-stage separation is problematic because:

- **Heterogeneous preferences**: Different investors value forecast precision for different stocks, yet MSE produces one-size-fits-all forecasts.
- **Economic constraints**: Position limits and budget constraints are ignored during prediction.
- **Transaction costs**: Real-world frictions are absent from the statistical objective.

The E2E framework merges prediction and optimization by backpropagating portfolio-level objectives (return, risk, transaction cost) through the optimization layer directly into the prediction model, using the KKT implicit function theorem.

## Repository Structure

```
├── Data/
│   ├── hs300_factor_data_filled.parquet     # Raw CSI 300 factor data (filled)
│   ├── factor_panel_v2_processed.parquet    # Processed cross-sectional factor panel
│   └── 03v3_ic_summary.csv                 # Factor IC statistics for screening
│
├── E2E_optimization_experiment/
│   ├── Return-Only_experiment.ipynb         # Sec 4.2: Pure return maximization
│   ├── Return_with_Risk_experiment.ipynb    # Sec 4.4: Mean-SD utility (risk aversion)
│   └── Return_Risk_with_TC_experiment.ipynb # Sec 4.4: Full Eq.(21) with transaction costs
│
└── README.md
```

## Experiments

### 1. Return-Only (Section 4.2)

**Notebook**: `Return-Only_experiment.ipynb`

Optimizes the pure return objective with a smoothed LP layer:

$$\min_w \; -\mu^\top w + \frac{\zeta}{2}\|w\|^2 \quad \text{s.t.} \quad \sum w_i = 1,\; 0 \le w_i \le \frac{10}{N}$$

- Uses a **dual smoothing** strategy: large $\zeta$ during training (rich gradients via more free variables) and small $\zeta$ at test time (strict top-10% equal-weight portfolio, as the paper intends).
- Forward pass solves the LP via `brentq` bisection (O(N log(1/eps))) — ~1500x faster than SLSQP.
- Backward pass applies the **KKT implicit function theorem** on the free variable set.

**Key result**: E2E Sharpe = 1.124 vs MSE Sharpe = 0.848, with lower turnover (120% vs 147%).

### 2. Return with Risk (Section 4.4, v5b)

**Notebook**: `Return_with_Risk_experiment.ipynb`

Extends the inner optimization to include a **real risk term** with diagonal covariance:

$$\min_w \; -\mu^\top w + \lambda\sqrt{w^\top D w} + \frac{\zeta}{2}\|w\|^2 \quad \text{s.t.} \quad \sum w_i = 1,\; 0 \le w_i \le \frac{10}{N}$$

The outer loss is the **Mean-SD utility**: $-(w^{*\top} r - \lambda\sqrt{w^{*\top} D w^*})$.

- Solved via **fixed-point iteration** on $P = \sqrt{w^\top D w}$, with per-stock effective $\zeta_i = \lambda \sigma_i^2 / P + \zeta$.
- Multi-$\lambda$ sweep traces out a **realized efficient frontier**, demonstrating investor heterogeneity.

**Key result** ($\lambda=1.0$): E2E Sharpe = 1.131, MSE Sharpe = 0.646. The frontier shows E2E consistently dominates MSE across risk aversion levels.

### 3. Return, Risk & Transaction Costs (Section 4.4, v5c)

**Notebook**: `Return_Risk_with_TC_experiment.ipynb`

Full alignment with **Eq.(21)** of the paper, adding smoothed transaction costs:

$$\min_w \; -\mu^\top w + \frac{\gamma}{2}\sum_i \sqrt{(w_i - w_{i,t-1})^2 + \varepsilon} + \lambda\sqrt{w^\top D w} + \frac{\zeta}{2}\|w\|^2$$

- The $\ell_1$ turnover term $|w_i - w_{i,t-1}|$ is smoothed as $\sqrt{(\cdot)^2 + \varepsilon}$ to maintain differentiability.
- **Inner and outer objectives share the same $\lambda$ and $\gamma$**, eliminating the loss-constraint mismatch that would otherwise arise.
- Sequential weight propagation ($w_{t-1}$) within each training epoch.

**Key result** ($\lambda=1.0, \gamma=0.003$): E2E Sharpe = 1.215, MSE-Opt Sharpe = 0.769, with 20% lower turnover.

## Methodology

### Model Architecture

- **Cross-sectional MLP**: 2-hidden-layer network (10 → 32 → 16 → 1) with LayerNorm, BatchNorm, Dropout — shared across all stocks in each cross-section (~1K parameters).
- **Factor selection**: 10 significant factors screened by IC t-stat > 2.0 from 100+ candidates.
- **Training**: MSE warm-up (10 epochs) → E2E fine-tuning (60 epochs), with IC auxiliary loss (5% weight) to provide gradient signal beyond free variables.

### Differentiable Optimization Layers

| Layer | Inner Problem | Solver | Gradient |
|-------|--------------|--------|----------|
| LP (Return-only) | $-\mu^\top w + \frac{\zeta}{2}\|w\|^2$ | `brentq` bisection | KKT on free set |
| SOCP (Return+Risk) | $-\mu^\top w + \lambda\sqrt{w^\top D w} + \frac{\zeta}{2}\|w\|^2$ | Fixed-point on P | KKT with rank-1 correction |
| SOCP+TC (Full Eq.21) | Above + $\frac{\gamma}{2}\text{TC}_\varepsilon(w, w_{t-1})$ | Outer TC linearization + inner fixed-point | KKT with TC Hessian |

### Rolling Window Evaluation

- **Universe**: ~252 CSI 300 stocks with ≥70% data coverage.
- **Period**: 2020-04 to 2025-09 (68 months).
- **Windows**: 10 rolling windows, 36-month training + 3-month test, step=3.
- **Transaction costs**: 30 bps single-side (0.3%) deducted at test time.

## Key Findings

| Setting | E2E Sharpe | Baseline Sharpe | E2E Ann Return | Baseline Ann Return |
|---------|-----------|----------------|---------------|-------------------|
| Return-only | 1.124 | 0.848 (MSE) | +33.0% | +24.9% |
| Return+Risk ($\lambda=1$) | 1.131 | 0.646 (MSE) | +27.3% | +15.2% |
| Return+Risk+TC ($\lambda=1, \gamma=0.003$) | 1.215 | 0.769 (MSE-Opt) | +29.3% | +18.0% |

E2E consistently outperforms the two-stage approach across all settings, with:
- Higher risk-adjusted returns (Sharpe improvements of +0.28 to +0.49)
- Lower portfolio turnover (20–27% reduction)
- Shallower maximum drawdowns

## Tech Stack

- **Python 3.x** / **PyTorch** — model training and autograd
- **NumPy** / **SciPy** (`brentq`) — optimization layer forward solve
- **Pandas** — data processing
- **Matplotlib** — visualization

## Reference

Wang, Y., Gao, H., Harvey, C. R., Liu, Y., & Tao, X. (2026). *Machine Learning Meets Markowitz*. SSRN.
