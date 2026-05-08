# End-to-End Markowitz Portfolio Optimization

Reproduction and empirical study of **"Machine Learning Meets Markowitz"** (Wang, Gao, Harvey, Liu & Tao, 2026), which proposes an end-to-end (E2E) framework that unifies return prediction and portfolio optimization into a single differentiable pipeline, applied to CSI 300 stocks.

## Paper Summary

The traditional "predict-then-optimize" approach trains a return forecast model using statistical objectives (e.g., MSE), then feeds the forecasts into a Markowitz optimizer. This two-stage separation is problematic because:

- **Heterogeneous preferences**: MSE produces one-size-fits-all forecasts, ignoring that different investors value forecast precision for different stocks.
- **Economic constraints**: Position limits and budget constraints are ignored during prediction.
- **Transaction costs**: Real-world frictions are absent from the statistical objective.

The E2E framework merges prediction and optimization by backpropagating portfolio-level objectives (return, risk, transaction cost) through the optimization layer directly into the prediction model, using the KKT implicit function theorem.

## Repository Structure

```
investment/
├── Data/                                    # Processed datasets
│   ├── hs300_factor_data_filled.parquet     #   Raw CSI 300 factor data (NA filled)
│   ├── factor_panel_v2_processed.parquet    #   Processed cross-sectional factor panel (68 months × 252 stocks)
│   └── 03v3_ic_summary.csv                 #   Factor IC statistics (screening 10 significant factors)
│
├── E2E_optimization_experiment/             # E2E Markowitz experiments (main results)
│   ├── Return-Only_experiment.ipynb         #   Sec 4.2: Pure return maximization (LP layer)
│   ├── Return_with_Risk_experiment.ipynb    #   Sec 4.4: Mean-SD utility (SOCP layer)
│   └── Return_Risk_with_TC_experiment.ipynb #   Sec 4.4: Full Eq.(21) with transaction costs (SOCP+TC layer)
│
├── SPO_experiment/                          # Smart Predict-then-Optimize baselines
│   ├── spo_rtn.ipynb                        #   SPO with true regret loss (LP oracle)
│   ├── spo_rtn_rsk.ipynb                    #   SPO+ with oracle surrogate loss (SOCP oracle)
│   └── spo_rtn_rsk_TC.ipynb                 #   SPO+ LS hybrid loss (SOCP+TC oracle)
│
├── PPP_experiment/                          # Parametric Portfolio Policy baseline
│   └── PPP_experiment_plus_rtn_rsk_TC.ipynb #   Brandt et al. linear parametric policy + MSE-Opt reference
│
├── The_paper/                               # Reference papers
│   ├── Machine_Learning_Meets_Markowitz.pdf #   Primary paper
│   ├── Smart_Predict_then_Optimize.pdf      #   SPO reference (Elmachtoub & Grigas)
│   ├── parametric-portfolio.pdf             #   PPP reference (Brandt, Santa-Clara, Valkanov)
│   └── Our_paper/                           #   Team's own manuscript
│
└── README.md
```

## Experiments

### 1. E2E Return-Only (Section 4.2)

**Notebook**: `E2E_optimization_experiment/Return-Only_experiment.ipynb`

Optimizes the pure return objective with a smoothed LP layer:

$$\min_w \; -\mu^\top w + \frac{\zeta}{2}\|w\|^2 \quad \text{s.t.} \quad \sum w_i = 1,\; 0 \le w_i \le \frac{10}{N}$$

- **Dual smoothing**: large $\zeta$ during training (rich gradients via more free variables), small $\zeta$ at test time (strict top-10% equal-weight portfolio).
- **Solver**: `brentq` bisection — O(N log(1/eps)), ~1500x faster than SLSQP.
- **Backward**: KKT implicit function theorem on the free variable set.

**Key result**: E2E Sharpe = 1.124 vs MSE Sharpe = 0.848.

### 2. E2E Return with Risk (Section 4.4)

**Notebook**: `E2E_optimization_experiment/Return_with_Risk_experiment.ipynb`

Extends the inner optimization to include a risk term with diagonal covariance:

$$\min_w \; -\mu^\top w + \lambda\sqrt{w^\top D w} + \frac{\zeta}{2}\|w\|^2 \quad \text{s.t.} \quad \sum w_i = 1,\; 0 \le w_i \le \frac{10}{N}$$

- Solved via **fixed-point iteration** on $P = \sqrt{w^\top D w}$.
- Multi-$\lambda$ sweep traces out a **realized efficient frontier**.

**Key result** ($\lambda=1.0$): E2E Sharpe = 1.131, MSE Sharpe = 0.646.

### 3. E2E Return, Risk & Transaction Costs (Section 4.4, Full Eq.21)

**Notebook**: `E2E_optimization_experiment/Return_Risk_with_TC_experiment.ipynb`

Full alignment with Eq.(21), adding smoothed transaction costs:

$$\min_w \; -\mu^\top w + \frac{\gamma}{2}\sum_i \sqrt{(w_i - w_{i,t-1})^2 + \varepsilon} + \lambda\sqrt{w^\top D w} + \frac{\zeta}{2}\|w\|^2$$

- The $\ell_1$ turnover term is smoothed as $\sqrt{(\cdot)^2 + \varepsilon}$ for differentiability.
- **Inner and outer objectives share the same $\lambda$ and $\gamma$**.
- Sequential weight propagation ($w_{t-1}$) within each training epoch.

**Key result** ($\lambda=1.0, \gamma=0.003$): E2E Sharpe = 1.215, MSE-Opt Sharpe = 0.769.

### 4. SPO Baselines

**Notebooks**: `SPO_experiment/`

Smart Predict-then-Optimize replaces the E2E training loss with SPO-style losses while keeping the same optimization oracle and rolling window design.

| Notebook | SPO Variant | Oracle | E2E Sharpe | SPO Sharpe | MSE Sharpe |
|----------|-------------|--------|-----------|-----------|-----------|
| `spo_rtn.ipynb` | True SPO regret | LP (return-only) | 1.124 | 1.062 | 0.798 |
| `spo_rtn_rsk.ipynb` | SPO+ oracle surrogate | SOCP (return+risk) | 1.131 | 0.946 | 0.647 |
| `spo_rtn_rsk_TC.ipynb` | SPO+ LS hybrid | SOCP+TC (full Eq.21) | 1.215 | 0.958 | 0.777 |

SPO improves over MSE but consistently underperforms E2E, especially as the optimization problem grows more complex.

### 5. PPP Baseline

**Notebook**: `PPP_experiment/PPP_experiment_plus_rtn_rsk_TC.ipynb`

Parametric Portfolio Policy (Brandt, Santa-Clara & Valkanov, 2009): a linear parametric approach where weights are directly parameterized as a function of firm characteristics via capped logistic mapping, optimized with scipy Powell method. Includes MSE-Opt alignment verification against the E2E notebook.

## Methodology

### Model Architecture

- **Cross-sectional MLP**: 2-hidden-layer network (10 → 32 → 16 → 1) with LayerNorm, BatchNorm, Dropout — shared across all stocks (~1K parameters).
- **Factor selection**: 10 significant factors screened by IC t-stat > 2.0 from 100+ candidates.
- **Training**: MSE warm-up (10 epochs) → E2E fine-tuning (60 epochs), with IC auxiliary loss (5% weight).

### Differentiable Optimization Layers

| Layer | Inner Problem | Solver | Gradient |
|-------|--------------|--------|----------|
| LP (Return-only) | $-\mu^\top w + \frac{\zeta}{2}\|w\|^2$ | `brentq` bisection | KKT on free set |
| SOCP (Return+Risk) | $-\mu^\top w + \lambda\sqrt{w^\top D w} + \frac{\zeta}{2}\|w\|^2$ | Fixed-point on P | KKT with rank-1 correction |
| SOCP+TC (Full Eq.21) | Above + $\frac{\gamma}{2}\text{TC}_\varepsilon(w, w_{t-1})$ | Outer TC linearization + inner fixed-point | KKT with TC Hessian |

### Rolling Window Evaluation

- **Universe**: ~252 CSI 300 stocks with >=70% data coverage.
- **Period**: 2020-04 to 2025-09 (68 months).
- **Windows**: 10 rolling windows, 36-month training + 3-month test, step=3.
- **Transaction costs**: 30 bps single-side (0.3%) deducted at test time.

## Key Findings

| Setting | E2E Sharpe | Baseline Sharpe | E2E Ann Return | Baseline Ann Return |
|---------|-----------|----------------|---------------|-------------------|
| Return-only | 1.124 | 0.848 (MSE) | +33.0% | +24.9% |
| Return+Risk ($\lambda=1$) | 1.131 | 0.646 (MSE) | +27.3% | +15.2% |
| Return+Risk+TC ($\lambda=1, \gamma=0.003$) | 1.215 | 0.769 (MSE-Opt) | +29.3% | +18.0% |

E2E consistently outperforms the two-stage approach across all settings:
- Higher risk-adjusted returns (Sharpe improvements of +0.28 to +0.49)
- Lower portfolio turnover (20-27% reduction)
- Shallower maximum drawdowns

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running Experiments

1. **Data**: The `Data/` directory contains all processed datasets needed to run the experiments.
2. **E2E experiments**: Open any notebook in `E2E_optimization_experiment/` and run all cells.
3. **SPO baselines**: Open any notebook in `SPO_experiment/` and run all cells.
4. **PPP baseline**: Open `PPP_experiment/PPP_experiment_plus_rtn_rsk_TC.ipynb` and run all cells.

All experiments are self-contained and use the same rolling window evaluation protocol.

## Reference

- Wang, Y., Gao, H., Harvey, C. R., Liu, Y., & Tao, X. (2026). *Machine Learning Meets Markowitz*. SSRN.
- Elmachtoub, A. N., & Grigas, P. (2022). *Smart "Predict, then Optimize"*. Management Science.
- Brandt, M. W., Santa-Clara, P., & Valkanov, R. (2009). *Parametric Portfolio Policies: Exploiting Characteristics in the Cross-Section of Equity Returns*. Review of Financial Studies.
