"""
================================================================================
PRACTITIONER'S GUIDE TO FACTOR MODELS — GRINOLD FRAMEWORK
================================================================================
A comprehensive Python implementation of the multi-factor risk model as
described by Richard Grinold in "A Practitioner's Guide to Factor Models."

Covers:
  1. Factor Exposure Estimation (cross-sectional regression)
  2. Factor Covariance Matrix Estimation
  3. Specific (Idiosyncratic) Risk Estimation
  4. Portfolio Risk Decomposition (systematic vs specific)
  5. Marginal Contribution to Risk (MCTR)
  6. Active Risk (Tracking Error) Analysis
  7. Risk Attribution by Factor
  8. Portfolio Expected Return via Factor Model
  9. Information Ratio & Transfer Coefficient
 10. Bias Statistics for Model Validation

Usage:
    python grinold_factor_model.py

The script generates synthetic data to demonstrate all metrics.
For real-world use, replace the data generation section with actual data.

Author: Generated for educational / practitioner use
================================================================================
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv, multi_dot
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)


# =============================================================================
# SECTION 1 — DATA STRUCTURES
# =============================================================================

@dataclass
class FactorModelData:
    """Container for all inputs to the factor model."""
    returns: pd.DataFrame           # T × N asset returns
    factor_exposures: pd.DataFrame  # N × K exposures (one snapshot or dict of dates)
    factor_names: list
    asset_names: list
    benchmark_weights: np.ndarray   # N × 1
    portfolio_weights: np.ndarray   # N × 1


@dataclass
class FactorModelResults:
    """Container for all estimated model outputs."""
    factor_returns: pd.DataFrame        # T × K
    factor_covariance: pd.DataFrame     # K × K
    specific_variance: pd.Series        # N × 1  (diagonal of Δ)
    specific_returns: pd.DataFrame      # T × N  residuals
    r_squared: pd.Series                # N × 1  per-asset R²
    asset_covariance: pd.DataFrame      # N × N  full covariance V = X F X' + Δ


# =============================================================================
# SECTION 2 — SYNTHETIC DATA GENERATION (replace with real data)
# =============================================================================

def generate_synthetic_data(
    n_assets: int = 50,
    n_periods: int = 120,       # monthly periods (10 years)
    n_factors: int = 5,
    seed: int = 42,
) -> FactorModelData:
    """
    Generate synthetic stock returns and factor exposures.

    Factors are stylised as:
        Market, Size, Value, Momentum, Volatility
    """
    np.random.seed(seed)

    factor_names = ["Market", "Size", "Value", "Momentum", "Volatility"][:n_factors]
    asset_names = [f"Stock_{i+1:03d}" for i in range(n_assets)]

    # --- True factor returns (monthly, in %) ---
    true_factor_means = np.array([0.8, 0.2, 0.3, 0.4, -0.1])[:n_factors] / 100
    true_factor_vols = np.array([4.5, 2.0, 2.0, 2.5, 2.0])[:n_factors] / 100

    # Factor correlation matrix
    rho = np.eye(n_factors)
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            rho[i, j] = rho[j, i] = np.random.uniform(-0.3, 0.3)
    # Ensure PSD
    eigvals = np.linalg.eigvalsh(rho)
    if eigvals.min() < 0:
        rho += (-eigvals.min() + 0.01) * np.eye(n_factors)
        d = np.sqrt(np.diag(rho))
        rho = rho / np.outer(d, d)

    factor_cov_true = np.outer(true_factor_vols, true_factor_vols) * rho
    factor_returns = np.random.multivariate_normal(true_factor_means, factor_cov_true, n_periods)

    # --- Factor exposures (cross-sectional, standardised) ---
    X = np.random.randn(n_assets, n_factors)
    X[:, 0] = np.random.uniform(0.7, 1.5, n_assets)   # Market beta
    # Standardise non-market exposures to zero mean, unit std
    for k in range(1, n_factors):
        X[:, k] = (X[:, k] - X[:, k].mean()) / X[:, k].std()

    # --- Specific (idiosyncratic) returns ---
    specific_vols = np.random.uniform(3.0, 10.0, n_assets) / 100  # monthly
    specific_returns = np.random.randn(n_periods, n_assets) * specific_vols

    # --- Total returns ---
    total_returns = factor_returns @ X.T + specific_returns

    # --- Portfolio & benchmark weights ---
    raw_w = np.random.dirichlet(np.ones(n_assets))
    portfolio_weights = raw_w
    benchmark_weights = np.ones(n_assets) / n_assets   # equal-weight benchmark

    returns_df = pd.DataFrame(total_returns, columns=asset_names,
                              index=pd.date_range("2015-01-31", periods=n_periods, freq="ME"))
    exposures_df = pd.DataFrame(X, index=asset_names, columns=factor_names)

    return FactorModelData(
        returns=returns_df,
        factor_exposures=exposures_df,
        factor_names=factor_names,
        asset_names=asset_names,
        benchmark_weights=benchmark_weights,
        portfolio_weights=portfolio_weights,
    )


# =============================================================================
# SECTION 3 — FACTOR RETURN ESTIMATION  (Cross-Sectional Regression)
# =============================================================================

def estimate_factor_returns_cross_sectional(
    returns: pd.DataFrame,
    exposures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Grinold's cross-sectional regression (GLS/OLS):
        r_t = X f_t + u_t

    For each period t, regress the N asset returns on the N × K exposure
    matrix to obtain the K × 1 vector of factor returns f_t.

    Returns
    -------
    factor_returns : pd.DataFrame  (T × K)
    residuals      : pd.DataFrame  (T × N)
    """
    X = exposures.values                       # N × K
    T = returns.shape[0]
    K = X.shape[1]
    N = X.shape[0]

    XtX_inv = inv(X.T @ X)                    # K × K

    factor_rets = np.empty((T, K))
    residuals = np.empty((T, N))

    for t in range(T):
        r_t = returns.iloc[t].values           # N × 1
        f_t = XtX_inv @ X.T @ r_t             # K × 1  (OLS)
        factor_rets[t] = f_t
        residuals[t] = r_t - X @ f_t

    factor_returns = pd.DataFrame(factor_rets, index=returns.index,
                                  columns=exposures.columns)
    residuals_df = pd.DataFrame(residuals, index=returns.index,
                                columns=returns.columns)
    return factor_returns, residuals_df


# =============================================================================
# SECTION 4 — COVARIANCE AND SPECIFIC RISK ESTIMATION
# =============================================================================

def estimate_factor_covariance(
    factor_returns: pd.DataFrame,
    half_life: Optional[int] = 36,
) -> pd.DataFrame:
    """
    Estimate the K × K factor covariance matrix F using exponentially
    weighted observations (Grinold recommends half-life ≈ 36 months).

    Parameters
    ----------
    factor_returns : T × K DataFrame
    half_life      : exponential weighting half-life in periods
    """
    T, K = factor_returns.shape
    if half_life:
        lam = 0.5 ** (1.0 / half_life)
        weights = np.array([lam ** (T - 1 - t) for t in range(T)])
        weights /= weights.sum()
    else:
        weights = np.ones(T) / T

    mu = (weights[:, None] * factor_returns.values).sum(axis=0)
    demeaned = factor_returns.values - mu
    F = (demeaned * weights[:, None]).T @ demeaned

    return pd.DataFrame(F, index=factor_returns.columns,
                        columns=factor_returns.columns)


def estimate_specific_variance(
    residuals: pd.DataFrame,
    half_life: Optional[int] = 36,
    newey_west_lags: int = 0,
) -> pd.Series:
    """
    Estimate the N × 1 vector of specific (idiosyncratic) variances.
    Uses exponential weighting consistent with the factor covariance.
    Optionally applies Newey-West adjustment for serial correlation.
    """
    T, N = residuals.shape

    if half_life:
        lam = 0.5 ** (1.0 / half_life)
        weights = np.array([lam ** (T - 1 - t) for t in range(T)])
        weights /= weights.sum()
    else:
        weights = np.ones(T) / T

    mu = (weights[:, None] * residuals.values).sum(axis=0)
    demeaned = residuals.values - mu
    spec_var = (demeaned ** 2 * weights[:, None]).sum(axis=0)

    # Newey-West adjustment for autocorrelation (optional)
    if newey_west_lags > 0:
        for lag in range(1, newey_west_lags + 1):
            bartlett = 1 - lag / (newey_west_lags + 1)
            gamma = (demeaned[lag:] * demeaned[:-lag] * weights[lag:, None]).sum(axis=0)
            spec_var += 2 * bartlett * gamma

    return pd.Series(spec_var, index=residuals.columns, name="specific_variance")


def compute_asset_covariance(
    exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific_var: pd.Series,
) -> pd.DataFrame:
    """
    Full N × N asset covariance matrix (Grinold Eq.):
        V = X F X' + Δ

    where X is N×K exposures, F is K×K factor covariance,
    and Δ is the N×N diagonal specific variance matrix.
    """
    X = exposures.values
    F = factor_cov.values
    Delta = np.diag(specific_var.values)
    V = X @ F @ X.T + Delta
    return pd.DataFrame(V, index=exposures.index, columns=exposures.index)


# =============================================================================
# SECTION 5 — BUILD THE FULL MODEL
# =============================================================================

def build_factor_model(data: FactorModelData, half_life: int = 36) -> FactorModelResults:
    """Run the full Grinold factor model estimation pipeline."""

    # Step 1: Cross-sectional regression for factor returns
    factor_returns, residuals = estimate_factor_returns_cross_sectional(
        data.returns, data.factor_exposures
    )

    # Step 2: Factor covariance
    factor_cov = estimate_factor_covariance(factor_returns, half_life=half_life)

    # Step 3: Specific variance
    specific_var = estimate_specific_variance(residuals, half_life=half_life)

    # Step 4: R² per asset
    total_var = data.returns.var()
    r_squared = 1 - specific_var / total_var
    r_squared = r_squared.clip(0, 1)

    # Step 5: Full covariance
    asset_cov = compute_asset_covariance(data.factor_exposures, factor_cov, specific_var)

    return FactorModelResults(
        factor_returns=factor_returns,
        factor_covariance=factor_cov,
        specific_variance=specific_var,
        specific_returns=residuals,
        r_squared=r_squared,
        asset_covariance=asset_cov,
    )


# =============================================================================
# SECTION 6 — PORTFOLIO-LEVEL ANALYTICS
# =============================================================================

class PortfolioAnalytics:
    """
    All Grinold-style portfolio risk and return analytics computed from
    the factor model.
    """

    def __init__(self, data: FactorModelData, model: FactorModelResults):
        self.data = data
        self.model = model

        self.w_p = data.portfolio_weights
        self.w_b = data.benchmark_weights
        self.w_a = self.w_p - self.w_b          # active weights

        self.X = data.factor_exposures.values    # N × K
        self.F = model.factor_covariance.values  # K × K
        self.Delta = np.diag(model.specific_variance.values)  # N × N
        self.V = model.asset_covariance.values   # N × N

    # -----------------------------------------------------------------
    # 6a. Portfolio Factor Exposures
    # -----------------------------------------------------------------
    def portfolio_exposures(self) -> pd.Series:
        """x_p = X' w_p  — K × 1 vector of portfolio factor exposures."""
        exp = self.X.T @ self.w_p
        return pd.Series(exp, index=self.data.factor_names, name="Portfolio Exposure")

    def benchmark_exposures(self) -> pd.Series:
        exp = self.X.T @ self.w_b
        return pd.Series(exp, index=self.data.factor_names, name="Benchmark Exposure")

    def active_exposures(self) -> pd.Series:
        """x_a = X' w_a  — active factor exposures (portfolio minus benchmark)."""
        exp = self.X.T @ self.w_a
        return pd.Series(exp, index=self.data.factor_names, name="Active Exposure")

    # -----------------------------------------------------------------
    # 6b. Portfolio Total Risk
    # -----------------------------------------------------------------
    def portfolio_variance(self) -> float:
        """σ²_p = w_p' V w_p"""
        return float(self.w_p @ self.V @ self.w_p)

    def portfolio_risk(self) -> float:
        """σ_p = sqrt(w_p' V w_p)  — total portfolio volatility."""
        return np.sqrt(self.portfolio_variance())

    # -----------------------------------------------------------------
    # 6c. Systematic vs Specific Risk Decomposition
    # -----------------------------------------------------------------
    def systematic_variance(self) -> float:
        """σ²_sys = w_p' (X F X') w_p"""
        x_p = self.X.T @ self.w_p
        return float(x_p @ self.F @ x_p)

    def specific_variance_portfolio(self) -> float:
        """σ²_spec = w_p' Δ w_p"""
        return float(self.w_p @ self.Delta @ self.w_p)

    def risk_decomposition(self) -> pd.Series:
        total = self.portfolio_variance()
        sys = self.systematic_variance()
        spec = self.specific_variance_portfolio()
        return pd.Series({
            "Total Variance": total,
            "Systematic Variance": sys,
            "Specific Variance": spec,
            "Total Risk (vol)": np.sqrt(total),
            "Systematic Risk (vol)": np.sqrt(sys),
            "Specific Risk (vol)": np.sqrt(spec),
            "% Systematic": sys / total * 100,
            "% Specific": spec / total * 100,
        })

    # -----------------------------------------------------------------
    # 6d. Factor Risk Contribution (Grinold decomposition)
    # -----------------------------------------------------------------
    def factor_risk_contribution(self) -> pd.DataFrame:
        """
        Decompose systematic variance into per-factor contributions.

        For factor k:
            contribution_k = x_p[k] * (F @ x_p)[k]
        
        These sum to the total systematic variance.
        """
        x_p = self.X.T @ self.w_p                 # K × 1
        Fx = self.F @ x_p                          # K × 1
        contributions = x_p * Fx
        sys_var = contributions.sum()

        df = pd.DataFrame({
            "Exposure": x_p,
            "Marginal Contribution": Fx,
            "Variance Contribution": contributions,
            "Risk Contribution": np.sign(contributions) * np.sqrt(np.abs(contributions)),
            "% of Systematic Var": contributions / sys_var * 100,
        }, index=self.data.factor_names)
        return df

    # -----------------------------------------------------------------
    # 6e. Active Risk (Tracking Error)
    # -----------------------------------------------------------------
    def active_variance(self) -> float:
        """ω² = w_a' V w_a"""
        return float(self.w_a @ self.V @ self.w_a)

    def tracking_error(self) -> float:
        """ω = sqrt(w_a' V w_a)"""
        return np.sqrt(self.active_variance())

    def active_systematic_variance(self) -> float:
        x_a = self.X.T @ self.w_a
        return float(x_a @ self.F @ x_a)

    def active_specific_variance(self) -> float:
        return float(self.w_a @ self.Delta @ self.w_a)

    def active_risk_decomposition(self) -> pd.Series:
        total = self.active_variance()
        sys = self.active_systematic_variance()
        spec = self.active_specific_variance()
        return pd.Series({
            "Active Variance": total,
            "Active Systematic Variance": sys,
            "Active Specific Variance": spec,
            "Tracking Error (vol)": np.sqrt(total),
            "Active Systematic Risk": np.sqrt(sys),
            "Active Specific Risk": np.sqrt(spec),
            "% Systematic": sys / total * 100 if total > 0 else 0,
            "% Specific": spec / total * 100 if total > 0 else 0,
        })

    # -----------------------------------------------------------------
    # 6f. Active Factor Risk Contribution
    # -----------------------------------------------------------------
    def active_factor_risk_contribution(self) -> pd.DataFrame:
        x_a = self.X.T @ self.w_a
        Fx = self.F @ x_a
        contributions = x_a * Fx
        sys_var = contributions.sum()
        pct = contributions / sys_var * 100 if sys_var != 0 else contributions * 0

        df = pd.DataFrame({
            "Active Exposure": x_a,
            "Marginal Contribution": Fx,
            "Variance Contribution": contributions,
            "% of Active Sys Var": pct,
        }, index=self.data.factor_names)
        return df

    # -----------------------------------------------------------------
    # 6g. Marginal Contribution to Risk (MCTR)
    # -----------------------------------------------------------------
    def mctr_total(self) -> pd.Series:
        """
        MCTR_i = (V w_p)_i / σ_p
        
        The change in portfolio risk for a marginal increase in weight i.
        """
        Vw = self.V @ self.w_p
        sigma = self.portfolio_risk()
        return pd.Series(Vw / sigma, index=self.data.asset_names, name="MCTR")

    def mctr_active(self) -> pd.Series:
        """MCTR_i (active) = (V w_a)_i / ω"""
        Vw = self.V @ self.w_a
        te = self.tracking_error()
        return pd.Series(Vw / te, index=self.data.asset_names, name="Active MCTR")

    def contribution_to_risk(self) -> pd.DataFrame:
        """
        CTR_i = w_i × MCTR_i
        Sums to total portfolio risk.
        """
        mctr = self.mctr_total()
        ctr = self.w_p * mctr
        sigma = self.portfolio_risk()
        df = pd.DataFrame({
            "Weight": self.w_p,
            "MCTR": mctr.values,
            "CTR": ctr.values,
            "% of Risk": ctr.values / sigma * 100,
        }, index=self.data.asset_names)
        return df

    # -----------------------------------------------------------------
    # 6h. Expected Return via Factor Model
    # -----------------------------------------------------------------
    def portfolio_expected_return(self, factor_premiums: Optional[np.ndarray] = None) -> float:
        """
        E[r_p] = x_p' μ_f

        where μ_f is the vector of expected factor premiums.
        If not provided, uses sample means of estimated factor returns.
        """
        if factor_premiums is None:
            factor_premiums = self.model.factor_returns.mean().values
        x_p = self.X.T @ self.w_p
        return float(x_p @ factor_premiums)

    def active_expected_return(self, factor_premiums: Optional[np.ndarray] = None) -> float:
        """E[r_a] = x_a' μ_f"""
        if factor_premiums is None:
            factor_premiums = self.model.factor_returns.mean().values
        x_a = self.X.T @ self.w_a
        return float(x_a @ factor_premiums)

    # -----------------------------------------------------------------
    # 6i. Information Ratio
    # -----------------------------------------------------------------
    def information_ratio(self, factor_premiums: Optional[np.ndarray] = None) -> float:
        """IR = E[r_a] / ω"""
        alpha = self.active_expected_return(factor_premiums)
        te = self.tracking_error()
        return alpha / te if te > 0 else 0.0

    # -----------------------------------------------------------------
    # 6j. Transfer Coefficient (TC)
    # -----------------------------------------------------------------
    def transfer_coefficient(self, alpha_scores: Optional[np.ndarray] = None) -> float:
        """
        TC = Cor(α, w_a * σ_i)

        Measures how efficiently alpha signals are transferred
        into active weights after accounting for constraints.

        If alpha_scores not provided, uses portfolio active expected
        return contribution as a proxy.
        """
        if alpha_scores is None:
            # Use factor-model-implied alphas
            factor_premiums = self.model.factor_returns.mean().values
            alpha_scores = self.X @ factor_premiums

        sigma_i = np.sqrt(np.diag(self.V))
        risk_adj_active = self.w_a * sigma_i
        tc, _ = stats.pearsonr(alpha_scores, risk_adj_active)
        return tc

    # -----------------------------------------------------------------
    # 6k. Proportion of Risk from Specific Bets (Diversification)
    # -----------------------------------------------------------------
    def effective_number_of_bets(self) -> float:
        """
        Herfindahl-based measure:
        EN = 1 / Σ (CTR_i / σ_p)²
        """
        sigma = self.portfolio_risk()
        mctr = self.mctr_total().values
        ctr = self.w_p * mctr
        ctr_pct = ctr / sigma
        return 1.0 / np.sum(ctr_pct ** 2)


# =============================================================================
# SECTION 7 — MODEL VALIDATION (BIAS STATISTICS)
# =============================================================================

def compute_bias_statistics(
    model: FactorModelResults,
    data: FactorModelData,
    n_test_portfolios: int = 100,
) -> pd.DataFrame:
    """
    Grinold's bias statistic (b):

        b = std(z_t) where z_t = r_p,t / σ_p (predicted)

    A well-calibrated model has b ≈ 1.0.
    Values >> 1 mean the model underestimates risk.
    Values << 1 mean the model overestimates risk.
    """
    V = model.asset_covariance.values
    N = data.returns.shape[1]
    results = []

    for i in range(n_test_portfolios):
        # Random portfolio
        w = np.random.dirichlet(np.ones(N))
        predicted_var = w @ V @ w
        predicted_std = np.sqrt(predicted_var)

        # Realised returns
        realised_rets = data.returns.values @ w
        z = realised_rets / predicted_std
        bias_stat = z.std()
        results.append({
            "Portfolio": i + 1,
            "Predicted Vol": predicted_std,
            "Realised Vol": realised_rets.std(),
            "Bias Statistic": bias_stat,
        })

    return pd.DataFrame(results)


# =============================================================================
# SECTION 8 — CORRELATION & EIGENVALUE ANALYSIS
# =============================================================================

def factor_correlation_matrix(factor_cov: pd.DataFrame) -> pd.DataFrame:
    """Convert factor covariance to correlation."""
    D_inv = np.diag(1.0 / np.sqrt(np.diag(factor_cov.values)))
    corr = D_inv @ factor_cov.values @ D_inv
    return pd.DataFrame(corr, index=factor_cov.index, columns=factor_cov.columns)


def eigenvalue_analysis(factor_cov: pd.DataFrame) -> pd.DataFrame:
    """Eigenvalue decomposition of the factor covariance matrix."""
    eigvals, eigvecs = np.linalg.eigh(factor_cov.values)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    explained = eigvals / eigvals.sum() * 100
    cumulative = np.cumsum(explained)

    df = pd.DataFrame({
        "Eigenvalue": eigvals,
        "% Variance Explained": explained,
        "Cumulative %": cumulative,
    }, index=[f"PC{i+1}" for i in range(len(eigvals))])
    return df


# =============================================================================
# SECTION 9 — REPORTING
# =============================================================================

def print_section(title: str, width: int = 72):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def annualise(monthly_val: float, is_variance: bool = False) -> float:
    if is_variance:
        return monthly_val * 12
    return monthly_val * np.sqrt(12)


def run_full_analysis():
    """Execute the full Grinold factor model pipeline and print all metrics."""

    # --- Generate data ---
    data = generate_synthetic_data(n_assets=50, n_periods=120, n_factors=5)

    # --- Build model ---
    model = build_factor_model(data, half_life=36)

    # --- Analytics ---
    analytics = PortfolioAnalytics(data, model)

    # =================================================================
    print_section("FACTOR MODEL SUMMARY")
    # =================================================================
    print(f"Universe : {len(data.asset_names)} assets")
    print(f"Periods  : {data.returns.shape[0]} months "
          f"({data.returns.index[0].date()} → {data.returns.index[-1].date()})")
    print(f"Factors  : {', '.join(data.factor_names)}")

    # =================================================================
    print_section("ESTIMATED FACTOR RETURNS (annualised)")
    # =================================================================
    fr = model.factor_returns
    ann_mean = fr.mean() * 12 * 100
    ann_vol = fr.std() * np.sqrt(12) * 100
    t_stats = fr.mean() / (fr.std() / np.sqrt(len(fr)))
    summary = pd.DataFrame({
        "Ann. Mean (%)": ann_mean,
        "Ann. Vol (%)": ann_vol,
        "t-stat": t_stats,
        "Sharpe": ann_mean / ann_vol,
    })
    print(summary.round(3).to_string())

    # =================================================================
    print_section("FACTOR COVARIANCE MATRIX (annualised, ×10⁴)")
    # =================================================================
    print((model.factor_covariance * 12 * 1e4).round(3).to_string())

    # =================================================================
    print_section("FACTOR CORRELATION MATRIX")
    # =================================================================
    print(factor_correlation_matrix(model.factor_covariance).round(3).to_string())

    # =================================================================
    print_section("EIGENVALUE ANALYSIS OF FACTOR COVARIANCE")
    # =================================================================
    print(eigenvalue_analysis(model.factor_covariance).round(3).to_string())

    # =================================================================
    print_section("SPECIFIC RISK (annualised, top/bottom 5 assets)")
    # =================================================================
    spec_risk_ann = np.sqrt(model.specific_variance * 12) * 100
    spec_risk_ann.name = "Specific Risk (% ann.)"
    print("\nHighest specific risk:")
    print(spec_risk_ann.nlargest(5).round(2).to_string())
    print("\nLowest specific risk:")
    print(spec_risk_ann.nsmallest(5).round(2).to_string())

    # =================================================================
    print_section("MODEL FIT — R² PER ASSET (summary)")
    # =================================================================
    r2 = model.r_squared
    print(f"Mean R²   : {r2.mean():.3f}")
    print(f"Median R² : {r2.median():.3f}")
    print(f"Min R²    : {r2.min():.3f}")
    print(f"Max R²    : {r2.max():.3f}")

    # =================================================================
    print_section("PORTFOLIO FACTOR EXPOSURES")
    # =================================================================
    exp_df = pd.DataFrame({
        "Portfolio": analytics.portfolio_exposures(),
        "Benchmark": analytics.benchmark_exposures(),
        "Active": analytics.active_exposures(),
    })
    print(exp_df.round(4).to_string())

    # =================================================================
    print_section("PORTFOLIO TOTAL RISK DECOMPOSITION (annualised)")
    # =================================================================
    rd = analytics.risk_decomposition()
    print(f"Total Risk       : {annualise(rd['Total Risk (vol)'])*100:.2f} %")
    print(f"  Systematic     : {annualise(rd['Systematic Risk (vol)'])*100:.2f} %  "
          f"({rd['% Systematic']:.1f}%)")
    print(f"  Specific       : {annualise(rd['Specific Risk (vol)'])*100:.2f} %  "
          f"({rd['% Specific']:.1f}%)")

    # =================================================================
    print_section("FACTOR RISK CONTRIBUTION (systematic, annualised)")
    # =================================================================
    frc = analytics.factor_risk_contribution()
    frc_ann = frc.copy()
    frc_ann["Variance Contribution"] *= 12
    frc_ann["Risk Contribution"] = np.sign(frc["Variance Contribution"]) * \
        np.sqrt(np.abs(frc["Variance Contribution"]) * 12)
    frc_ann["Marginal Contribution"] *= 12
    print(frc_ann[["Exposure", "Variance Contribution", "% of Systematic Var"]].round(4).to_string())

    # =================================================================
    print_section("ACTIVE RISK (TRACKING ERROR) DECOMPOSITION (annualised)")
    # =================================================================
    ard = analytics.active_risk_decomposition()
    print(f"Tracking Error   : {annualise(ard['Tracking Error (vol)'])*100:.2f} %")
    print(f"  Active Sys.    : {annualise(ard['Active Systematic Risk'])*100:.2f} %  "
          f"({ard['% Systematic']:.1f}%)")
    print(f"  Active Spec.   : {annualise(ard['Active Specific Risk'])*100:.2f} %  "
          f"({ard['% Specific']:.1f}%)")

    # =================================================================
    print_section("ACTIVE FACTOR RISK CONTRIBUTION")
    # =================================================================
    afrc = analytics.active_factor_risk_contribution()
    print(afrc.round(4).to_string())

    # =================================================================
    print_section("MARGINAL CONTRIBUTION TO RISK (top 10 by |MCTR|)")
    # =================================================================
    ctr_df = analytics.contribution_to_risk()
    top_ctr = ctr_df.reindex(ctr_df["MCTR"].abs().nlargest(10).index)
    print(top_ctr.round(5).to_string())

    # =================================================================
    print_section("EXPECTED RETURN & INFORMATION RATIO (annualised)")
    # =================================================================
    exp_ret_p = analytics.portfolio_expected_return() * 12 * 100
    exp_ret_a = analytics.active_expected_return() * 12 * 100
    ir = analytics.information_ratio()
    ir_ann = ir * np.sqrt(12)

    print(f"Expected Portfolio Return : {exp_ret_p:.3f} %")
    print(f"Expected Active Return    : {exp_ret_a:.3f} %")
    print(f"Tracking Error (ann.)     : {annualise(ard['Tracking Error (vol)'])*100:.2f} %")
    print(f"Information Ratio (ann.)  : {ir_ann:.3f}")

    # =================================================================
    print_section("TRANSFER COEFFICIENT")
    # =================================================================
    tc = analytics.transfer_coefficient()
    print(f"Transfer Coefficient : {tc:.3f}")
    print("  (1.0 = perfect transfer, 0.0 = no signal in active weights)")

    # =================================================================
    print_section("EFFECTIVE NUMBER OF BETS")
    # =================================================================
    enb = analytics.effective_number_of_bets()
    print(f"Effective N bets : {enb:.1f}  (out of {len(data.asset_names)} assets)")

    # =================================================================
    print_section("BIAS STATISTICS (model validation)")
    # =================================================================
    bias_df = compute_bias_statistics(model, data, n_test_portfolios=200)
    print(f"Mean bias statistic  : {bias_df['Bias Statistic'].mean():.3f}  (ideal ≈ 1.0)")
    print(f"Median               : {bias_df['Bias Statistic'].median():.3f}")
    print(f"Std                  : {bias_df['Bias Statistic'].std():.3f}")
    print(f"[5th, 95th] pctile   : [{bias_df['Bias Statistic'].quantile(0.05):.3f}, "
          f"{bias_df['Bias Statistic'].quantile(0.95):.3f}]")

    # =================================================================
    print_section("SUMMARY DASHBOARD")
    # =================================================================
    dash = pd.Series({
        "N Assets": len(data.asset_names),
        "N Factors": len(data.factor_names),
        "N Periods": data.returns.shape[0],
        "Portfolio Risk (ann %)": annualise(rd['Total Risk (vol)']) * 100,
        "Systematic Risk (ann %)": annualise(rd['Systematic Risk (vol)']) * 100,
        "Specific Risk (ann %)": annualise(rd['Specific Risk (vol)']) * 100,
        "Tracking Error (ann %)": annualise(ard['Tracking Error (vol)']) * 100,
        "Exp. Active Return (ann %)": exp_ret_a,
        "Information Ratio (ann)": ir_ann,
        "Transfer Coefficient": tc,
        "Effective N Bets": enb,
        "Mean Bias Statistic": bias_df['Bias Statistic'].mean(),
        "Mean R²": r2.mean(),
    })
    print(dash.round(4).to_string())

    return data, model, analytics


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    data, model, analytics = run_full_analysis()
