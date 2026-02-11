"""
Portfolio PnL Attribution & Risk Decomposition Along Orthogonalized Stock Baskets
==================================================================================

Given a long/short portfolio, this module measures performance, PnL attribution,
and risk exposure along thousands of long-only stock baskets, after orthogonalizing
each basket against its corresponding market index (e.g., S&P 500).

Workflow:
    1. Construct basket returns from constituent weights
    2. Orthogonalize each basket against its market index via OLS (keep residual)
    3. Estimate portfolio exposures to orthogonalized baskets via ridge regression
    4. Attribute PnL = exposure x orthogonalized basket return
    5. Decompose risk via MCTR/ACTR framework
    6. Present results with formatted tables and plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

ANNUAL_FACTOR = 252


# =============================================================================
# Core Functions
# =============================================================================

def orthogonalize_baskets(basket_ret, market_ret_series, basket_market_map):
    """
    Orthogonalize each basket's returns against its corresponding market index.

    For each basket b with market m_b, runs OLS:
        R_basket_b(t) = alpha_b + beta_b * R_market_m(t) + eps_b(t)
    and returns the residual eps_b(t) as the orthogonalized return.

    Parameters:
        basket_ret (pd.DataFrame): T x B basket returns
        market_ret_series (pd.Series, pd.DataFrame, or dict):
            Market returns keyed by market name.
            If a single Series, wraps into a dict using Series.name as key.
        basket_market_map (dict): basket_name -> market_index_name

    Returns:
        ortho_ret (pd.DataFrame): T x B orthogonalized basket returns (residuals)
        reg_stats (pd.DataFrame): B rows with columns [market, alpha_ann, beta, R2]
    """
    if isinstance(market_ret_series, pd.Series):
        market_ret_series = {market_ret_series.name: market_ret_series}
    elif isinstance(market_ret_series, pd.DataFrame):
        market_ret_series = {c: market_ret_series[c] for c in market_ret_series.columns}

    ortho_ret = pd.DataFrame(index=basket_ret.index, columns=basket_ret.columns, dtype=float)
    stats = []

    for b in basket_ret.columns:
        mkt_name = basket_market_map[b]
        mkt = market_ret_series[mkt_name].reindex(basket_ret.index)
        y = basket_ret[b].values
        X = np.column_stack([np.ones(len(mkt)), mkt.values])

        # OLS: y = X @ [alpha, beta] + eps
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha_b, beta_b = coeffs
        fitted = X @ coeffs
        eps = y - fitted

        ss_res = np.sum(eps ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        ortho_ret[b] = eps
        stats.append({'basket': b, 'market': mkt_name,
                      'alpha_ann': alpha_b * ANNUAL_FACTOR,
                      'beta': beta_b, 'R2': r2})

    reg_stats = pd.DataFrame(stats).set_index('basket')
    return ortho_ret, reg_stats


def estimate_basket_exposures(port_ret, ortho_ret, ridge_lambda=0.01):
    """
    Estimate portfolio exposures to orthogonalized baskets via ridge regression.

    Solves: R_port(t) = sum_b delta_b * eps_b(t) + eta(t)
    using ridge: delta = (X'X + lambda*I)^-1 X'y

    Parameters:
        port_ret (pd.Series): T portfolio returns
        ortho_ret (pd.DataFrame): T x B orthogonalized basket returns
        ridge_lambda (float): L2 regularization parameter.
            Larger values shrink exposures toward zero.

    Returns:
        exposures (pd.Series): B exposures (regression coefficients)
        r2 (float): in-sample R-squared
        residual (pd.Series): T unexplained returns
    """
    X = ortho_ret.values  # T x B
    y = port_ret.values   # T
    B = X.shape[1]

    XtX = X.T @ X
    Xty = X.T @ y
    coeffs = np.linalg.solve(XtX + ridge_lambda * np.eye(B), Xty)

    fitted = X @ coeffs
    resid = y - fitted
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    exposures = pd.Series(coeffs, index=ortho_ret.columns, name='exposure')
    residual = pd.Series(resid, index=port_ret.index, name='unexplained')
    return exposures, r2, residual


def compute_pnl_attribution(exposures, ortho_ret, port_ret):
    """
    Compute PnL attribution of portfolio returns to orthogonalized baskets.

    Daily PnL attributed to basket b: PnL_b(t) = delta_b * eps_b(t)

    Parameters:
        exposures (pd.Series): B exposures
        ortho_ret (pd.DataFrame): T x B orthogonalized basket returns
        port_ret (pd.Series): T portfolio returns

    Returns:
        daily_attr (pd.DataFrame): T x (B+1) daily PnL attribution (baskets + Unexplained)
        cum_attr (pd.DataFrame): T x (B+1) cumulative PnL attribution
        summary (pd.DataFrame): (B+1) rows with summary stats per source
    """
    daily_attr = ortho_ret.mul(exposures, axis=1)
    daily_attr['Unexplained'] = port_ret.values - daily_attr.sum(axis=1).values

    cum_attr = daily_attr.cumsum()

    total_pnl = daily_attr.sum()
    ann_return = daily_attr.mean() * ANNUAL_FACTOR
    ann_vol = daily_attr.std() * np.sqrt(ANNUAL_FACTOR)
    sharpe = ann_return / ann_vol.replace(0, np.nan)

    summary = pd.DataFrame({
        'Total PnL (cum ret)': total_pnl,
        'Ann. Return': ann_return,
        'Ann. Vol': ann_vol,
        'Sharpe': sharpe,
        'Pct of Total PnL': total_pnl / port_ret.sum() * 100
    })

    return daily_attr, cum_attr, summary


def compute_risk_decomposition(exposures, ortho_ret, annual_factor=252):
    """
    Decompose portfolio risk along orthogonalized baskets using MCTR/ACTR.

    Given exposures delta and covariance matrix Sigma of orthogonalized returns:
        sigma_p = sqrt(delta' Sigma delta)
        MCTR_b  = (Sigma delta)_b / sigma_p
        ACTR_b  = delta_b * MCTR_b
    ACTR sums to sigma_p.

    Parameters:
        exposures (pd.Series): B exposures
        ortho_ret (pd.DataFrame): T x B orthogonalized basket returns
        annual_factor (int): annualization factor (default 252)

    Returns:
        risk_df (pd.DataFrame): B rows with Exposure, MCTR, ACTR, Pct of Risk
        portfolio_vol (float): annualized portfolio vol from this model
    """
    cov_matrix = ortho_ret.cov() * annual_factor
    delta = exposures.values
    cov = cov_matrix.values

    port_var = delta @ cov @ delta
    port_vol = np.sqrt(port_var)

    mctr = (cov @ delta) / port_vol
    actr = delta * mctr
    pct_risk = actr / port_vol * 100

    risk_df = pd.DataFrame({
        'Exposure': exposures.values,
        'MCTR (ann)': mctr,
        'ACTR (ann)': actr,
        'Pct of Risk': pct_risk
    }, index=exposures.index)

    return risk_df, port_vol


# =============================================================================
# Visualization
# =============================================================================

def plot_cumulative_attribution(cum_attr, daily_attr, portfolio_returns, top_n=8):
    """
    Plot cumulative PnL attribution for the top contributing baskets.

    Parameters:
        cum_attr (pd.DataFrame): T x (B+1) cumulative attribution
        daily_attr (pd.DataFrame): T x (B+1) daily attribution
        portfolio_returns (pd.Series): T portfolio returns
        top_n (int): number of top baskets to show
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    basket_cols = [c for c in cum_attr.columns if c != 'Unexplained']
    top_baskets = daily_attr[basket_cols].sum().abs().nlargest(top_n).index.tolist()

    ax = axes[0]
    cum_attr[top_baskets].plot(ax=ax, linewidth=1.5)
    if 'Unexplained' in cum_attr.columns:
        cum_attr['Unexplained'].plot(ax=ax, linewidth=1.5, linestyle='--',
                                     color='grey', label='Unexplained')
    portfolio_returns.cumsum().plot(ax=ax, linewidth=2.5, color='black',
                                   label='Total Portfolio')
    ax.set_title(f'Cumulative PnL Attribution (Top {top_n} Baskets)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Return')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax2 = axes[1]
    rolling_attr = daily_attr[top_baskets].rolling(21).sum()
    rolling_attr.plot.area(ax=ax2, linewidth=0, alpha=0.7, stacked=True)
    ax2.set_title(f'Rolling 21-Day PnL Attribution (Top {top_n} Baskets)', fontsize=12)
    ax2.set_ylabel('21-Day Return')
    ax2.legend(loc='upper left', fontsize=7, ncol=2)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    plt.show()


def plot_risk_decomposition(risk_decomp, top_n=20):
    """
    Plot risk decomposition: bar chart of ACTR and pie chart of risk share.

    Parameters:
        risk_decomp (pd.DataFrame): output of compute_risk_decomposition
        top_n (int): number of top contributors to show
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top_risk = risk_decomp['ACTR (ann)'].abs().nlargest(top_n)
    colors = ['#d65f5f' if v < 0 else '#5fba7d'
              for v in risk_decomp.loc[top_risk.index, 'ACTR (ann)']]
    risk_decomp.loc[top_risk.index, 'ACTR (ann)'].plot.barh(ax=axes[0], color=colors)
    axes[0].set_title(f'Top {top_n} ACTR Contributors', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Annualized ACTR')
    axes[0].axvline(0, color='black', linewidth=0.5)
    axes[0].invert_yaxis()

    top10 = risk_decomp['ACTR (ann)'].abs().nlargest(10)
    other = risk_decomp['ACTR (ann)'].abs().sum() - top10.sum()
    pie_data = pd.concat([top10, pd.Series({'Other': other})])
    pie_data.plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90, fontsize=8)
    axes[1].set_title('Risk Share (|ACTR|)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(ortho_ret, baskets, title='Correlation of Orthogonalized Basket Returns'):
    """
    Plot correlation heatmap for selected orthogonalized basket returns.

    Parameters:
        ortho_ret (pd.DataFrame): T x B orthogonalized returns
        baskets (list): basket names to include
        title (str): plot title
    """
    corr_matrix = ortho_ret[baskets].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_orthogonalization_diagnostics(reg_stats):
    """
    Plot distribution of basket market betas and R-squared values.

    Parameters:
        reg_stats (pd.DataFrame): output of orthogonalize_baskets (second return)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(reg_stats['beta'], bins=40, edgecolor='white', alpha=0.8, color='steelblue')
    axes[0].axvline(reg_stats['beta'].mean(), color='red', linestyle='--',
                    label=f'Mean={reg_stats["beta"].mean():.2f}')
    axes[0].set_title('Distribution of Basket Market Betas', fontweight='bold')
    axes[0].set_xlabel('Beta to Market')
    axes[0].legend()

    axes[1].hist(reg_stats['R2'], bins=40, edgecolor='white', alpha=0.8, color='darkorange')
    axes[1].axvline(reg_stats['R2'].mean(), color='red', linestyle='--',
                    label=f'Mean={reg_stats["R2"].mean():.2f}')
    axes[1].set_title('Distribution of Market R² per Basket', fontweight='bold')
    axes[1].set_xlabel('R²')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# Dashboard
# =============================================================================

def build_dashboard(attr_summary, risk_decomp, regression_stats):
    """
    Build a combined attribution & risk dashboard DataFrame.

    Parameters:
        attr_summary (pd.DataFrame): output of compute_pnl_attribution (third return)
        risk_decomp (pd.DataFrame): output of compute_risk_decomposition (first return)
        regression_stats (pd.DataFrame): output of orthogonalize_baskets (second return)

    Returns:
        dashboard (pd.DataFrame): combined table sorted by |Cum PnL|
    """
    dashboard = attr_summary.join(risk_decomp[['MCTR (ann)', 'ACTR (ann)', 'Pct of Risk']])
    dashboard = dashboard.drop('Unexplained', errors='ignore')
    dashboard = dashboard.join(regression_stats[['beta', 'R2']])
    dashboard.columns = [
        'Cum PnL', 'Ann Return', 'Ann Vol', 'Sharpe',
        'PnL %', 'MCTR', 'ACTR', 'Risk %',
        'Mkt Beta', 'Mkt R²'
    ]
    dashboard = dashboard.reindex(
        dashboard['Cum PnL'].abs().sort_values(ascending=False).index
    )
    return dashboard


# =============================================================================
# Data Simulation (for demonstration / testing)
# =============================================================================

def simulate_data(n_stocks=200, t_days=504, n_baskets=50,
                  basket_size_range=(10, 40), seed=42):
    """
    Generate simulated stock returns, basket compositions, market returns,
    and a long/short portfolio for demonstration purposes.

    Parameters:
        n_stocks (int): number of stocks
        t_days (int): number of trading days
        n_baskets (int): number of baskets
        basket_size_range (tuple): (min, max) stocks per basket
        seed (int): random seed

    Returns:
        stock_returns (pd.DataFrame): T x N daily stock returns
        market_returns (pd.Series): T daily market returns
        basket_weights (dict): basket_name -> pd.Series of weights
        basket_market_map (dict): basket_name -> market index name
        portfolio_returns (pd.Series): T daily portfolio returns
        portfolio_weights (pd.Series): N portfolio weights
    """
    np.random.seed(seed)

    dates = pd.bdate_range('2023-01-02', periods=t_days, freq='B')
    stock_names = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    basket_names = [f'BASKET_{i:03d}' for i in range(n_baskets)]

    # Market
    market_returns = pd.Series(
        np.random.normal(0.08 / ANNUAL_FACTOR, 0.16 / np.sqrt(ANNUAL_FACTOR), t_days),
        index=dates, name='SP500'
    )

    # Stocks
    betas = np.random.uniform(0.5, 1.5, n_stocks)
    idio_vols = np.random.uniform(0.15, 0.45, n_stocks) / np.sqrt(ANNUAL_FACTOR)
    alphas = np.random.normal(0, 0.02 / ANNUAL_FACTOR, n_stocks)

    stock_returns = pd.DataFrame(index=dates, columns=stock_names, dtype=float)
    for i, s in enumerate(stock_names):
        stock_returns[s] = (
            alphas[i] + betas[i] * market_returns.values
            + np.random.normal(0, idio_vols[i], t_days)
        )

    # Baskets
    basket_weights = {}
    basket_market_map = {}
    for b in basket_names:
        size = np.random.randint(*basket_size_range)
        members = np.random.choice(stock_names, size=size, replace=False)
        basket_weights[b] = pd.Series(1.0 / size, index=members, name=b)
        basket_market_map[b] = 'SP500'

    # Long/short portfolio
    raw_pos = np.random.randn(n_stocks)
    raw_pos -= raw_pos.mean()
    portfolio_weights = pd.Series(raw_pos / np.abs(raw_pos).sum(),
                                  index=stock_names, name='portfolio')
    portfolio_returns = (stock_returns * portfolio_weights).sum(axis=1)
    portfolio_returns.name = 'Portfolio'

    return (stock_returns, market_returns, basket_weights,
            basket_market_map, portfolio_returns, portfolio_weights)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the full orthogonalized basket attribution pipeline on simulated data."""

    sns.set_theme(style='whitegrid', font_scale=1.1)
    pd.options.display.float_format = '{:,.4f}'.format

    # --- 1. Generate data ---
    print('=' * 70)
    print('1. Generating simulated data')
    print('=' * 70)
    (stock_returns, market_returns, basket_weights,
     basket_market_map, portfolio_returns, portfolio_weights) = simulate_data()

    # Compute basket returns
    basket_names = list(basket_weights.keys())
    basket_returns = pd.DataFrame(index=stock_returns.index,
                                  columns=basket_names, dtype=float)
    for b in basket_names:
        w = basket_weights[b]
        basket_returns[b] = stock_returns[w.index].values @ w.values

    print(f'  Stocks:    {stock_returns.shape[1]}')
    print(f'  Days:      {stock_returns.shape[0]}')
    print(f'  Baskets:   {len(basket_names)}')
    print(f'  Portfolio gross leverage: {portfolio_weights.abs().sum():.2f}')
    print(f'  Portfolio net exposure:   {portfolio_weights.sum():.4f}')
    print(f'  Portfolio ann. return:    {portfolio_returns.mean() * ANNUAL_FACTOR:.2%}')
    print(f'  Portfolio ann. vol:       {portfolio_returns.std() * np.sqrt(ANNUAL_FACTOR):.2%}')

    # --- 2. Orthogonalize ---
    print(f'\n{"=" * 70}')
    print('2. Orthogonalizing baskets against market index')
    print('=' * 70)
    ortho_basket_returns, regression_stats = orthogonalize_baskets(
        basket_returns, market_returns, basket_market_map
    )

    corr_with_market = ortho_basket_returns.corrwith(market_returns)
    print(f'  Correlation with market after orthogonalization:')
    print(f'    Mean: {corr_with_market.mean():.6f}  Max|corr|: {corr_with_market.abs().max():.6f}')
    print(f'\n  Regression stats (first 10):')
    print(regression_stats.head(10).to_string())

    # --- 3. Estimate exposures ---
    print(f'\n{"=" * 70}')
    print('3. Estimating portfolio exposures via ridge regression')
    print('=' * 70)
    exposures, model_r2, unexplained = estimate_basket_exposures(
        portfolio_returns, ortho_basket_returns, ridge_lambda=0.01
    )
    print(f'  Model R²: {model_r2:.4f}')
    print(f'\n  Top 10 exposures (by magnitude):')
    top_exp = exposures.abs().nlargest(10)
    print(exposures.loc[top_exp.index].to_string())

    # --- 4. PnL attribution ---
    print(f'\n{"=" * 70}')
    print('4. PnL attribution')
    print('=' * 70)
    daily_attr, cum_attr, attr_summary = compute_pnl_attribution(
        exposures, ortho_basket_returns, portfolio_returns
    )
    top_contributors = attr_summary['Total PnL (cum ret)'].abs().nlargest(16).index
    print(attr_summary.loc[top_contributors].to_string())

    # --- 5. Risk decomposition ---
    print(f'\n{"=" * 70}')
    print('5. Risk decomposition (MCTR / ACTR)')
    print('=' * 70)
    risk_decomp, model_port_vol = compute_risk_decomposition(
        exposures, ortho_basket_returns
    )
    print(f'  Portfolio vol (basket model): {model_port_vol:.4f} ({model_port_vol:.2%})')
    print(f'  Sum of ACTR:                  {risk_decomp["ACTR (ann)"].sum():.4f}')
    top_risk = risk_decomp['ACTR (ann)'].abs().nlargest(15).index
    print(f'\n  Top 15 risk contributors:')
    print(risk_decomp.loc[top_risk].to_string())

    # --- 6. Dashboard ---
    print(f'\n{"=" * 70}')
    print('6. Combined dashboard (top 20)')
    print('=' * 70)
    dashboard = build_dashboard(attr_summary, risk_decomp, regression_stats)
    print(dashboard.head(20).to_string())

    # --- 7. Plots ---
    print(f'\n{"=" * 70}')
    print('7. Generating plots...')
    print('=' * 70)

    plot_cumulative_attribution(cum_attr, daily_attr, portfolio_returns)
    plot_risk_decomposition(risk_decomp)
    plot_correlation_heatmap(ortho_basket_returns, dashboard.head(15).index.tolist())
    plot_orthogonalization_diagnostics(regression_stats)

    print('\nDone.')
    return dashboard


if __name__ == '__main__':
    dashboard = main()
