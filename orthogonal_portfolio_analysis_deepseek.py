"""
Orthogonal Portfolio Analysis with Market-Neutral Decomposition
================================================================

This module provides a complete framework for analyzing portfolio exposures
to thematic stock baskets after orthogonalizing each basket against a market
index. It downloads real market data via yfinance and produces both static
(matplotlib) and interactive (plotly) visualizations.

Workflow:
    1. Download historical price data for portfolio, baskets, and market index
    2. Orthogonalize each basket's returns against the market via OLS regression
    3. Compute orthogonal performance metrics (IR, correlation, beta, risk)
    4. Decompose risk via covariance-based MCTR/ACTR framework
    5. Generate visualizations and reports

Classes:
    MarketOrthogonalizer          - OLS-based market orthogonalization engine
    OrthogonalPortfolioAnalyzer   - End-to-end analysis pipeline
    OrthogonalResultsVisualizer   - Static and interactive visualizations
    RiskAttributionAnalyzer       - Risk contribution decomposition

Functions:
    main()                        - Run full analysis with example data
    create_orthogonal_report()    - Export results to Excel
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS, add_constant
import statsmodels.api as sm

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================
# 1. ORTHOGONALIZATION ENGINE
# ============================================

class MarketOrthogonalizer:
    """
    Orthogonalize basket returns with respect to a market index via OLS.

    For each basket b, runs:
        R_basket(t) = alpha + beta * R_market(t) + epsilon(t)

    The residual epsilon(t) is the market-neutral (orthogonalized) component.

    Parameters
    ----------
    market_ticker : str
        Ticker for market index (default: S&P 500)

    Attributes
    ----------
    beta_coefficients : dict
        {basket_name: beta} from OLS regressions
    r_squared_values : dict
        {basket_name: R²} from OLS regressions
    orthogonalized_returns : dict
        {basket_name: dict} with orthogonal returns, residuals, model params
    """

    def __init__(self, market_ticker='^GSPC'):
        self.market_ticker = market_ticker
        self.market_returns = None
        self.beta_coefficients = {}
        self.r_squared_values = {}
        self.orthogonalized_returns = {}

    def calculate_market_returns(self, price_data):
        """
        Calculate market returns from price data.

        Parameters
        ----------
        price_data : pd.DataFrame
            Price data with market ticker as one of the columns

        Returns
        -------
        pd.Series
            Daily market returns
        """
        if self.market_ticker in price_data.columns:
            market_prices = price_data[self.market_ticker]
            self.market_returns = market_prices.pct_change().dropna()
        else:
            print(f"Market ticker {self.market_ticker} not in price data")
            market_data = yf.download(self.market_ticker,
                                     start=price_data.index[0],
                                     end=price_data.index[-1])
            self.market_returns = market_data['Adj Close'].pct_change().dropna()
        return self.market_returns

    def orthogonalize_basket(self, basket_returns, market_returns, basket_name):
        """
        Orthogonalize a single basket's returns against the market.

        Parameters
        ----------
        basket_returns : pd.Series
            Daily returns of the basket
        market_returns : pd.Series
            Daily returns of the market index
        basket_name : str
            Name identifier for the basket

        Returns
        -------
        orthogonal : pd.Series
            Market-neutral component (residual + alpha)
        beta : float
            Beta coefficient from OLS regression
        alpha : float
            Intercept (persistent alpha)
        residuals : pd.Series
            Pure residual component (no alpha)
        """
        # Align dates
        aligned_data = pd.concat([basket_returns, market_returns], axis=1).dropna()
        basket_aligned = aligned_data.iloc[:, 0]
        market_aligned = aligned_data.iloc[:, 1]

        # OLS regression: basket = alpha + beta * market + epsilon
        X = sm.add_constant(market_aligned)
        model = OLS(basket_aligned, X).fit()

        alpha = model.params[0]
        beta = model.params[1]
        self.beta_coefficients[basket_name] = beta
        self.r_squared_values[basket_name] = model.rsquared

        # Orthogonal components
        predicted = model.predict(X)
        orthogonal = basket_aligned - predicted + alpha  # Keep alpha as separate component
        residuals = basket_aligned - predicted            # Pure residual

        self.orthogonalized_returns[basket_name] = {
            'orthogonal_total': orthogonal,
            'residuals': residuals,
            'market_component': predicted - alpha,
            'alpha_component': pd.Series(alpha, index=basket_aligned.index),
            'beta': beta,
            'alpha': alpha,
            'r_squared': model.rsquared,
            'model': model
        }

        return orthogonal, beta, alpha, residuals

    def orthogonalize_all_baskets(self, basket_returns_dict, market_returns):
        """
        Orthogonalize all baskets against the market.

        Parameters
        ----------
        basket_returns_dict : dict
            {basket_name: pd.Series} of daily basket returns
        market_returns : pd.Series
            Daily market returns

        Returns
        -------
        dict
            {basket_name: dict} with orthogonal returns, beta, alpha, R²
        """
        results = {}
        for basket_name, basket_return in basket_returns_dict.items():
            try:
                orthogonal, beta, alpha, residuals = self.orthogonalize_basket(
                    basket_return, market_returns, basket_name
                )
                results[basket_name] = {
                    'orthogonal_returns': orthogonal,
                    'residual_returns': residuals,
                    'beta': beta,
                    'alpha': alpha,
                    'r_squared': self.r_squared_values[basket_name]
                }
            except Exception as e:
                print(f"Error orthogonalizing {basket_name}: {e}")
                continue
        return results


# ============================================
# 2. ENHANCED PORTFOLIO ANALYZER WITH ORTHOGONALIZATION
# ============================================

class OrthogonalPortfolioAnalyzer:
    """
    End-to-end pipeline for orthogonal portfolio analysis.

    Downloads price data, computes returns, orthogonalizes baskets against a
    market index, and calculates comprehensive performance and risk metrics.

    Parameters
    ----------
    portfolio_weights : dict
        {ticker: weight} with positive for long, negative for short
    basket_definitions : dict
        {basket_name: {ticker: weight}} defining each basket's composition
    market_ticker : str
        Market index ticker for orthogonalization (default: '^GSPC')
    start_date : str
        Start date for historical data (default: '2020-01-01')
    """

    def __init__(self, portfolio_weights, basket_definitions,
                 market_ticker='^GSPC', start_date='2020-01-01'):
        self.portfolio_weights = portfolio_weights
        self.basket_definitions = basket_definitions
        self.market_ticker = market_ticker
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')

        # Collect all unique tickers including market
        all_tickers = set(portfolio_weights.keys())
        all_tickers.add(market_ticker)
        for basket in basket_definitions.values():
            all_tickers.update(basket.keys())
        self.all_tickers = list(all_tickers)

        # Data storage
        self.price_data = None
        self.returns_data = None
        self.market_returns = None
        self.portfolio_returns = None
        self.basket_returns = None

        # Orthogonalization
        self.orthogonalizer = MarketOrthogonalizer(market_ticker)
        self.orthogonal_results = None
        self.analysis_results = None

    def download_price_data(self):
        """Download historical price data for all tickers via yfinance."""
        print(f"Downloading price data for {len(self.all_tickers)} tickers...")

        chunk_size = 100
        price_chunks = []

        for i in range(0, len(self.all_tickers), chunk_size):
            chunk_tickers = self.all_tickers[i:i+chunk_size]
            try:
                chunk_data = yf.download(
                    chunk_tickers,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False
                )['Adj Close']
                price_chunks.append(chunk_data)
            except Exception as e:
                print(f"Error downloading chunk {i}: {e}")
                continue

        if price_chunks:
            self.price_data = pd.concat(price_chunks, axis=1)
            self.price_data.columns = [col.upper() for col in self.price_data.columns]
            print(f"Downloaded data shape: {self.price_data.shape}")
        else:
            raise ValueError("No price data downloaded")

    def calculate_returns(self):
        """Calculate daily returns for all assets, portfolio, and baskets."""
        self.returns_data = self.price_data.pct_change().dropna()

        # Market returns
        if self.market_ticker in self.returns_data.columns:
            self.market_returns = self.returns_data[self.market_ticker]
        else:
            print(f"Market ticker {self.market_ticker} not found, downloading separately")
            market_data = yf.download(self.market_ticker,
                                     start=self.start_date,
                                     end=self.end_date)
            self.market_returns = market_data['Adj Close'].pct_change().dropna()

        # Portfolio returns (normalized by sum of absolute weights)
        portfolio_weights_series = pd.Series(self.portfolio_weights)
        portfolio_tickers = portfolio_weights_series.index.intersection(self.returns_data.columns)

        if len(portfolio_tickers) < len(self.portfolio_weights):
            missing = set(self.portfolio_weights.keys()) - set(portfolio_tickers)
            print(f"Warning: Missing data for portfolio tickers: {missing}")

        portfolio_weights_aligned = portfolio_weights_series[portfolio_tickers]
        portfolio_weights_aligned = portfolio_weights_aligned / portfolio_weights_aligned.abs().sum()
        self.portfolio_returns = self.returns_data[portfolio_tickers].dot(portfolio_weights_aligned)

        # Basket returns
        self.basket_returns = {}
        for basket_name, basket_weights in self.basket_definitions.items():
            basket_weights_series = pd.Series(basket_weights)
            basket_tickers = basket_weights_series.index.intersection(self.returns_data.columns)

            if len(basket_tickers) < len(basket_weights):
                missing = set(basket_weights.keys()) - set(basket_tickers)
                print(f"Warning: Missing data for basket '{basket_name}' tickers: {missing}")

            if len(basket_tickers) > 0:
                basket_weights_aligned = basket_weights_series[basket_tickers]
                basket_weights_aligned = basket_weights_aligned / basket_weights_aligned.sum()
                self.basket_returns[basket_name] = self.returns_data[basket_tickers].dot(basket_weights_aligned)

        print(f"Calculated returns for {len(self.basket_returns)} baskets")

    def orthogonalize_and_analyze(self):
        """
        Perform orthogonalization and compute comprehensive metrics.

        Returns
        -------
        pd.DataFrame
            Analysis results with one row per basket containing:
            - Market model statistics (beta, alpha, R²)
            - Orthogonal component statistics (correlation, beta)
            - Risk metrics (volatility, tracking error, max drawdown)
            - Return metrics (annualized returns, Sharpe ratios)
            - Variance decomposition
        """
        # Orthogonalize all baskets
        self.orthogonalizer.calculate_market_returns(self.price_data)
        self.orthogonal_results = self.orthogonalizer.orthogonalize_all_baskets(
            self.basket_returns, self.market_returns
        )

        # Also orthogonalize portfolio returns
        portfolio_orthogonal, portfolio_beta, portfolio_alpha, _ = \
            self.orthogonalizer.orthogonalize_basket(
                self.portfolio_returns, self.market_returns, 'PORTFOLIO'
            )

        # Calculate comprehensive metrics per basket
        results = []
        for basket_name, basket_data in self.orthogonal_results.items():
            basket_ortho = basket_data['orthogonal_returns']
            basket_residuals = basket_data['residual_returns']
            basket_beta = basket_data['beta']
            basket_alpha = basket_data['alpha']
            basket_r2 = basket_data['r_squared']

            # Align all series
            aligned_data = pd.concat([
                portfolio_orthogonal.rename('portfolio_ortho'),
                basket_ortho.rename('basket_ortho'),
                basket_residuals.rename('basket_residuals'),
                self.market_returns.rename('market')
            ], axis=1).dropna()

            portfolio_ortho = aligned_data['portfolio_ortho']
            basket_ortho_aligned = aligned_data['basket_ortho']
            basket_residuals_aligned = aligned_data['basket_residuals']
            market_aligned = aligned_data['market']

            # Active orthogonal returns
            active_ortho = portfolio_ortho - basket_ortho_aligned

            # Correlation matrix
            corr_matrix = aligned_data[
                ['portfolio_ortho', 'basket_ortho', 'basket_residuals', 'market']
            ].corr()

            # Orthogonal beta (portfolio_ortho vs basket_ortho)
            if basket_ortho_aligned.std() > 0:
                ortho_beta = (np.cov(portfolio_ortho, basket_ortho_aligned)[0, 1]
                              / np.var(basket_ortho_aligned))
            else:
                ortho_beta = np.nan

            # Residual beta (portfolio_ortho vs basket_residuals)
            if basket_residuals_aligned.std() > 0:
                residual_beta = (np.cov(portfolio_ortho, basket_residuals_aligned)[0, 1]
                                 / np.var(basket_residuals_aligned))
            else:
                residual_beta = np.nan

            metrics = {
                'basket_name': basket_name,
                'num_constituents': len(self.basket_definitions[basket_name]),

                # Market model statistics
                'basket_beta_to_market': basket_beta,
                'basket_alpha_annual': basket_alpha * 252,
                'basket_r_squared': basket_r2,
                'portfolio_beta_to_market': portfolio_beta,
                'portfolio_alpha_annual': portfolio_alpha * 252,

                # Orthogonal component statistics
                'orthogonal_correlation': corr_matrix.loc['portfolio_ortho', 'basket_ortho'],
                'residual_correlation': corr_matrix.loc['portfolio_ortho', 'basket_residuals'],
                'market_correlation': corr_matrix.loc['portfolio_ortho', 'market'],

                # Beta metrics
                'orthogonal_beta': ortho_beta,
                'residual_beta': residual_beta,

                # Risk metrics (annualized)
                'portfolio_ortho_vol': portfolio_ortho.std() * np.sqrt(252),
                'basket_ortho_vol': basket_ortho_aligned.std() * np.sqrt(252),
                'basket_residual_vol': basket_residuals_aligned.std() * np.sqrt(252),
                'market_vol': market_aligned.std() * np.sqrt(252),

                # Return metrics (annualized)
                'portfolio_ortho_return': portfolio_ortho.mean() * 252,
                'basket_ortho_return': basket_ortho_aligned.mean() * 252,
                'basket_residual_return': basket_residuals_aligned.mean() * 252,
                'market_return': market_aligned.mean() * 252,

                # Tracking error metrics
                'orthogonal_tracking_error': active_ortho.std() * np.sqrt(252),
                'orthogonal_information_ratio': (
                    (active_ortho.mean() * 252) / (active_ortho.std() * np.sqrt(252))
                    if active_ortho.std() > 0 else 0
                ),

                # Sharpe ratios
                'portfolio_ortho_sharpe': (
                    (portfolio_ortho.mean() * 252) / (portfolio_ortho.std() * np.sqrt(252))
                    if portfolio_ortho.std() > 0 else 0
                ),
                'basket_ortho_sharpe': (
                    (basket_ortho_aligned.mean() * 252) / (basket_ortho_aligned.std() * np.sqrt(252))
                    if basket_ortho_aligned.std() > 0 else 0
                ),

                # Variance decomposition
                'market_explained_variance': basket_r2,
                'idiosyncratic_variance': 1 - basket_r2,

                # Drawdowns
                'portfolio_ortho_max_dd': self.calculate_max_drawdown(portfolio_ortho),
                'basket_ortho_max_dd': self.calculate_max_drawdown(basket_ortho_aligned),
                'basket_residual_max_dd': self.calculate_max_drawdown(basket_residuals_aligned)
            }

            results.append(metrics)

        self.analysis_results = pd.DataFrame(results)
        return self.analysis_results

    @staticmethod
    def calculate_max_drawdown(returns):
        """
        Calculate maximum drawdown from a return series.

        Parameters
        ----------
        returns : pd.Series
            Daily returns

        Returns
        -------
        float
            Maximum drawdown (negative value)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline: download, compute, orthogonalize.

        Returns
        -------
        pd.DataFrame
            Full analysis results
        """
        self.download_price_data()
        self.calculate_returns()
        results = self.orthogonalize_and_analyze()
        return results


# ============================================
# 3. ADVANCED VISUALIZATION FOR ORTHOGONAL ANALYSIS
# ============================================

class OrthogonalResultsVisualizer:
    """
    Visualization suite for orthogonal portfolio analysis results.

    Produces static matplotlib plots and interactive Plotly dashboards.

    Parameters
    ----------
    analyzer : OrthogonalPortfolioAnalyzer
        Completed analyzer with analysis_results populated
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.results = analyzer.analysis_results

    def create_summary_report(self):
        """
        Create a formatted summary report sorted by orthogonal IR.

        Returns
        -------
        pd.DataFrame
            Formatted summary with key metrics
        """
        summary = self.results.copy()
        summary = summary.sort_values('orthogonal_information_ratio', ascending=False)

        key_columns = [
            'basket_name', 'num_constituents',
            'orthogonal_information_ratio', 'orthogonal_correlation',
            'basket_beta_to_market', 'basket_r_squared',
            'portfolio_ortho_return', 'basket_ortho_return',
            'portfolio_ortho_vol', 'basket_ortho_vol',
            'orthogonal_beta', 'residual_beta',
            'portfolio_ortho_max_dd', 'basket_ortho_max_dd'
        ]

        formatted_summary = summary[key_columns].copy()

        format_dict = {
            'orthogonal_information_ratio': '{:.2f}',
            'orthogonal_correlation': '{:.3f}',
            'basket_beta_to_market': '{:.3f}',
            'basket_r_squared': '{:.1%}',
            'portfolio_ortho_return': '{:.1%}',
            'basket_ortho_return': '{:.1%}',
            'portfolio_ortho_vol': '{:.1%}',
            'basket_ortho_vol': '{:.1%}',
            'orthogonal_beta': '{:.3f}',
            'residual_beta': '{:.3f}',
            'portfolio_ortho_max_dd': '{:.1%}',
            'basket_ortho_max_dd': '{:.1%}'
        }

        for col, fmt in format_dict.items():
            if col in formatted_summary.columns:
                formatted_summary[col] = formatted_summary[col].apply(lambda x: fmt.format(x))

        return formatted_summary

    def plot_variance_decomposition(self):
        """
        Plot variance decomposition: R² distribution and beta distribution.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # R² distribution
        axes[0].hist(self.results['basket_r_squared'], bins=30, alpha=0.7,
                     edgecolor='black', color='skyblue')
        axes[0].axvline(x=self.results['basket_r_squared'].mean(), color='red',
                       linestyle='--', linewidth=2,
                       label=f'Mean: {self.results["basket_r_squared"].mean():.1%}')
        axes[0].set_xlabel('Market Explained Variance (R²)')
        axes[0].set_ylabel('Number of Baskets')
        axes[0].set_title('Distribution of Market Exposure')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Beta distribution
        axes[1].hist(self.results['basket_beta_to_market'], bins=30, alpha=0.7,
                     edgecolor='black', color='lightcoral')
        axes[1].axvline(x=1, color='black', linestyle='-', linewidth=1.5,
                       label='Market Beta = 1')
        axes[1].axvline(x=self.results['basket_beta_to_market'].mean(), color='red',
                       linestyle='--', linewidth=2,
                       label=f'Mean: {self.results["basket_beta_to_market"].mean():.2f}')
        axes[1].set_xlabel('Beta to Market')
        axes[1].set_ylabel('Number of Baskets')
        axes[1].set_title('Distribution of Market Betas')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_orthogonal_performance(self, top_n=20):
        """
        Plot performance metrics for top baskets by orthogonal IR.

        Parameters
        ----------
        top_n : int
            Number of top baskets to display

        Returns
        -------
        matplotlib.figure.Figure
        """
        top_baskets = self.results.nlargest(top_n, 'orthogonal_information_ratio')

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # 1. Orthogonal Information Ratio bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        axes[0].barh(range(top_n), top_baskets['orthogonal_information_ratio'].values,
                     color=colors, edgecolor='black')
        axes[0].set_yticks(range(top_n))
        axes[0].set_yticklabels(top_baskets['basket_name'].values)
        axes[0].set_xlabel('Orthogonal Information Ratio')
        axes[0].set_title(f'Top {top_n} Baskets by Orthogonal IR')
        axes[0].axvline(x=0, color='red', linestyle='-', alpha=0.5)

        # 2. Orthogonal Correlation vs Beta to Market
        scatter = axes[1].scatter(top_baskets['basket_beta_to_market'],
                                 top_baskets['orthogonal_correlation'],
                                 c=top_baskets['orthogonal_information_ratio'],
                                 cmap='RdYlGn', s=100, alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Beta to Market')
        axes[1].set_ylabel('Orthogonal Correlation')
        axes[1].set_title('Orthogonal Correlation vs Market Beta')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='red', linestyle='-', alpha=0.5)
        axes[1].axvline(x=0, color='red', linestyle='-', alpha=0.5)
        plt.colorbar(scatter, ax=axes[1], label='Orthogonal IR')

        # 3. Risk-Return Scatter (Orthogonal)
        scatter2 = axes[2].scatter(top_baskets['basket_ortho_vol'],
                                  top_baskets['basket_ortho_return'],
                                  c=top_baskets['basket_r_squared'],
                                  cmap='plasma', s=100, alpha=0.8, edgecolor='black')
        axes[2].set_xlabel('Orthogonal Volatility (Annualized)')
        axes[2].set_ylabel('Orthogonal Return (Annualized)')
        axes[2].set_title('Risk-Return Tradeoff (Orthogonal Component)')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='red', linestyle='-', alpha=0.5)
        axes[2].axvline(x=top_baskets['basket_ortho_vol'].mean(),
                       color='blue', linestyle='--', alpha=0.7,
                       label=f'Mean Vol: {top_baskets["basket_ortho_vol"].mean():.1%}')
        axes[2].legend()
        plt.colorbar(scatter2, ax=axes[2], label='R² to Market')

        # 4. Market vs Idiosyncratic Volatility (stacked bar)
        x = range(len(top_baskets))
        width = 0.35
        market_vol_component = top_baskets['basket_ortho_vol'] * np.sqrt(top_baskets['basket_r_squared'])
        idiosyncratic_vol = top_baskets['basket_ortho_vol'] * np.sqrt(1 - top_baskets['basket_r_squared'])

        axes[3].bar(x, market_vol_component, width, label='Market Component',
                   color='lightcoral', edgecolor='black')
        axes[3].bar(x, idiosyncratic_vol, width, bottom=market_vol_component,
                   label='Idiosyncratic Component', color='skyblue', edgecolor='black')
        axes[3].set_xlabel('Basket')
        axes[3].set_ylabel('Volatility Contribution')
        axes[3].set_title('Volatility Decomposition (Market vs Idiosyncratic)')
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(top_baskets['basket_name'].values, rotation=45, ha='right')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_correlation_structure(self):
        """
        Plot correlation heatmap between orthogonal components of top baskets.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        top_baskets = self.results.nlargest(15, 'orthogonal_information_ratio')['basket_name'].tolist()

        ortho_returns_dict = {}
        for basket in top_baskets:
            if basket in self.analyzer.orthogonal_results:
                ortho_returns_dict[basket] = self.analyzer.orthogonal_results[basket]['orthogonal_returns']

        if ortho_returns_dict:
            ortho_returns_df = pd.DataFrame(ortho_returns_dict)
            correlation_matrix = ortho_returns_df.corr()

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                       ax=ax)
            ax.set_title('Correlation Matrix of Orthogonal Basket Components', fontsize=14)
            plt.tight_layout()
            return fig

        return None

    def create_interactive_orthogonal_dashboard(self):
        """
        Create interactive Plotly dashboard for orthogonal analysis.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        top_30 = self.results.nlargest(30, 'orthogonal_information_ratio')

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Orthogonal IR vs Market Beta',
                          'Risk-Return Orthogonal Space',
                          'Correlation vs R²',
                          'Volatility Decomposition',
                          'Top Orthogonal IR Baskets',
                          'Market Exposure Distribution'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'histogram'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.15
        )

        # 1. Orthogonal IR vs Market Beta
        fig.add_trace(
            go.Scatter(
                x=top_30['basket_beta_to_market'],
                y=top_30['orthogonal_information_ratio'],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=top_30['orthogonal_correlation'],
                    colorscale='RdYlBu',
                    showscale=True,
                    colorbar=dict(title="Correlation", x=0.45, y=0.9)
                ),
                text=top_30['basket_name'],
                textposition="top center",
                hoverinfo='text+x+y',
                hovertext=[
                    f"{name}<br>Beta: {beta:.2f}<br>IR: {ir:.2f}<br>Corr: {corr:.2f}"
                    for name, beta, ir, corr in zip(
                        top_30['basket_name'],
                        top_30['basket_beta_to_market'],
                        top_30['orthogonal_information_ratio'],
                        top_30['orthogonal_correlation']
                    )
                ]
            ),
            row=1, col=1
        )

        # 2. Risk-Return Orthogonal Space
        fig.add_trace(
            go.Scatter(
                x=top_30['basket_ortho_vol'],
                y=top_30['basket_ortho_return'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=top_30['basket_r_squared'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="R²", x=1.02, y=0.9)
                ),
                text=top_30['basket_name'],
                hoverinfo='text+x+y',
                hovertext=[
                    f"{name}<br>Vol: {vol:.1%}<br>Return: {ret:.1%}<br>R²: {r2:.1%}"
                    for name, vol, ret, r2 in zip(
                        top_30['basket_name'],
                        top_30['basket_ortho_vol'],
                        top_30['basket_ortho_return'],
                        top_30['basket_r_squared']
                    )
                ]
            ),
            row=1, col=2
        )

        # 3. Correlation vs R²
        fig.add_trace(
            go.Scatter(
                x=top_30['basket_r_squared'],
                y=top_30['orthogonal_correlation'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=top_30['orthogonal_information_ratio'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Orthogonal IR", x=0.45, y=0.4)
                ),
                text=top_30['basket_name'],
                hoverinfo='text+x+y'
            ),
            row=2, col=1
        )

        # 4. Volatility Decomposition (top 10)
        top_10 = top_30.head(10)
        fig.add_trace(
            go.Bar(
                name='Market Component',
                x=top_10['basket_name'],
                y=top_10['basket_ortho_vol'] * np.sqrt(top_10['basket_r_squared']),
                marker_color='indianred'
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(
                name='Idiosyncratic Component',
                x=top_10['basket_name'],
                y=top_10['basket_ortho_vol'] * np.sqrt(1 - top_10['basket_r_squared']),
                marker_color='lightblue'
            ),
            row=2, col=2
        )

        # 5. Top Orthogonal IR Baskets
        fig.add_trace(
            go.Bar(
                x=top_10['orthogonal_information_ratio'],
                y=top_10['basket_name'],
                orientation='h',
                marker_color='lightgreen',
                text=top_10['orthogonal_information_ratio'].round(2),
                textposition='auto'
            ),
            row=3, col=1
        )

        # 6. Market Exposure Distribution
        fig.add_trace(
            go.Histogram(
                x=self.results['basket_beta_to_market'],
                nbinsx=30,
                marker_color='lightcoral',
                opacity=0.7
            ),
            row=3, col=2
        )

        # Layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Orthogonal Basket Analysis Dashboard",
            barmode='stack'
        )

        fig.update_xaxes(title_text="Beta to Market", row=1, col=1)
        fig.update_yaxes(title_text="Orthogonal Information Ratio", row=1, col=1)
        fig.update_xaxes(title_text="Orthogonal Volatility", row=1, col=2)
        fig.update_yaxes(title_text="Orthogonal Return", row=1, col=2)
        fig.update_xaxes(title_text="R² to Market", row=2, col=1)
        fig.update_yaxes(title_text="Orthogonal Correlation", row=2, col=1)
        fig.update_xaxes(title_text="Basket", row=2, col=2)
        fig.update_yaxes(title_text="Volatility Contribution", row=2, col=2)
        fig.update_xaxes(title_text="Orthogonal Information Ratio", row=3, col=1)
        fig.update_xaxes(title_text="Beta to Market", row=3, col=2)
        fig.update_yaxes(title_text="Frequency", row=3, col=2)

        return fig

    def plot_orthogonal_returns_ts(self, basket_names, window=20):
        """
        Plot time series of orthogonal returns for selected baskets.

        Parameters
        ----------
        basket_names : list
            Basket names to plot (up to 4)
        window : int
            Rolling window for volatility calculation

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for idx, basket_name in enumerate(basket_names[:4]):
            if basket_name in self.analyzer.orthogonal_results:
                ortho_data = self.analyzer.orthogonal_results[basket_name]
                ortho_returns = ortho_data['orthogonal_returns']
                residuals = ortho_data['residual_returns']

                # Cumulative returns
                cum_ortho = (1 + ortho_returns).cumprod() - 1
                cum_residuals = (1 + residuals).cumprod() - 1

                ir_val = self.results[
                    self.results["basket_name"] == basket_name
                ]["orthogonal_information_ratio"].values[0]

                axes[idx].plot(cum_ortho.index, cum_ortho.values,
                             label=f'Orthogonal Total (IR: {ir_val:.2f})',
                             linewidth=2)
                axes[idx].plot(cum_residuals.index, cum_residuals.values,
                             label='Pure Residuals', linewidth=1, alpha=0.7, linestyle='--')

                # Rolling volatility on twin axis
                rolling_vol = ortho_returns.rolling(window=window).std() * np.sqrt(252)
                axes_twin = axes[idx].twinx()
                axes_twin.plot(rolling_vol.index, rolling_vol.values,
                             color='red', alpha=0.5, label=f'{window}-day Rolling Vol')

                axes[idx].set_title(
                    f'{basket_name}\n'
                    f'Beta: {ortho_data["beta"]:.2f}, R²: {ortho_data["r_squared"]:.1%}'
                )
                axes[idx].set_xlabel('Date')
                axes[idx].set_ylabel('Cumulative Return')
                axes_twin.set_ylabel('Annualized Volatility', color='red')
                axes[idx].legend(loc='upper left')
                axes_twin.legend(loc='upper right')
                axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# ============================================
# 4. RISK ATTRIBUTION AND EXPOSURE ANALYSIS
# ============================================

class RiskAttributionAnalyzer:
    """
    Risk contribution analysis for orthogonalized basket components.

    Computes marginal and percent risk contributions, as well as
    diversification benefits from portfolio-basket covariance structure.

    Parameters
    ----------
    analyzer : OrthogonalPortfolioAnalyzer
        Completed analyzer with orthogonal results
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.results = analyzer.analysis_results

    def calculate_risk_contributions(self):
        """
        Calculate risk contributions from orthogonal components.

        Returns
        -------
        pd.DataFrame
            Risk decomposition per basket with marginal and percent contributions
        """
        risk_contributions = []

        for _, row in self.results.iterrows():
            basket_name = row['basket_name']

            if basket_name in self.analyzer.orthogonal_results:
                ortho_returns = self.analyzer.orthogonal_results[basket_name]['orthogonal_returns']
                portfolio_ortho = self.analyzer.orthogonalizer.orthogonalized_returns[
                    'PORTFOLIO'
                ]['orthogonal_total']

                aligned = pd.concat([portfolio_ortho, ortho_returns], axis=1).dropna()
                portfolio_aligned = aligned.iloc[:, 0]
                basket_aligned = aligned.iloc[:, 1]

                cov_matrix = np.cov(portfolio_aligned, basket_aligned)
                portfolio_var = cov_matrix[0, 0]
                basket_var = cov_matrix[1, 1]
                covariance = cov_matrix[0, 1]

                marginal_contribution = covariance / np.sqrt(portfolio_var) if portfolio_var > 0 else 0
                percent_contribution = covariance / portfolio_var if portfolio_var > 0 else 0

                risk_contributions.append({
                    'basket_name': basket_name,
                    'portfolio_variance': portfolio_var * 252,
                    'basket_variance': basket_var * 252,
                    'covariance': covariance * 252,
                    'marginal_risk_contribution': marginal_contribution * np.sqrt(252),
                    'percent_risk_contribution': percent_contribution,
                    'diversification_benefit': (
                        (portfolio_var - basket_var) / portfolio_var if portfolio_var > 0 else 0
                    )
                })

        return pd.DataFrame(risk_contributions)

    def plot_risk_contribution_analysis(self):
        """
        Plot risk contribution analysis: marginal, percent, diversification.

        Returns
        -------
        tuple
            (matplotlib.figure.Figure, pd.DataFrame of risk contributions)
        """
        risk_df = self.calculate_risk_contributions()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Marginal Risk Contribution
        top_20_risk = risk_df.nlargest(20, 'marginal_risk_contribution')
        axes[0, 0].barh(range(len(top_20_risk)),
                        top_20_risk['marginal_risk_contribution'].values,
                        color=plt.cm.RdYlGn(np.linspace(0, 1, len(top_20_risk))))
        axes[0, 0].set_yticks(range(len(top_20_risk)))
        axes[0, 0].set_yticklabels(top_20_risk['basket_name'].values)
        axes[0, 0].set_xlabel('Marginal Risk Contribution (Annualized)')
        axes[0, 0].set_title('Top 20 Baskets by Marginal Risk Contribution')
        axes[0, 0].axvline(x=0, color='red', linestyle='-', alpha=0.5)

        # 2. Percent Risk Contribution
        top_20_pct = risk_df.nlargest(20, 'percent_risk_contribution')
        axes[0, 1].barh(range(len(top_20_pct)),
                        top_20_pct['percent_risk_contribution'].values,
                        color=plt.cm.plasma(np.linspace(0, 1, len(top_20_pct))))
        axes[0, 1].set_yticks(range(len(top_20_pct)))
        axes[0, 1].set_yticklabels(top_20_pct['basket_name'].values)
        axes[0, 1].set_xlabel('Percent Risk Contribution')
        axes[0, 1].set_title('Top 20 Baskets by Percent Risk Contribution')
        axes[0, 1].axvline(x=0, color='red', linestyle='-', alpha=0.5)

        # 3. Diversification Benefit
        top_20_div = risk_df.nlargest(20, 'diversification_benefit')
        axes[1, 0].barh(range(len(top_20_div)),
                        top_20_div['diversification_benefit'].values,
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_20_div))))
        axes[1, 0].set_yticks(range(len(top_20_div)))
        axes[1, 0].set_yticklabels(top_20_div['basket_name'].values)
        axes[1, 0].set_xlabel('Diversification Benefit')
        axes[1, 0].set_title('Top 20 Baskets by Diversification Benefit')

        # 4. Variance vs Percent Risk Contribution
        scatter = axes[1, 1].scatter(risk_df['basket_variance'],
                                     risk_df['percent_risk_contribution'],
                                     c=risk_df['diversification_benefit'],
                                     cmap='coolwarm', s=50, alpha=0.8)
        axes[1, 1].set_xlabel('Basket Variance (Annualized)')
        axes[1, 1].set_ylabel('Percent Risk Contribution')
        axes[1, 1].set_title('Risk Contribution vs Basket Variance')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='Diversification Benefit')

        plt.tight_layout()
        return fig, risk_df


# ============================================
# 5. MAIN EXECUTION
# ============================================

def main():
    """
    Run the full orthogonal portfolio analysis with example data.

    Creates a long/short portfolio, defines thematic baskets, generates
    additional random basket variations, and runs the complete pipeline.

    Returns
    -------
    tuple
        (analyzer, results DataFrame, risk DataFrame)
    """
    # --- Example portfolio (long/short) ---
    portfolio_weights = {
        'AAPL': 0.15,
        'MSFT': 0.12,
        'GOOGL': 0.10,
        'AMZN': -0.08,
        'TSLA': 0.05,
        'JPM': -0.06,
        'JNJ': 0.04,
        'WMT': 0.03,
        'XOM': -0.05,
        'BAC': -0.04,
        'NVDA': 0.08,
        'META': 0.07,
        'UNH': -0.03,
        'HD': 0.02,
        'MA': 0.03,
    }

    # --- Thematic basket definitions ---
    basket_definitions = {
        # Technology baskets
        'Big_Tech': {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.20, 'AMZN': 0.15, 'META': 0.15},
        'Cloud_Computing': {'MSFT': 0.40, 'AMZN': 0.30, 'GOOGL': 0.20, 'ORCL': 0.10},
        'Semiconductors': {'NVDA': 0.40, 'AMD': 0.25, 'INTC': 0.20, 'TSM': 0.15},

        # Financial baskets
        'Large_Banks': {'JPM': 0.30, 'BAC': 0.25, 'C': 0.20, 'WFC': 0.15, 'GS': 0.10},
        'FinTech': {'PYPL': 0.40, 'SQ': 0.30, 'V': 0.20, 'MA': 0.10},

        # Consumer baskets
        'E_Commerce': {'AMZN': 0.40, 'SHOP': 0.30, 'ETSY': 0.20, 'BABA': 0.10},
        'Retail': {'WMT': 0.30, 'TGT': 0.25, 'COST': 0.25, 'HD': 0.20},

        # Healthcare baskets
        'Pharma': {'JNJ': 0.30, 'PFE': 0.25, 'MRK': 0.25, 'ABT': 0.20},
        'Biotech': {'AMGN': 0.35, 'GILD': 0.25, 'BIIB': 0.20, 'REGN': 0.20},

        # Energy baskets
        'Oil_Gas': {'XOM': 0.30, 'CVX': 0.25, 'COP': 0.20, 'SLB': 0.15, 'EOG': 0.10},
        'Renewable': {'NEE': 0.40, 'ENPH': 0.25, 'FSLR': 0.20, 'SEDG': 0.15},

        # Automotive baskets
        'EV_Makers': {'TSLA': 0.60, 'RIVN': 0.20, 'LCID': 0.10, 'F': 0.10},
        'Auto_Manufacturers': {'F': 0.40, 'GM': 0.40, 'TM': 0.20},

        # Broad sector proxies
        'XLK_Tech': {
            'AAPL': 0.20, 'MSFT': 0.20, 'NVDA': 0.15, 'AVGO': 0.10, 'CSCO': 0.10,
            'ORCL': 0.10, 'ACN': 0.05, 'IBM': 0.05, 'QCOM': 0.05
        },
        'XLF_Financial': {
            'JPM': 0.15, 'BAC': 0.15, 'WFC': 0.10, 'C': 0.10, 'GS': 0.10,
            'MS': 0.10, 'BLK': 0.05, 'SCHW': 0.05, 'AXP': 0.05, 'SPGI': 0.05
        },
    }

    # --- Generate additional baskets with randomized weight variations ---
    additional_baskets = {}
    sectors = ['Tech', 'Financial', 'Healthcare', 'Consumer', 'Industrial', 'Energy']

    for i in range(50):
        sector = sectors[i % len(sectors)]
        base_basket = list(basket_definitions.keys())[i % len(basket_definitions)]
        new_name = f"{sector}_Basket_{i+1:03d}"

        if base_basket in basket_definitions:
            base_weights = basket_definitions[base_basket]
            noise = np.random.normal(0, 0.1, len(base_weights))
            new_weights = {
                ticker: max(0.01, weight + noise[idx])
                for idx, (ticker, weight) in enumerate(base_weights.items())
            }
            total = sum(new_weights.values())
            new_weights = {k: v / total for k, v in new_weights.items()}
            additional_baskets[new_name] = new_weights

    all_baskets = {**basket_definitions, **additional_baskets}

    # --- Run analysis ---
    print("=" * 60)
    print("ORTHOGONAL BASKET ANALYSIS")
    print("=" * 60)
    print(f"Portfolio has {len(portfolio_weights)} positions")
    print(f"Analyzing against {len(all_baskets)} thematic baskets")
    print(f"Market index: ^GSPC (S&P 500)")
    print("=" * 60)

    analyzer = OrthogonalPortfolioAnalyzer(
        portfolio_weights,
        all_baskets,
        market_ticker='^GSPC',
        start_date='2021-01-01'
    )

    print("\nRunning orthogonal analysis...")
    results = analyzer.run_complete_analysis()

    # --- Visualizations ---
    visualizer = OrthogonalResultsVisualizer(analyzer)

    # Summary report
    print("\n" + "=" * 60)
    print("TOP 10 BASKETS BY ORTHOGONAL INFORMATION RATIO:")
    print("=" * 60)
    summary = visualizer.create_summary_report()
    print(summary.head(10).to_string())

    # Statistics
    print("\n" + "=" * 60)
    print("ORTHOGONAL ANALYSIS STATISTICS:")
    print("=" * 60)

    stats_data = {
        'Metric': [
            'Mean Orthogonal Information Ratio',
            'Median Orthogonal IR',
            'Mean Orthogonal Correlation',
            'Mean Beta to Market',
            'Mean R² to Market',
            'Portfolio Beta to Market',
            'Portfolio Alpha (Annual)',
            'Baskets with Negative Orthogonal Beta',
            'Baskets with Negative Market Beta'
        ],
        'Value': [
            f"{results['orthogonal_information_ratio'].mean():.3f}",
            f"{results['orthogonal_information_ratio'].median():.3f}",
            f"{results['orthogonal_correlation'].mean():.3f}",
            f"{results['basket_beta_to_market'].mean():.3f}",
            f"{results['basket_r_squared'].mean():.1%}",
            f"{analyzer.orthogonalizer.beta_coefficients['PORTFOLIO']:.3f}",
            f"{analyzer.orthogonalizer.orthogonalized_returns['PORTFOLIO']['alpha'] * 252:.2%}",
            f"{(results['orthogonal_beta'] < 0).sum()} "
            f"({(results['orthogonal_beta'] < 0).sum() / len(results) * 100:.1f}%)",
            f"{(results['basket_beta_to_market'] < 0).sum()} "
            f"({(results['basket_beta_to_market'] < 0).sum() / len(results) * 100:.1f}%)"
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))

    # Generate and save plots
    print("\nGenerating visualizations...")

    fig1 = visualizer.plot_variance_decomposition()
    plt.savefig('variance_decomposition.png', dpi=150, bbox_inches='tight')

    fig2 = visualizer.plot_orthogonal_performance(top_n=20)
    plt.savefig('orthogonal_performance.png', dpi=150, bbox_inches='tight')

    fig3 = visualizer.plot_correlation_structure()
    if fig3:
        plt.savefig('orthogonal_correlation_matrix.png', dpi=150, bbox_inches='tight')

    fig4 = visualizer.create_interactive_orthogonal_dashboard()
    fig4.write_html("orthogonal_analysis_dashboard.html")

    top_baskets = results.nlargest(4, 'orthogonal_information_ratio')['basket_name'].tolist()
    fig5 = visualizer.plot_orthogonal_returns_ts(top_baskets)
    plt.savefig('orthogonal_returns_timeseries.png', dpi=150, bbox_inches='tight')

    # Risk attribution
    print("\n" + "=" * 60)
    print("RISK ATTRIBUTION ANALYSIS:")
    print("=" * 60)

    risk_analyzer = RiskAttributionAnalyzer(analyzer)
    fig6, risk_df = risk_analyzer.plot_risk_contribution_analysis()
    plt.savefig('risk_attribution.png', dpi=150, bbox_inches='tight')

    print("\nTop 10 Baskets by Marginal Risk Contribution:")
    print(risk_df.nlargest(10, 'marginal_risk_contribution')[
        ['basket_name', 'marginal_risk_contribution', 'percent_risk_contribution']
    ].to_string())

    # Save results to CSV
    print("\nSaving detailed results...")
    results.to_csv('orthogonal_basket_analysis.csv', index=False)
    risk_df.to_csv('risk_attribution_analysis.csv', index=False)

    model_params = []
    for basket_name, data in analyzer.orthogonalizer.orthogonalized_returns.items():
        if basket_name != 'PORTFOLIO':
            model_params.append({
                'basket_name': basket_name,
                'beta': data['beta'],
                'alpha_annual': data['alpha'] * 252,
                'r_squared': data['r_squared'],
                't_stat_beta': (
                    data['model'].tvalues[1] if len(data['model'].tvalues) > 1 else np.nan
                ),
                'p_value_beta': (
                    data['model'].pvalues[1] if len(data['model'].pvalues) > 1 else np.nan
                )
            })
    pd.DataFrame(model_params).to_csv('regression_parameters.csv', index=False)

    # Key insights
    print("\n" + "=" * 60)
    print("KEY ORTHOGONAL INSIGHTS:")
    print("=" * 60)

    market_neutral = results.loc[results['basket_beta_to_market'].abs().argsort()[:5]]
    print("\n1. Most Market-Neutral Baskets (Lowest |Beta|):")
    for _, row in market_neutral.iterrows():
        print(f"   - {row['basket_name']}: Beta={row['basket_beta_to_market']:.3f}, "
              f"R²={row['basket_r_squared']:.1%}")

    high_ir = results.nlargest(3, 'orthogonal_information_ratio')
    print("\n2. Highest Orthogonal Information Ratio:")
    for _, row in high_ir.iterrows():
        print(f"   - {row['basket_name']}: IR={row['orthogonal_information_ratio']:.2f}, "
              f"Corr={row['orthogonal_correlation']:.3f}, Beta={row['orthogonal_beta']:.3f}")

    portfolio_beta = analyzer.orthogonalizer.beta_coefficients['PORTFOLIO']
    portfolio_alpha = analyzer.orthogonalizer.orthogonalized_returns['PORTFOLIO']['alpha'] * 252
    print(f"\n3. Portfolio Market Exposure:")
    print(f"   - Beta to market: {portfolio_beta:.3f}")
    print(f"   - Annual alpha: {portfolio_alpha:.2%}")

    print(f"\n4. Average Basket Risk Decomposition:")
    print(f"   - Market explained variance: {results['basket_r_squared'].mean():.1%}")
    print(f"   - Idiosyncratic variance: {1 - results['basket_r_squared'].mean():.1%}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Generated files:")
    print("  1. orthogonal_basket_analysis.csv")
    print("  2. risk_attribution_analysis.csv")
    print("  3. regression_parameters.csv")
    print("  4. variance_decomposition.png")
    print("  5. orthogonal_performance.png")
    print("  6. orthogonal_correlation_matrix.png")
    print("  7. orthogonal_analysis_dashboard.html")
    print("  8. orthogonal_returns_timeseries.png")
    print("  9. risk_attribution.png")

    return analyzer, results, risk_df


# ============================================
# 6. UTILITY FUNCTIONS
# ============================================

def create_orthogonal_report(analyzer, results, output_path='orthogonal_analysis_report.xlsx'):
    """
    Create comprehensive Excel report with multiple sheets.

    Parameters
    ----------
    analyzer : OrthogonalPortfolioAnalyzer
        Completed analyzer
    results : pd.DataFrame
        Analysis results from orthogonalize_and_analyze()
    output_path : str
        Output Excel file path
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary = results.sort_values('orthogonal_information_ratio', ascending=False)
        summary.to_excel(writer, sheet_name='Summary', index=False)

        market_exp = summary[
            ['basket_name', 'basket_beta_to_market', 'basket_r_squared', 'basket_alpha_annual']
        ].sort_values('basket_beta_to_market')
        market_exp.to_excel(writer, sheet_name='Market_Exposure', index=False)

        ortho_perf = summary[
            ['basket_name', 'orthogonal_information_ratio', 'orthogonal_correlation',
             'orthogonal_beta', 'portfolio_ortho_return', 'basket_ortho_return']
        ].sort_values('orthogonal_information_ratio', ascending=False)
        ortho_perf.to_excel(writer, sheet_name='Orthogonal_Performance', index=False)

        risk_metrics = summary[
            ['basket_name', 'portfolio_ortho_vol', 'basket_ortho_vol',
             'basket_residual_vol', 'orthogonal_tracking_error',
             'portfolio_ortho_max_dd', 'basket_ortho_max_dd']
        ]
        risk_metrics.to_excel(writer, sheet_name='Risk_Metrics', index=False)

        corr_analysis = summary[
            ['basket_name', 'orthogonal_correlation', 'residual_correlation', 'market_correlation']
        ]
        corr_analysis.to_excel(writer, sheet_name='Correlations', index=False)

        portfolio_stats = pd.DataFrame({
            'Metric': [
                'Portfolio Beta to Market', 'Portfolio Alpha (Annual)',
                'Portfolio Orthogonal Return', 'Portfolio Orthogonal Volatility',
                'Portfolio Orthogonal Sharpe', 'Portfolio Max Drawdown (Ortho)'
            ],
            'Value': [
                analyzer.orthogonalizer.beta_coefficients['PORTFOLIO'],
                analyzer.orthogonalizer.orthogonalized_returns['PORTFOLIO']['alpha'] * 252,
                results['portfolio_ortho_return'].mean(),
                results['portfolio_ortho_vol'].mean(),
                results['portfolio_ortho_sharpe'].mean(),
                results['portfolio_ortho_max_dd'].mean()
            ]
        })
        portfolio_stats.to_excel(writer, sheet_name='Portfolio_Stats', index=False)

    print(f"Comprehensive report saved to {output_path}")


# ============================================
# EXECUTION
# ============================================

if __name__ == "__main__":
    analyzer, results, risk_df = main()

    # Uncomment to generate Excel report:
    # create_orthogonal_report(analyzer, results, 'orthogonal_analysis_report.xlsx')
