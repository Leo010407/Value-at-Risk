import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

print("Parametric VaR")

# =============================================================================
# 1. Configuration & Parameters
# =============================================================================

# Define the historical time window for calculating returns and volatility
years = 15
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365 * years)

# Define the assets in the portfolio (European bank stocks)
tickers = ['DBK.DE', 'CBK.DE', 'SAN.MC', 'BNP.PA']

# Portfolio and VaR projection parameters
days = 100                # Time horizon in trading observations
prf_value = 1000000        # Initial portfolio value in EUR

# =============================================================================
# 2. Data Acquisition & Preprocessing
# =============================================================================

# Fetch historical market data from Yahoo Finance
# auto_adjust=True adjusts prices for stock splits and dividends
data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True
)

if data.empty:
    raise ValueError("No market data was downloaded.")

adj_cl = data['Close']

# Keep prices in the same order as the portfolio weights
adj_cl = adj_cl.reindex(columns=tickers)

if adj_cl.empty or adj_cl.isna().all().any():
    raise ValueError("Missing price data for one or more tickers.")

if (adj_cl <= 0).any().any():
    raise ValueError("Prices must be positive.")

# Calculate daily simple returns without filling missing prices
returns = adj_cl.pct_change(fill_method=None).dropna()

if len(returns) < max(days, 2):
    raise ValueError("Not enough historical data for the selected horizon.")

if not np.isfinite(returns.to_numpy()).all():
    raise ValueError("Returns contain non-finite values.")

# =============================================================================
# 3. Portfolio Volatility Calculation
# =============================================================================

# Create an equally weighted portfolio (1/n allocated to each asset)
weights = np.array([1 / len(tickers)] * len(tickers))

# Calculate the annualized covariance matrix
# (assuming approximately 252 trading days in a calendar year)
cov_matrix = returns.cov() * 252

# Calculate the annualized portfolio standard deviation (volatility)
# Formula: sqrt(Weights^T * Covariance_Matrix * Weights)
prf_std_dv = np.sqrt(weights.T @ cov_matrix @ weights)

# =============================================================================
# 4. Parametric VaR Calculation
# =============================================================================

confidence_levels = [0.90, 0.95, 0.99]
VaRs = []

# Normal-return approximation with zero expected return.
# Square-root-of-time scaling assumes constant variance and
# no serial correlation. This is approximate over 100 days.
for cl in confidence_levels:
    # Parametric VaR Formula:
    # Portfolio_Value * Annualized_Volatility * Z-Score * sqrt(days / 252)
    VaR = prf_value * prf_std_dv * norm.ppf(cl) * np.sqrt(days / 252)
    VaRs.append(VaR)

# Display the calculated VaR metrics in a formatted table
print(f'\n{"Confidence Level":<20} {"Value at Risk (EUR)":<20}')
print('-' * 40)

for cl, VaR in zip(confidence_levels, VaRs):
    print(f'{cl*100:>15.0f}%: {"":<2} €{VaR:>10,.2f}')

# =============================================================================
# 5. Visualization (Historical Context vs. Parametric Estimation)
# =============================================================================

# Compare historical compounded P&L with parametric VaR thresholds.
# This is an in-sample comparison, not an out-of-sample backtest.

# Calculate daily portfolio returns.
# Assumes daily rebalancing to equal weights, without transaction costs.
hist_ret = (returns * weights).sum(axis=1)

# Compound daily simple returns over the selected horizon
roll_ret = (
    (1 + hist_ret)
    .rolling(window=days)
    .apply(np.prod, raw=True)
    - 1
).dropna()

# Convert compounded simple returns into monetary profit/loss
pnl = roll_ret * prf_value

# Initialize the plot
plt.figure(figsize=(10, 6))

# Plot the distribution of historical compounded P&L
plt.hist(
    pnl,
    bins=50,
    density=True,
    alpha=0.5,
    label=f'Historical Compounded {days}-Day P&L'
)

# Overlay the Parametric VaR thresholds as vertical lines
colors = ['yellow', 'orange', 'red']

for cl, VaR, color in zip(confidence_levels, VaRs, colors):
    plt.axvline(
        x=-VaR,
        linestyle='--',
        color=color,
        linewidth=2,
        label=f'Parametric VaR ({cl*100:.0f}%)'
    )

# Formatting the chart
plt.xlabel(f'{days}-Day Portfolio Profit / Loss (EUR)')
plt.ylabel('Probability Density')
plt.title(
    f'Distribution of Portfolio {days}-Day P&L '
    'vs Parametric VaR Estimates'
)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Render the plot
plt.show()
