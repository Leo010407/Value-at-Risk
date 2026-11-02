import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt

print("Historical VaR")

# ---------------------------------------------------------
# 1. Parameter Definitions
# ---------------------------------------------------------

# Define the historical time window for data retrieval
years = 15
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365 * years)

# Define the assets in the portfolio (European bank stocks)
tickers = ['DBK.DE', 'CBK.DE', 'SAN.MC', 'BNP.PA']

# VaR Parameters
days = 100                # Time horizon in trading observations
prf_value = 1000000        # Initial portfolio value in EUR
conf_lvl = 0.95            # Confidence level

# ---------------------------------------------------------
# 2. Data Fetching & Processing
# ---------------------------------------------------------

# Download historical adjusted closing prices
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

# ---------------------------------------------------------
# 3. Historical Portfolio Returns Calculation
# ---------------------------------------------------------

# Create an equally weighted portfolio
weights = np.array([1 / len(tickers)] * len(tickers))

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

# Convert compounded returns into monetary profit/loss
pnl = roll_ret * prf_value

# ---------------------------------------------------------
# 4. Value at Risk Calculation & Visualization
# ---------------------------------------------------------

# VaR is the negative of the lower-tail P&L percentile.
# It is a loss threshold, not the maximum possible loss.
VaR = -np.percentile(pnl, 100 * (1 - conf_lvl))

print(f'\nPortfolio value: €{prf_value:,.2f}')
print(f'Horizon: {days} trading observations')
print(f'Historical scenarios: {len(pnl):,}')
print(f'VaR at {conf_lvl:.0%} confidence: €{VaR:,.2f}')

# Historical windows overlap, so scenarios are not independent.
plt.figure(figsize=(10, 6))

plt.hist(
    pnl,
    bins=50,
    density=True,
    alpha=0.75,
    edgecolor='black',
    label=f'Historical Compounded {days}-Day P&L'
)

plt.axvline(
    -VaR,
    color='r',
    linestyle='dashed',
    linewidth=2,
    label=f'VaR at {conf_lvl:.0%} confidence'
)

plt.xlabel(f'{days}-Day Portfolio Profit / Loss (EUR)')
plt.ylabel('Probability Density')
plt.title(f'Distribution of Historical {days}-Day Portfolio P&L')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.show()
