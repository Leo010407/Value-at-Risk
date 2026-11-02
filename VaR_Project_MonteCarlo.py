import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def gen_z_score(rng):
    # Generate a standard normal market shock
    return rng.normal(0, 1)


def gain_loss(prf_value, prf_std_dev, z_score, prf_exp_ret, days):
    # Linear normal approximation to portfolio profit/loss.
    # This does not simulate compounded daily price paths.
    return (
        prf_value * prf_exp_ret * days
        + prf_value * prf_std_dev * z_score * np.sqrt(days)
    )


def calc_std_dev(weights, cov_mat):
    # Calculate daily portfolio volatility
    variance = weights.T @ cov_mat @ weights
    return np.sqrt(variance)


# ---------------------------------------------------------
# Simulation Setup & Parameters
# ---------------------------------------------------------

print("Monte Carlo VaR")

# Define the historical time window
years = 15
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365 * years)

# Define the assets in the portfolio (European bank stocks)
tickers = ['DBK.DE', 'CBK.DE', 'SAN.MC', 'BNP.PA']

# Define simulation parameters
simulations = 10000
days = 100                # Time horizon in trading observations
prf_value = 1000000        # Initial portfolio value in EUR
conf_lvl = 0.95

# A fixed seed makes the simulation reproducible for identical data
rng = np.random.default_rng(42)

# ---------------------------------------------------------
# Data Fetching & Processing
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

# Calculate the daily covariance matrix
cov_mat = returns.cov()

# ---------------------------------------------------------
# Portfolio Metrics Calculation
# ---------------------------------------------------------

# Create an equally weighted portfolio
weights = np.array([1 / len(tickers)] * len(tickers))

# Use zero expected return to match the parametric model
prf_exp_ret = 0.0

# Calculate daily portfolio volatility
prf_std_dev = calc_std_dev(weights, cov_mat)

# ---------------------------------------------------------
# Monte Carlo Simulation Engine
# ---------------------------------------------------------

# Assumes normal returns, constant variance, and no serial correlation.
# Square-root-of-time scaling is approximate over 100 days.
scen_ret = []

for i in range(simulations):
    z_score = gen_z_score(rng)

    scen_ret.append(
        gain_loss(
            prf_value,
            prf_std_dev,
            z_score,
            prf_exp_ret,
            days
        )
    )

scen_ret = np.array(scen_ret)

# ---------------------------------------------------------
# Value at Risk Calculation & Visualization
# ---------------------------------------------------------

VaR = -np.percentile(scen_ret, 100 * (1 - conf_lvl))

# Analytical VaR under the same normal-return model.
# Monte Carlo should approach this value as simulations increase.
parametric_VaR = prf_value * (
    prf_std_dev * norm.ppf(conf_lvl) * np.sqrt(days)
    - prf_exp_ret * days
)

print(f'\nPortfolio value: €{prf_value:,.2f}')
print(f'Horizon: {days} trading observations')
print(f'Simulations: {simulations:,}')
print(f'Monte Carlo VaR at {conf_lvl:.0%}: €{VaR:,.2f}')
print(f'Analytical VaR at {conf_lvl:.0%}: €{parametric_VaR:,.2f}')

plt.figure(figsize=(10, 6))

plt.hist(
    scen_ret,
    bins=50,
    density=True,
    alpha=0.75,
    edgecolor='black',
    label='Simulated Portfolio P&L'
)

plt.axvline(
    -VaR,
    color='r',
    linestyle='dashed',
    linewidth=2,
    label=f'Monte Carlo VaR ({conf_lvl:.0%})'
)

plt.axvline(
    -parametric_VaR,
    color='black',
    linestyle='dotted',
    linewidth=2,
    label=f'Analytical VaR ({conf_lvl:.0%})'
)

plt.xlabel(f'{days}-Day Portfolio Profit / Loss (EUR)')
plt.ylabel('Probability Density')
plt.title(f'Distribution of Simulated {days}-Day Portfolio P&L')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.show()
