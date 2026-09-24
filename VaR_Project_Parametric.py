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

# Define the historical time window for calculating returns and volatility (15 years)
years = 15
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365 * years)

# Define the assets in the portfolio (e.g., European Banks and Gold)
tickers = ['DBK.DE', 'CBK.DE', 'SAN.MC', 'BNP.PA', 'GLD']

# Portfolio and VaR projection parameters
days = 100                # Time horizon for the VaR projection (in trading days)
prf_value = 1000000       # Initial portfolio value (in base currency, e.g., USD/EUR)

# =============================================================================
# 2. Data Acquisition & Preprocessing
# =============================================================================

# Fetch historical market data from Yahoo Finance
# auto_adjust=True automatically adjusts for stock splits and dividend payouts
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
adj_cl = data['Close']

# Calculate daily logarithmic returns (continuous compounding)
# Formula: ln(P_t / P_{t-1})
ln_ret = np.log(adj_cl / adj_cl.shift(1))
ln_ret = ln_ret.dropna() 

# =============================================================================
# 3. Portfolio Volatility Calculation
# =============================================================================

# Create an equally weighted portfolio (1/n allocated to each asset)
weights = np.array([1 / len(tickers)] * len(tickers))

# Calculate the annualized covariance matrix 
# (assuming ~252 trading days in a calendar year)
cov_matrix = ln_ret.cov() * 252

# Calculate the annualized portfolio standard deviation (volatility)
# Formula: sqrt(Weights^T * Covariance_Matrix * Weights)
prf_std_dv = np.sqrt(weights.T @ cov_matrix @ weights)

# =============================================================================
# 4. Parametric VaR Calculation
# =============================================================================

confidence_levels = [0.90, 0.95, 0.99]
VaRs = []

for cl in confidence_levels:
    # Parametric VaR Formula: Portfolio_Value * Volatility * Z-Score * sqrt(T)
    # norm.ppf(cl) gets the Z-score (inverse of the standard normal CDF)
    # np.sqrt(days/252) scales the annualized volatility down to the desired time horizon
    VaR = prf_value * prf_std_dv * norm.ppf(cl) * np.sqrt(days / 252)
    VaRs.append(VaR)

# Display the calculated VaR metrics in a formatted table
print(f'\n{"Confidence Level":<20} {"Value at Risk":<20}')
print('-' * 40)

for cl, VaR in zip(confidence_levels, VaRs):
    print(f'{cl*100:>15.0f}%: {"":<2} ${VaR:>10,.2f}')

# =============================================================================
# 5. Visualization (Historical Context vs. Parametric Estimation)
# =============================================================================

# To evaluate our Parametric VaR, we calculate the actual historical returns 
# of the portfolio over the specified rolling window.
hist_ret = (ln_ret * weights).sum(axis=1)
roll_ret = hist_ret.rolling(window=days).sum().dropna()

# Convert the rolling percentage returns into absolute monetary profit/loss (PnL)
pnl = roll_ret * prf_value

# Initialize the plot
plt.figure(figsize=(10, 6))

# Plot the distribution of the actual historical PnL
plt.hist(pnl, bins=50, density=True, alpha=0.5, label=f'Historical {days}-Day Returns')

# Overlay the Parametric VaR thresholds as vertical lines
colors = ['yellow', 'orange', 'red']
for cl, VaR, color in zip(confidence_levels, VaRs, colors):
    plt.axvline(x=-VaR, linestyle='--', color=color, linewidth=2, 
                label=f'Parametric VaR ({cl*100:.0f}%)')

# Formatting the chart
plt.xlabel(f'{days}-Day Portfolio Return (Base Currency)')
plt.ylabel('Density / Frequency')
plt.title(f'Distribution of Portfolio {days}-Day Returns vs Parametric VaR Estimates')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Render the plot
plt.show()
