import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt

print("Historical VaR") 

# ---------------------------------------------------------
# 1. Parameter Definitions
# ---------------------------------------------------------
# Define the historical time window for data retrieval (last 15 years)
years = 15
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days = 365*years)

# Define the assets in the portfolio (e.g., European banks and Gold)
tickers = ['DBK.DE', 'CBK.DE' , 'SAN.MC', 'BNP.PA', 'GLD']

# VaR Parameters
days = 100                # Investment horizon for the VaR (100-day window)
prf_value = 1000000       # Initial portfolio value in USD/EUR
conf_lvl = 0.95           # Confidence level (95% -> we look at the worst 5% of the history)

# ---------------------------------------------------------
# 2. Data Fetching & Processing
# ---------------------------------------------------------
# Download historical adjusted closing prices from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
adj_cl = data['Close']

# Calculate daily logarithmic returns
# (Log returns are standard in finance as they are additive over time)
ln_ret = np.log(adj_cl / adj_cl.shift(1))
ln_ret = ln_ret.dropna() 

# ---------------------------------------------------------
# 3. Historical Portfolio Returns Calculation
# ---------------------------------------------------------
# Create an equally weighted portfolio (1/n for each ticker)
weights = np.array([1 / len(tickers)] * len(tickers))

# Calculate the actual daily historical returns of the total portfolio
hist_ret = (ln_ret * weights).sum(axis=1)
print("Daily Portfolio Returns:\n", hist_ret)

# Calculate the 100-day returns using a rolling window.
# Since we use log returns, we can simply sum up the daily returns (.sum)
roll_ret = hist_ret.rolling(window=days).sum()
roll_ret = roll_ret.dropna() 

# ---------------------------------------------------------
# 4. Value at Risk (VaR) Calculation & Visualization
# ---------------------------------------------------------
# Calculate Historical VaR:
# We find the percentile of the worst-case scenarios. At 95% confidence, this is the 5th percentile (100 - 95).
# The minus sign inverts the value, as VaR is traditionally reported as a positive loss amount.
VaR = -np.percentile(roll_ret, 100 - (conf_lvl * 100)) * prf_value
print(f"\nVaR (100 Days, 95%): {VaR:,.2f}")

# Convert the 100-day percentage returns into absolute monetary amounts (Profit and Loss)
pnl = roll_ret * prf_value

# Visualize the distribution as a histogram
plt.hist(pnl, bins=50, density=True, alpha=0.75, edgecolor='black')
plt.xlabel(f'{days}-Day Portfolio Return (Dollar Value)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Historical {days}-Day Portfolio Returns')

# Draw the VaR limit as a red dashed line in the loss zone
plt.axvline(-VaR, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {conf_lvl:.0%} confidence')

plt.legend()
plt.show()
