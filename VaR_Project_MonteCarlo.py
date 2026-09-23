import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def gen_z_score():
    # Generates a random number from a standard normal distribution (Mean = 0, Std Dev = 1)
    # This represents the random market shocks in our simulation.
    return np.random.normal(0,1)

def gain_loss(prf_value, prf_std_dev, z_score,prf_exp_ret, days):
    # Calculates the simulated portfolio gain or loss over the specified time horizon.
    # Formula components: 
    # 1. Expected drift: (prf_value * prf_exp_ret * days)
    # 2. Random shock: (prf_value * prf_std_dev * z_score * sqrt(days))
    return prf_value * prf_exp_ret * days + prf_value * prf_std_dev * z_score * np.sqrt(days)

def calc_exp_ret(weights, ln_ret):
    # Calculates the expected daily return of the portfolio by taking the 
    # dot product of the historical mean returns of each asset and their respective weights.
    return np.sum(ln_ret.mean() * weights)

def calc_std_dev(weights, cov_mat):
    # Calculates the daily volatility (standard deviation) of the portfolio.
    # Uses matrix multiplication: sqrt(Weights transposed * Covariance Matrix * Weights)
    variance = weights.T @ cov_mat @ weights
    return np.sqrt(variance)

# ---------------------------------------------------------
# Simulation Setup & Parameters
# ---------------------------------------------------------
print("Monte Carlo VaR")

# Define the historical time window for calculating returns and volatility
years = 15
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days = 365*years)

# Define the assets in the portfolio (e.g., European Banks and Gold)
tickers = ['DBK.DE', 'CBK.DE' , 'SAN.MC', 'BNP.PA', 'GLD']

# Define the Monte Carlo simulation parameters
simulations = 10000       # Number of random scenarios to generate
days = 100                # Time horizon for the VaR projection (e.g., 100 trading days)
prf_value = 1000000       # Initial portfolio value in dollars

# ---------------------------------------------------------
# Data Fetching & Processing
# ---------------------------------------------------------
adj_cl = pd.DataFrame()

# Download historical adjusted closing prices from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
adj_cl = data['Close']

# Calculate daily logarithmic returns (standard practice in financial modeling)
ln_ret = np.log(adj_cl / adj_cl.shift(1))
ln_ret = ln_ret.dropna() # Remove the first row which will be NaN due to the shift

# Calculate the covariance matrix to understand how the assets move together
cov_mat = ln_ret.cov()

# ---------------------------------------------------------
# Portfolio Metrics Calculation
# ---------------------------------------------------------

# Create an equally weighted portfolio (each asset gets 1/N of the capital)
weights = np.array([1 / len(tickers)] * len(tickers))

# Calculate the historical expected return and standard deviation of our specific portfolio
prf_exp_ret = calc_exp_ret(weights, ln_ret)
prf_std_dev = calc_std_dev(weights, cov_mat)

# ---------------------------------------------------------
# Monte Carlo Simulation Engine
# ---------------------------------------------------------
scen_ret = []

# Run the simulation "simulations" times to create a distribution of possible outcomes
for i in range(simulations):
    z_score = gen_z_score()
    # Calculate the simulated gain/loss for this specific random scenario and save it
    scen_ret.append(gain_loss(prf_value, prf_std_dev, z_score, prf_exp_ret, days))

# ---------------------------------------------------------
# Value at Risk (VaR) Calculation & Visualization
# ---------------------------------------------------------

# Set confidence level (95% means we expect losses to exceed VaR only 5% of the time)
conf_lvl = 0.95

# Calculate VaR by finding the exact percentile in our sorted simulated returns
# For 95% confidence, we look at the 5th percentile (100 * (1 - 0.95))
VaR = np.percentile(scen_ret, 100 * (1 - conf_lvl))
print("VaR:", VaR)

# Plot the distribution of all 10,000 simulated portfolio outcomes
plt.hist(scen_ret, bins=50, density=True)
plt.xlabel('Scenario Gain/Loss ($)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Portfolio Gain/Loss Over {days} Days')

# Draw a vertical line representing our worst-case VaR threshold
plt.axvline(VaR, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {conf_lvl:.0%} confidence level')

plt.legend()
plt.show()
