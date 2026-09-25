# Portfolio Value at Risk (VaR) Simulator

A quantitative finance project calculating the Value at Risk (VaR) for a diversified portfolio using three distinct mathematical modeling approaches. The project fetches real-time historical market data, calculates logarithmic returns, and evaluates portfolio downside risk over a 100-day time horizon at various confidence intervals (e.g., 95%, 99%).

## Methodologies Implemented

This repository contains three independent scripts, each demonstrating a different standard industry approach to risk modeling:

1. **Monte Carlo Simulation (`VaR_Project_MonteCarlo.py`)**
   - Generates 10,000 random portfolio return scenarios based on a normal distribution of market shocks (Z-scores).
   - Utilizes historical expected returns and the portfolio's covariance matrix to simulate stochastic price paths.

2. **Parametric / Variance-Covariance Approach (`VaR_Project_Parametric.py`)**
   - An analytical approach assuming a normal distribution of returns.
   - Calculates annualized portfolio volatility using a covariance matrix and scales it to the specified time horizon using the square root of time rule.

3. **Historical Simulation (`VaR_Project_Historical.py`)**
   - A non-parametric approach that applies actual historical daily logarithmic returns over a 15-year rolling window to the current portfolio weights to find the worst-case percentiles.

## Portfolio Composition & Data

- **Data Source:** Yahoo Finance API (`yfinance`)
- **Time Horizon:** custom, for example 15 years of historical adjusted closing prices.
- **Assets:** A custom equally-weighted portfolio.

## Tech Stack

- **Python 3.x**
- **NumPy & SciPy:** Matrix operations, statistical distributions, and percentiles.
- **Pandas:** Time-series data manipulation and rolling window calculations.
- **Matplotlib:** Data visualization and histogram generation.
- **yfinance:** Market data pipeline.

## Visualizations

The project generates visualizations for each methodology to illustrate the portfolio return distributions and the calculated VaR thresholds.

##  Acknowledgments & References
The core mathematical concepts and foundational Python implementation in this project were inspired by the educational quantitative finance materials provided by Ryan O'Connell, CFA, FRM. 
