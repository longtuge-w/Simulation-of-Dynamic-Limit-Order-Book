import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from market import Market
from scipy.stats import norm, t
from utils import *
from simulate import simulate_trading


def main():
    # Set the random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Load and preprocess the data
    data = pd.read_csv("GOOGL_march_2024_intraday.csv", parse_dates=['Datetime'], index_col='Datetime')
    data = data.dropna()
    log_returns = np.log(data['Close']).diff()

    # Simulation parameters
    num_agents = 400
    num_intervals = 480
    num_days = 130
    J = 1
    alpha = 0.1
    beta = 0.1
    a = 0.1
    b = 0.1
    p_star = 100

    # Initialize the market
    market = Market(num_agents, num_intervals, J, alpha, a, b, p_star)
    market.populate_agents((1, 5), p_star, 1)

    # Simulate the market dynamics
    simulation_results = {'stock_prices': [], 'returns': []}
    for day in range(num_days):
        for interval in range(num_intervals):
            market.update_agents(interval, market.current_price)
            market.update_market(J, alpha, beta)
            market.current_price = market.calculate_price()
            simulation_results['stock_prices'].append(market.current_price)
        market.previous_price = market.current_price

    # Calculate the simulated returns
    simulation_results['returns'] = np.diff(np.log(simulation_results['stock_prices']))

    # Plot the simulated stock prices and returns
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(simulation_results['stock_prices'], 'b-')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Simulated Stock Prices')

    plt.subplot(1, 2, 2)
    plt.plot(simulation_results['returns'], 'r-')
    plt.xlabel('Time')
    plt.ylabel('Log Returns')
    plt.title('Simulated Log Returns')
    plt.tight_layout()
    plt.show()

    # Compare the return distribution with a normal distribution
    plt.figure(figsize=(8, 6))
    plt.hist(simulation_results['returns'], bins=50, density=True, alpha=0.7, label='Simulated Returns')
    x = np.linspace(min(simulation_results['returns']), max(simulation_results['returns']), 100)
    plt.plot(x, norm.pdf(x, loc=np.mean(simulation_results['returns']), scale=np.std(simulation_results['returns'])), 'r-', label='Normal Distribution')
    plt.xlabel('Log Returns')
    plt.ylabel('Density')
    plt.title('Return Distribution')
    plt.legend()
    plt.show()

    # Fit the tail of the return distribution with a power law function
    returns = simulation_results['returns']
    returns = np.sort(returns)
    n = len(returns)
    tail_start = int(0.9 * n)  # Consider the top 10% of the data as the tail
    x = returns[tail_start:]
    y = np.arange(tail_start, n) / n

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'bo', label='Tail Data')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Log Returns')
    plt.ylabel('Cumulative Probability')
    plt.title('Tail Distribution (Log-Log Scale)')

    # Fit the tail with a power law function
    coeff = np.polyfit(np.log(x), np.log(y), 1)
    plt.plot(x, np.exp(coeff[1]) * x**coeff[0], 'r-', label=f'Power Law Fit (alpha={-coeff[0]:.2f})')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()