# Simulation of Dynamic Limit Order Book
This repository is dedicated to our innovative project on simulating a dynamic limit order book based on the Ising model.

This project contains Python code that replicates and simulates the dynamic limit order book models described in the paper "The Mathematical Modeling of Dynamic Limit Order Book". The code allows you to run agent-based simulations of a continuous double auction stock market using the model frameworks proposed in the paper.

## Models

The paper puts forward two key model frameworks for simulating a dynamic limit order book:

1. **Ising Model**: This is a statistical mechanics model originally used to describe ferromagnetism. In the financial context, it models the interactions between neighboring traders in a 2D grid. Each trader's state (buy/sell) is influenced by the states of its adjacent neighbors. The Hamiltonian of the system is given by:

   $H = -J \sum_{(i,j)} S_i S_j - h \sum_i S_i$

   where $J$ is the interaction strength, $S_i$ is the state of trader $i$, and $h$ is the external field.

2. **Bornholdt-Kaizoji-Fujiwara (BKF) Model**: This extends the Ising model by introducing a minority game mechanism. Traders are driven by two conflicting economic forces:
   - "What will the neighbors do?" - local influence from adjacent traders  
   - "What will the minority group do?" - global influence to be in the minority

   The BKF model introduces a coupling term to the Hamiltonian:

   $h_i(t) = \sum_{(i,j)} J_{ij} S_i(t) - \alpha C_i(t) \frac{1}{N} \sum_j S_j(t)$

   where $\alpha$ is the coupling coefficient, $C_i$ is the strategy of trader $i$ (fundamentalist or noise trader), and $N$ is the total number of traders.

The paper improves upon the BKF model to better align with real stock market mechanisms:

- Instead of assuming a fixed "fair price", the model determines prices endogenously through the limit order book mechanism.
- Traders' order prices are drawn from normal distributions with means adjusted based on their buy/sell stance and the market trend.
- The model incorporates price limits, allowing prices to fluctuate within a certain range of the previous day's closing price.
- The waiting time between a trader's orders is modeled as an exponential distribution.

## Code Structure

The code is organized into the following files:

- `agent.py`: Defines the `Agent` class representing individual traders. Agents have attributes like position, trading time, order price, activity status, and strategies. 

- `market.py`: Implements the `Market` class encapsulating the entire market simulation. It handles agent initialization, order matching, price updates based on the limit order book, and implements the Ising and BKF model dynamics. Key methods include:
   - `init_agents`: Initializes agents with random trading times, prices, and strategies.
   - `update_agents`: Updates agents' states and prices based on the Ising and BKF model rules.
   - `update_market`: Matches buy and sell orders to determine the transaction price.

- `simulate.py`: Contains functions to run the trading simulation for a specified number of days. It orchestrates the market simulation, collects results like return rates and stock prices, and handles day-to-day market transitions.

- `utils.py`: Provides utility functions for analysis and plotting of simulation results. This includes plotting price series, return distributions, fitting theoretical distributions to returns, etc.

## How to Run

To run a simulation:

1. Ensure you have the required Python packages installed (numpy, matplotlib, scipy).

2. Import the `simulate_trading` function from `simulate.py` and the `Market` class from `market.py`. 

3. Create an instance of the `Market` class, specifying parameters like grid size, interaction strength, coupling coefficients, noise levels, etc.

4. Call `simulate_trading`, passing in the `Market` instance and the number of days to simulate. This will run the simulation and return results as dictionaries.

5. Use the functions in `utils.py` to analyze and visualize the simulation results.

For example:

```python
# Set the random seed for reproducibility
np.random.seed(42)
random.seed(42)

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
```

## Key Findings

The paper's simulations demonstrate several key stylized facts about financial markets:

- **Stock price dynamics**: The simulated stock prices exhibit upward trends punctuated by multiple peaks and troughs. This price behavior closely resembles the patterns observed in real financial markets, capturing the complex interplay of bull and bear market phases.

- **Daily return fluctuations**: The model generates daily returns that show significant fluctuations. The range of these fluctuations is consistent with the volatility levels typically observed in real stock markets. This suggests that the model effectively captures the inherent uncertainty and variability of financial returns.

- **Fat-tailed return distributions**: A striking feature of the simulated returns is their pronounced fat tails compared to a normal distribution. The return distributions exhibit a leptokurtic shape, with higher peaks and more extreme values than would be expected under a Gaussian model. This finding aligns with the well-established empirical fact that financial return series are characterized by excess kurtosis.

- **Time-varying tail behavior**: By fitting Student's t-distributions to the fat tails of the return distributions, the paper uncovers a time-dependent behavior. As the time interval over which returns are calculated increases, the degrees of freedom of the fitted t-distributions also increase. This implies that over longer time scales, the return distributions gradually approach a more "normal" shape. This finding sheds light on the time-aggregation properties of financial returns and suggests that the non-normality is most pronounced at high frequencies.

- **Power law tail exponents**: Complementing the t-distribution analysis, the paper also fits power law functions to the extreme tails of the return distributions. The results reveal that the power law exponents increase as the time interval grows. Interestingly, the exponents consistently fall within the range of [3.5, 4] across different time scales. This range is in line with the "inverse cubic law" often reported in empirical studies of financial markets.

- **Robustness to empirical data**: To validate the model's findings, the paper conducts an empirical analysis using high-frequency data from real stock markets. The results confirm that fat-tailed return distributions are indeed a ubiquitous feature across different markets and assets. While the exact tail exponents may vary depending on the specific market and time period, the overall presence of heavy tails is a robust empirical fact that the model successfully reproduces.

The proposed dynamic limit order book model stands out for its ability to generate realistic stock price dynamics and return distributions using a parsimonious set of assumptions. By incorporating key elements of trader behavior, such as their order prices, waiting times, and the influence of local and global market conditions, the model provides a mechanistic explanation for the emergence of stylized facts in financial markets.

The model's explanatory power lies in its ability to capture the complex interactions between heterogeneous traders without relying on exogenous noise or fundamental value processes. Instead, the rich market dynamics arise endogenously from the collective behavior of boundedly rational agents operating under simple rules.

Moreover, the model's focus on the limit order book mechanism highlights the crucial role of market microstructure in shaping the statistical properties of returns. By explicitly simulating the order flow and the evolution of the order book, the model provides a realistic representation of the price formation process in modern electronic markets.

The paper's findings have important implications for various domains of finance, including risk management, derivative pricing, and market regulation. Understanding the origin and nature of fat tails is crucial for accurately assessing and managing financial risks. The model's insights into the time-varying behavior of tail exponents can inform the design of more robust risk models and trading strategies.

Furthermore, the model's ability to generate realistic market dynamics from a bottom-up perspective opens up new avenues for exploring the effects of different market mechanisms and regulatory policies. By modifying the model's assumptions or introducing new trader types, researchers can gain valuable insights into how changes in market design or trader behavior might impact market stability and efficiency.

In conclusion, the dynamic limit order book model proposed in this paper offers a powerful and parsimonious framework for understanding the complex behavior of financial markets. The model's key findings, supported by extensive simulations and empirical evidence, contribute to our understanding of the origins of stylized facts and the importance of market microstructure in shaping the statistical properties of returns. The provided code and detailed explanations enable researchers to reproduce, extend, and build upon these findings, paving the way for further advancements in the field of financial modeling and market analysis.
