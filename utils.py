import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt

    

def plot_stock_prices(stock_prices):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(stock_prices)), stock_prices, marker='o', linestyle='-', color='blue')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.title('Time Series of Stock Prices')
    plt.grid(True)
    plt.savefig("stock_price.png")
    plt.show()
    plt.close()


def plot_daily_returns(return_rates):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(return_rates)), return_rates, marker='o', linestyle='-', color='blue')
    plt.xlabel('Days')
    plt.ylabel('Daily Return')
    plt.title('Daily Return Rates')
    plt.grid(True)
    plt.savefig("daily_return.png")
    plt.show()
    plt.close()


def plot_return_distribution(return_rates, bins=50):
    plt.figure(figsize=(10, 6))
    plt.hist(return_rates, bins=bins, density=True, alpha=0.7, color='blue')
    
    # Calculate and plot the normal distribution for comparison
    mean = np.mean(return_rates)
    std = np.std(return_rates)
    x = np.linspace(min(return_rates), max(return_rates), 100)
    normal_dist = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    plt.plot(x, normal_dist, color='red', linewidth=2, label='Normal Distribution')
    
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.title('Distribution of Daily Returns')
    plt.legend()
    plt.grid(True)
    plt.savefig("return_distribution.png")
    plt.show()
    plt.close()


def plot_log_returns(return_rates):
    log_returns = np.log(1 + np.array(return_rates))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(log_returns)), log_returns, marker='o', linestyle='-', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Daily Log Return')
    plt.title('Time Series of Daily Log Returns')
    plt.grid(True)
    plt.savefig('daily_log_return')
    plt.show()
    plt.close()


def plot_transaction_return_rates(transaction_returns, start_date=None, end_date=None):
    if start_date is None:
        start_date = 1
    if end_date is None:
        end_date = len(transaction_returns)

    # Flatten the transaction return rates into a single list
    return_rates = []
    for day in range(start_date, end_date+1):
        return_rates.extend(transaction_returns[day])

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(return_rates)), return_rates, linestyle='-', color='blue')
    plt.xlabel('Day')
    plt.ylabel('Return Rate (Percentage)')
    plt.title(f'Time Series of Return Rates from day {start_date} to {end_date}')
    plt.grid(True)
    plt.savefig(f'transaction_returns_{start_date}_{end_date}.png')
    plt.show()
    plt.close()


def plot_transaction_log_returns(transaction_returns, start_date=None, end_date=None):
    if start_date is None:
        start_date = 1
    if end_date is None:
        end_date = len(transaction_returns)

    # Flatten the transaction return rates into a single list
    return_rates = []
    for day in range(start_date, end_date+1):
        return_rates.extend(transaction_returns[day])

    log_returns = np.log(1 + np.array(return_rates) / 100)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(log_returns)), log_returns, linestyle='-', color='blue')
    plt.xlabel('Day')
    plt.ylabel('Log Return')
    plt.title(f'Time Series of Log Returns from day {start_date} to {end_date}')
    plt.grid(True)
    plt.savefig(f'transaction_log_returns_{start_date}_{end_date}.png')
    plt.show()
    plt.close()


def plot_transaction_return_distribution(transaction_returns, start_date=None, end_date=None, bins=50):
    if start_date is None:
        start_date = 1
    if end_date is None:
        end_date = len(transaction_returns)

    # Flatten the transaction return rates into a single list
    return_rates = []
    for day in range(start_date, end_date+1):
        return_rates.extend(transaction_returns[day])

    # Calculate and plot the normal distribution for comparison
    mean = np.mean(return_rates)
    std = np.std(return_rates)
    x = np.linspace(min(return_rates), max(return_rates), 100)
    normal_dist = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    plt.figure(figsize=(10, 6))
    plt.plot(x, normal_dist, color='red', linewidth=2, label='Normal Distribution')
    plt.hist(return_rates, bins=bins, density=True, alpha=0.7, color='blue')
    plt.xlabel('Return Rate')
    plt.ylabel('Density')
    plt.title(f'Distribution of Return Rates from day {start_date} to {end_date}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'transaction_return_distribution_{start_date}_{end_date}.png')
    plt.show()
    plt.close()


def plot_transaction_price(transaction_price, date):

    prices = transaction_price[date]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(prices)), prices)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'Price Time Series for Date {date}')
    plt.grid(True)
    plt.savefig(f'transaction_price_{date}.png')
    plt.show()
    plt.close()


def fit_return_distribution(transaction_returns, start_date=None, end_date=None, bins=50):

    if start_date is None:
        start_date = 1
    if end_date is None:
        end_date = len(transaction_returns)

    # Flatten the transaction return rates into a single list
    return_rates = []
    for day in range(start_date, end_date+1):
        return_rates.extend(transaction_returns[day])

    # Calculate the histogram of the return rates
    hist, bin_edges = np.histogram(return_rates, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Fit the Student's t-distribution
    df, loc, scale = stats.t.fit(return_rates)
    t_dist = stats.t.pdf(bin_centers, df, loc=loc, scale=scale)
    
    # Fit the power law function
    def power_law(x, a, b):
        return a * np.power(np.abs(x), -b)
    
    pos_returns = [r for r in return_rates if r > 0]
    neg_returns = [r for r in return_rates if r < 0]
    
    if len(pos_returns) > 0:
        pos_hist = np.interp(pos_returns, bin_centers, hist)
        pos_params, _ = opt.curve_fit(power_law, pos_returns, pos_hist)
        pos_power_law = power_law(bin_centers[bin_centers > 0], *pos_params)
    else:
        pos_params = [np.nan, np.nan]
        pos_power_law = np.zeros_like(bin_centers[bin_centers > 0])
    
    if len(neg_returns) > 0:
        neg_hist = np.interp(np.abs(neg_returns), bin_centers, hist)
        neg_params, _ = opt.curve_fit(power_law, np.abs(neg_returns), neg_hist)
        neg_power_law = power_law(np.abs(bin_centers[bin_centers < 0]), *neg_params)
    else:
        neg_params = [np.nan, np.nan]
        neg_power_law = np.zeros_like(bin_centers[bin_centers < 0])
    
    # Plot the fitted curves and the return distribution
    plt.figure(figsize=(10, 6))
    plt.hist(return_rates, bins=bins, density=True, alpha=0.7, color='blue', label='Return Distribution')
    plt.plot(bin_centers, t_dist, color='red', linewidth=2, label="Student's t-distribution")
    plt.plot(bin_centers[bin_centers > 0], pos_power_law, color='green', linewidth=2, label='Power Law (Positive Tail)')
    plt.plot(bin_centers[bin_centers < 0], neg_power_law, color='orange', linewidth=2, label='Power Law (Negative Tail)')
    plt.xlabel('Return Rate')
    plt.ylabel('Density')
    plt.title('Fitted Return Distribution')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'fitted_return_{start_date}_{end_date}.png')
    plt.show()
    plt.close()
    
    print("Student's t-distribution:")
    print(f"Degrees of Freedom: {df:.2f}")
    print(f"Location: {loc:.4f}")
    print(f"Scale: {scale:.4f}")
    
    print("\nPower Law (Positive Tail):")
    print(f"a: {pos_params[0]:.4f}")
    print(f"b: {pos_params[1]:.2f}")
    
    print("\nPower Law (Negative Tail):")
    print(f"a: {neg_params[0]:.4f}")
    print(f"b: {neg_params[1]:.2f}")


def plot_log_returns_and_distribution_stock(data, stock):
    # Calculate log returns
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    log_returns = data['LogReturn'].dropna()

    # Calculate return rates
    data['ReturnRate'] = data['Close'].pct_change()
    return_rates = data['ReturnRate'].dropna()

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot log returns
    ax1.plot(log_returns.index, log_returns, linestyle='-', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Log Return')
    ax1.set_title(f'Time Series of Log Returns for stock {stock}')
    ax1.grid(True)

    # Plot return distribution
    mean = return_rates.mean()
    std = return_rates.std()
    x = np.linspace(return_rates.min(), return_rates.max(), 100)
    normal_dist = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    ax2.plot(x, normal_dist, color='red', linewidth=2, label='Normal Distribution')
    ax2.hist(return_rates, bins=50, density=True, alpha=0.7, color='blue')
    ax2.set_xlabel('Return Rate')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Distribution of Return Rates for stock {stock}')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'log_returns_and_distribution_{stock}.png')
    plt.show()
    plt.close()