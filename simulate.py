import os


def simulate_trading(market, days=130, times=480):
    """
    Simulate trading for a specified number of days using the Bornholdt-Kaizoji-Fujiwara model.
    
    :param market: An instance of the Market class.
    :param days: Number of trading days to simulate.
    :param initial_time_range: Tuple specifying the range of initial times for agents to place their first order.
    :param initial_price_range: Tuple specifying the range of initial prices for agents.
    :return: A dictionary containing the return rates and stock prices for each trading day.
    """
    results = {
        'return_rates': [],
        'stock_prices': []
    }

    transaction_price_dict = {}
    transaction_return_dict = {}
    
    # Initialize agents for day 1
    market.init_agents()
    last_p_star = market.p_star
    up_down_days = 0

    for day in range(1, days + 1):
        print(f"Day {day}:")

        transaction_price_dict[day] = []
        transaction_return_dict[day] = []
        
        # Simulate each trading interval in the day
        for t in range(1, times + 1):
            # Here you'd implement the logic for matching orders and executing transactions
            # For simplicity, this example assumes transaction_prices are updated elsewhere
            market.update_agents(t)

            # Potentially, you update the market based on transactions and other factors
            market.update_market(t)  # Example beta value

            if market.p_star != market.previous_price:
                # print(f"Time {t}: Transaction Price: {market.p_star}")
                return_ = (market.p_star / market.previous_price - 1) * 100
                market.previous_price = market.p_star

                transaction_price_dict[day].append(market.p_star)
                transaction_return_dict[day].append(return_)

        if market.p_star >= last_p_star:
            if up_down_days >= 0:
                up_down_days += 1
            else:
                up_down_days = 1
            market.mu_init = market.p_star * 0.995 ** up_down_days
        else:
            if up_down_days <= 0:
                up_down_days -= 1
            else:
                up_down_days = -1
            market.mu_init = market.p_star * 1.025 ** abs(up_down_days)

        # Calculate stock return and update the price accordingly
        results['return_rates'].append(market.p_star / last_p_star - 1)
        results['stock_prices'].append(market.p_star)

        last_p_star = market.p_star

        market.populate_agents()
        
        # print(f"Return rate for day {day}: {return_rate}")
        print(f"Closing price for day {day}: {market.p_star}")
        
    print("Simulation complete.")
    return results, transaction_price_dict, transaction_return_dict