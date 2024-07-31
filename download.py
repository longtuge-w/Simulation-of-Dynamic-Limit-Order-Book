import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def download_intraday_data(stock, start_date, end_date):
    all_data = pd.DataFrame()
    current_date = start_date

    while current_date < end_date:
        week_later = current_date + timedelta(days=7)
        # Ensure end date is not beyond the overall end date
        if week_later > end_date:
            week_later = end_date
        data = yf.download(tickers=stock, start=current_date, end=week_later, interval="1m")
        all_data = pd.concat([all_data, data])
        current_date = week_later
    return all_data

# Define the stock symbols
stocks = ["GOOGL", "AAPL", "AMZN", "TSLA", "MSFT", "NVDA"]

# Define your start and end dates for the last full week available
start_date = datetime(2024, 3, 1)
end_date = datetime(2024, 3, 24)

# Adjust these dates to the last full week available when you run the code

# Download the data for each stock
for stock in stocks:
    print(f"Downloading data for {stock}")
    stock_data = download_intraday_data(stock, start_date, end_date)
    # Save to CSV
    stock_data.to_csv(f"{stock}_march_2024_intraday.csv")
