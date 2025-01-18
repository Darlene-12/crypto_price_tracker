# Importation of various libraries
from dash import Dash, html, dcc
import requests
import json
from pathlib import Path 
from typing import List, Dict, Optional
import time
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
import threading
import queue
import sqlite3
from abc import ABC, abstractmethod


# Creating the dataclass
@dataclass
class PriceAlert:
    crypto: str
    target_price: float
    condition: str # Above or below
    triggered: bool = False

class AlertHandler:
    def __init__(self):
        self.alerts: List[PriceAlert] = []
        self._alert_queue = queue.Queue()
        self._start_alert_worker()

    def add_alert(self, crypto: str, target_price: float, condition: str) -> None:
        alert = PriceAlert(crypto, target_price, condition)
        self.alerts.append(alert)

    def check_alerts(self, prices: Dict) -> None:
        for alert in self.alerts:
            if not alert.triggered:
                current_price = prices.get(alert.crypto, {}). get('usd', 0)
                if (alert.condition == 'above'and current_price > alert.target_price) or \
                    (alert.condition == 'below' and current_price < alert.target_price):
                    alert.triggered = True
                    self._alert_queue.put(
                        f"Alert: {alert.crypto.upper()} price is {alert.condition}"
                        f"${alert.target_price:,.2f} (Current price: ${current_price:,.2f})"
                    )
    def _start_alert_worker(self):
        def alert_worker():
            while True:
                alert = self. _alert_queue.get()
                print(f"\n{alert}\n")
                self._alert_queue.task_done()

        thread = threading.Thread(target = alert_worker, daemon = True)
        thread.start()

# Creating the data storage class
class DataStorage (ABC):
    @abstractmethod
    def save_price_data(self, data: Dict) -> None:
        pass

    @abstractmethod
    def load_price_data(self) -> Dict:
        pass

# Class for storing the data in the SQLite database
class SQLiteStorage(DataStorage):
    def __init__(self, db_path: str = "crypto_prices.db"):
        self.db_path = db_path
        self._init_db()  # Ensure this method exists

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    crypto TEXT,
                    price REAL,
                    timestamp TEXT,
                    PRIMARY KEY (crypto, timestamp)
                )
            """)

    def save_price_data(self, data:Dict) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for crypto, info in data.items():
                conn.execute(
                    "INSERT OR REPLACE INTO prices VALUES (?, ?, ? )",
                    (crypto, info ['usd'], info ['last_updated'])
                )
    def load_price_data(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn: 
            cursor = conn.execute (
                "SELECT crypto, price, timestamp FROM prices"
                "WHERE timestamp > datetime ('now', '-1 day')"
            )
            data = {}
            for row in cursor:
                if row [0] not in data:
                    data[row[0]] = []
                data [ row[0]].append({
                    'price': row [1],
                    'timestamp': row[2]
                })
            return data

# Class for fetching the data from the CoinGecko API
class CryptoPriceFetcher:
    def __init__(self):
        self.base_url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,litecoin,cardano,dogecoin&vs_currencies=usd&include_24hr_change=true&include_market_cap=true'
        self.cache_file = Path ("price_cache_json")
        self.cache_duration = 60
        self.supported_currencies = [
            "bitcoin", "ethereum", "litecoin", "cardano", 
            "dogecoin", "polkadot", "solana", "ripple"
        ]
        self. supported_flats = ["usd", "eur", "gbp", "jpy", "aud"]
        self.alert_handler = AlertHandler()
        self.storage = SQLiteStorage()

    # Add the load_cache method here
    def load_cache(self) -> Optional[Dict]:
        if self.cache_file.exists() and (time.time() - self.cache_file.stat().st_mtime) < self.cache_duration:
            with open(self.cache_file, 'r') as file:
                return json.load(file)
        return None

    # Add the _save_cache method here
    def _save_cache(self, data: Dict) -> None:
        with open(self.cache_file, 'w') as file:
            json.dump(data, file)

    def fetch_historical_data(
        self, 
        currency: str,
        days: int = 7,
        vs_currency: str = "usd"
    ) -> Optional [pd.DataFrame]:
        if vs_currency not in self.supported_flats:
            raise ValueError(f"Unsupported fiat currency: {vs_currency}")
 
        try:
            url = f"{self.base_url}/ coins/{currency}/ market_chart"
            params = {
                "vs_currency": vs_currency,
                "days": days,
                "interval": "daily"
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Convert the data to a Dataframe
            df = pd.DataFrame(data['prices'], columns = ['timestamp', 'price'])
            df ['timestamp'] = pd.to_datetime(df['timestamp'], unit = 'ms')
            return df

        except Exception as e:
                print ( f" Error fetching historical data for currency: {e}")
                return None
        

    def create_price_chart(
        self,
        currencies: List[str],
        days: int = 7,
        vs_currency: str = "usd"
    ) -> None:
        """Create an interactive price chart using plotly."""
        fig = go.Figure()
        
        for currency in currencies:
            df = self.fetch_historical_data(currency, days, vs_currency)
            if df is not None:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['price'],
                    name=currency.upper(),
                    mode='lines'
                ))
        
        fig.update_layout(
            title=f"Cryptocurrency Prices ({days} days)",
            xaxis_title="Date",
            yaxis_title=f"Price ({vs_currency.upper()})",
            hovermode='x unified'
        )
        
        # Save the chart
        fig.write_html("crypto_chart.html")
        print(f"\nChart saved as crypto_chart.html")

    def set_price_alert(self, crypto: str, target_price: float, condition: str) -> None:
        """Set a price alert for a cryptocurrency."""
        if condition not in ['above', 'below']:
            raise ValueError("Condition must be 'above' or 'below'")
        
        self.alert_handler.add_alert(crypto, target_price, condition)
        print(f"Alert set for {crypto.upper()} {condition} ${target_price:,.2f}")

    def fetch_prices(
        self,
        currencies: Optional[List[str]] = None,
        vs_currency: str = "usd"
    ) -> Optional[Dict]:
        if vs_currency not in self.supported_flats:
            raise ValueError(f" Unsupported flat currency: {vs_currency}")

        cached_data = self.load_cache()
        if cached_data:
            self.alert_handler.check_alerts(cached_data)
            return cached_data

        currencies = currencies or self.supported_currencies
        invalid_currencies = [c for c in currencies if not c in self.supported_currencies]
        if invalid_currencies:
            print(f" Warning: Unsupported currencies: {invalid_currencies}")
            currencies = [c for c in currencies if c in self.supported_currencies]

        if not currencies:
            raise ValueError("No valid currencies specified")

        params = {
            "ids": ",". join(currencies), 
            "vs_currencies": vs_currency,
            "include-24hr_change": True,
            "include_market_cap": True,
            "include_last_updated_at": True
        }

        try:
            response = requests.get(f"{self.base_url}/simple/price", params = params)

            if response.status_code ==429:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Rate limit hit. Waiting { retry_after} seconds...")
                time.sleep (retry_after)
                return self.fetch_prices(currencies, vs_currency)

            response.raise_for_status()
            data = response.json()

            for crypto in data:
                data[crypto]['last_updated'] = datetime.now().isoformat()

            self._save_cache(data)
            self.storage.save_price_data(data)
            self.alert_handler.check_alerts(data)  

            return data

        except Exception as e:
            print(f"Error fetching prices: {e}")
            if cached_data:
                print("Returning expired cache data due to error")
                return cached_data
            return None
    def get_price_summary(self, days: List[int] = [1,7,30]) -> Dict:
        summary = {}
        for currency in self.supported_currencies:
            df = self.fetch_historical_data(currency)
            if df is not None:
                summary[currency] = {
                    'current_price': df['price'].iloc[-1],
                    'changes': {}
                }
                for day in days:
                    if len(df) >= day:
                        change = ((df['price'].iloc[-1] -df ['price'].iloc[-day]) /
                         df['price'].iloc[-day] *100)
                        summary[currency]['changes'][f'{day}d'] = round(change, 2)
                    else:
                        summary[currency]['changes'][f'{day}d'] = None
        return summary

def main():
    fetcher = CryptoPriceFetcher()

    try:
        # Set some price alerts
        fetcher.set_price_alert("bitcoin", 45000, "above")
        fetcher.set_price_alert("ethereum", 2000, "above")

        # Fetch the current prices in different fiat currencies
        print("\nFetching prices in USD, EUR, GBP, JPY, AUD:")
        fetcher.fetch_prices(vs_currency="usd")
        fetcher.fetch_prices(vs_currency="eur")
        fetcher.fetch_prices(vs_currency="gbp")
        fetcher.fetch_prices(vs_currency="jpy")
        fetcher.fetch_prices(vs_currency="aud")

        # Fetch the current prices for the supported cryptocurrencies
        print("\nFetching prices for supported cryptocurrencies:")
        fetcher.fetch_prices()

        # Fetch the historical data for the supported cryptocurrencies
        print("\nFetching historical data for supported cryptocurrencies:")

        # Create the price charts
        fetcher.create_price_chart(
            currencies=["bitcoin", "ethereum", "litecoin", "cardano", "dogecoin"],
            days=7
        )

        # Get the summary of the prices
        summary = fetcher.get_price_summary()
        print("\nPrice Summary:")
        for crypto, data in summary.items():
            print(f"\n{crypto.upper()}")
            print(f"Current Price: ${data['current_price']:,.2f}")
            for days, change in data['changes'].items():
                print(f"{days} Change: {change:+.2f}%")

        # Monitor price alerts
        print("\nMonitoring price alerts (Press Ctrl+C to exit)...")
        while True:
            fetcher.fetch_prices()
            time.sleep(60)

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

# Important notes:
# Threading and queue: The multitaskers, enabling the program to perform multiple activities simultaneously (e.g., tracking prices and checking alerts without delays).
# These tools, along with others like dataclasses and abc (abstract base classes), form the backbone of our crypto monitoring system.
