# config.json
{
    "api_key": "CG-Nb934WDL84Tx8zZUdqyjJn4E",
    "supported_cryptocurrencies": ["bitcoin", "ethereum"],
    "supported_fiats": ["usd", "eur"],
    "update_interval": 60,
    "cache_duration": 60,
    "retry_limit": 3
}

# crypto_tracker.py
import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union
import time
from datetime import datetime
import threading
import queue
from abc import ABC, abstractmethod
import requests
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import sqlite3
from dataclasses import dataclass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management class"""
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from file or environment variables"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path) as f:
                    config = json.load(f)
            else:
                config = {
                    "api_key": os.getenv("COINGECKO_API_KEY"),
                    "supported_cryptocurrencies": ["bitcoin", "ethereum"],
                    "supported_fiats": ["usd", "eur"],
                    "update_interval": 60,
                    "cache_duration": 60,
                    "retry_limit": 3
                }
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

@dataclass
class PriceAlert:
    crypto: str
    target_price: float
    condition: str
    triggered: bool = False

class RetryHandler:
    """Handle API retry logic with exponential backoff"""
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def execute(self, func, *args, **kwargs):
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                wait_time = (2 ** retries) + random.uniform(0, 1)
                logger.warning(f"Request failed. Retry {retries + 1}/{self.max_retries} after {wait_time:.2f}s")
                time.sleep(wait_time)
                retries += 1
        raise Exception("Max retries exceeded")

class AlertHandler:
    """Handle cryptocurrency price alerts"""
    def __init__(self):
        self.alerts: List[PriceAlert] = []
        self._alert_queue = queue.Queue()
        self._start_alert_worker()

    def add_alert(self, crypto: str, target_price: float, condition: str) -> None:
        alert = PriceAlert(crypto, target_price, condition)
        self.alerts.append(alert)
        logger.info(f"Alert set: {crypto} {condition} ${target_price:,.2f}")

    def check_alerts(self, prices: Dict) -> None:
        for alert in self.alerts:
            if not alert.triggered:
                current_price = prices.get(alert.crypto, {}).get('usd', 0)
                if self._check_condition(alert, current_price):
                    alert.triggered = True
                    self._alert_queue.put(self._format_alert_message(alert, current_price))

    def _check_condition(self, alert: PriceAlert, current_price: float) -> bool:
        return ((alert.condition == 'above' and current_price > alert.target_price) or
                (alert.condition == 'below' and current_price < alert.target_price))

    def _format_alert_message(self, alert: PriceAlert, current_price: float) -> str:
        return (f"🔔 Alert: {alert.crypto.upper()} price is {alert.condition} "
                f"${alert.target_price:,.2f} (Current: ${current_price:,.2f})")

    def _start_alert_worker(self):
        def alert_worker():
            while True:
                alert = self._alert_queue.get()
                logger.info(alert)
                self._alert_queue.task_done()

        thread = threading.Thread(target=alert_worker, daemon=True)
        thread.start()

class DataStorage(ABC):
    """Abstract base class for data storage"""
    @abstractmethod
    def save_price_data(self, data: Dict) -> None:
        pass

    @abstractmethod
    def load_price_data(self) -> Dict:
        pass

class SQLiteStorage(DataStorage):
    """SQLite implementation of data storage"""
    def __init__(self, db_path: str = "crypto_prices.db"):
        self.db_path = db_path
        self._init_db()

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

    def save_price_data(self, data: Dict) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for crypto, info in data.items():
                conn.execute(
                    "INSERT OR REPLACE INTO prices VALUES (?, ?, ?)",
                    (crypto, info['usd'], info['last_updated'])
                )

    def load_price_data(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT crypto, price, timestamp FROM prices "
                "WHERE timestamp > datetime('now', '-1 day')"
            )
            return self._process_query_results(cursor)

    def _process_query_results(self, cursor) -> Dict:
        data = {}
        for row in cursor:
            if row[0] not in data:
                data[row[0]] = []
            data[row[0]].append({
                'price': row[1],
                'timestamp': row[2]
            })
        return data

class CryptoTracker:
    """Main cryptocurrency tracking class"""
    def __init__(self):
        self.config = Config()
        self.base_url = "https://api.coingecko.com/api/v3"
        self.cache_file = Path("price_cache.json")
        self.alert_handler = AlertHandler()
        self.storage = SQLiteStorage()
        self.retry_handler = RetryHandler(max_retries=self.config.config['retry_limit'])

    def _get_headers(self) -> Dict:
        return {"x-cg-demo-api-key": self.config.config['api_key']}

    def fetch_prices(self, currencies: Optional[List[str]] = None,
                    vs_currency: str = "usd") -> Optional[Dict]:
        """Fetch current cryptocurrency prices"""
        if vs_currency not in self.config.config['supported_fiats']:
            raise ValueError(f"Unsupported fiat currency: {vs_currency}")

        currencies = currencies or self.config.config['supported_cryptocurrencies']
        params = {
            "ids": ",".join(currencies),
            "vs_currencies": vs_currency,
            "include_24hr_change": True,
            "include_market_cap": True
        }

        def _fetch():
            response = requests.get(
                f"{self.base_url}/simple/price",
                params=params,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()

        try:
            data = self.retry_handler.execute(_fetch)
            self._process_price_data(data)
            return data
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return None

    def _process_price_data(self, data: Dict) -> None:
        """Process and store price data"""
        for crypto in data:
            data[crypto]['last_updated'] = datetime.now().isoformat()
        self.storage.save_price_data(data)
        self.alert_handler.check_alerts(data)

def create_dash_app(tracker: CryptoTracker) -> Dash:
    """Create Dash web application"""
    app = Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Cryptocurrency Price Tracker"),
        
        html.Div([
            html.Label("Select Cryptocurrency:"),
            dcc.Dropdown(
                id='crypto-dropdown',
                options=[
                    {'label': c.capitalize(), 'value': c}
                    for c in tracker.config.config['supported_cryptocurrencies']
                ],
                value=tracker.config.config['supported_cryptocurrencies'][0]
            ),
            
            html.Label("Alert Settings:"),
            dcc.Input(
                id='price-target',
                type='number',
                placeholder='Target Price'
            ),
            dcc.Dropdown(
                id='condition-dropdown',
                options=[
                    {'label': 'Above', 'value': 'above'},
                    {'label': 'Below', 'value': 'below'}
                ],
                value='above'
            ),
            html.Button('Set Alert', id='set-alert-button'),
        ]),
        
        html.Div(id='price-display'),
        dcc.Graph(id='price-chart'),
        dcc.Interval(
            id='interval-component',
            interval=tracker.config.config['update_interval'] * 1000
        )
    ])

    @app.callback(
        Output('price-display', 'children'),
        Output('price-chart', 'figure'),
        Input('interval-component', 'n_intervals'),
        State('crypto-dropdown', 'value')
    )
    def update_data(n, selected_crypto):
        prices = tracker.fetch_prices([selected_crypto])
        if not prices:
            return "Error fetching prices", {}

        price = prices[selected_crypto]['usd']
        figure = {
            'data': [{
                'x': [datetime.now()],
                'y': [price],
                'type': 'scatter',
                'mode': 'lines+markers'
            }],
            'layout': {
                'title': f'{selected_crypto.capitalize()} Price'
            }
        }
        return f"Current Price: ${price:,.2f}", figure

    @app.callback(
        Output('set-alert-button', 'children'),
        Input('set-alert-button', 'n_clicks'),
        State('crypto-dropdown', 'value'),
        State('price-target', 'value'),
        State('condition-dropdown', 'value')
    )
    def set_alert(n_clicks, crypto, target_price, condition):
        if n_clicks and target_price:
            tracker.alert_handler.add_alert(crypto, target_price, condition)
            return 'Alert Set!'
        return 'Set Alert'

    return app

def main():
    """Main function"""
    try:
        tracker = CryptoTracker()
        app = create_dash_app(tracker)
        logger.info("Starting Cryptocurrency Price Tracker")
        app.run_server(debug=True)
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()