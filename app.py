# config.json : Settings storage manager so that incase of anything we do not have to alter the whole code but just this settings. 
{
    "api_key": "CG-Nb934WDL84Tx8zZUdqyjJn4E",
    "supported_cryptocurrencies": ["bitcoin", "ethereum"], # Crypto currencies to be tracked by the app
    "supported_fiats": ["usd", "eur"], #The currecnies to be displayed or supported by the app.
    "update_interval": 60, # Interval (in seconds) at which the app updates the price data in real-time.
    "cache_duration": 60, # How long (in seconds) the cached data is considered valid before refetching.
    "retry_limit": 3
}

# crypto_tracker.py, importation of necessary libraries.
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
    def fetch_historical_data(
        self, 
        currency: str,
        days: int = 30,
        vs_currency: str = "usd",
        interval: str = "daily",
        include_volume: bool = True,
        include_market_cap: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical cryptocurrency data with enhanced features and error handling.
        
        Args:
            currency: The cryptocurrency to fetch data for (e.g., 'bitcoin')
            days: Number of days of historical data to fetch
            vs_currency: The currency to get prices in (e.g., 'usd')
            interval: Data interval ('daily', 'hourly', 'minutely')
            include_volume: Whether to include trading volume data
            include_market_cap: Whether to include market cap data
            
        Returns:
            DataFrame with timestamp and requested data columns, or None if fetch fails
        """
        if vs_currency not in self.config.config['supported_fiats']:
            raise ValueError(f"Unsupported fiat currency: {vs_currency}")
            
        if interval not in ['daily', 'hourly', 'minutely']:
            raise ValueError(f"Invalid interval: {interval}. Must be 'daily', 'hourly', or 'minutely'")
            
        # Validate days based on interval to prevent too many data points
        max_days = {
            'minutely': 1,    # Max 1 day for minute data
            'hourly': 90,     # Max 90 days for hourly data
            'daily': 365      # Max 365 days for daily data
        }
        
        if days > max_days[interval]:
            logger.warning(
                f"Requested days ({days}) exceeds maximum allowed for {interval} interval. "
                f"Limiting to {max_days[interval]} days."
            )
            days = max_days[interval]

        url = f"{self.base_url}/coins/{currency}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": days,
            "interval": interval
        }

        try:
            def fetch_data():
                response = requests.get(
                    url,
                    params=params,
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()

            # Use retry handler for the API request
            data = self.retry_handler.execute(fetch_data)
            
            # Validate response data
            if not data:
                logger.error(f"Empty response received for {currency}")
                return None
                
            if 'prices' not in data or not data['prices']:
                logger.warning(f"No price data available for {currency}")
                return None

            # Create the base DataFrame with prices
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add volume data if requested and available
            if include_volume and 'total_volumes' in data:
                volume_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                df = df.merge(volume_df, on='timestamp', how='left')

            # Add market cap data if requested and available
            if include_market_cap and 'market_caps' in data:
                market_cap_df = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
                market_cap_df['timestamp'] = pd.to_datetime(market_cap_df['timestamp'], unit='ms')
                df = df.merge(market_cap_df, on='timestamp', how='left')

            # Add metadata columns
            df['currency'] = currency
            df['vs_currency'] = vs_currency
            df['interval'] = interval
            
            # Ensure data is sorted by timestamp
            df = df.sort_values('timestamp')
            
            # Add percentage change columns
            df['price_change_pct'] = df['price'].pct_change() * 100
            if 'volume' in df.columns:
                df['volume_change_pct'] = df['volume'].pct_change() * 100
            if 'market_cap' in df.columns:
                df['market_cap_change_pct'] = df['market_cap'].pct_change() * 100

            # Add moving averages
            df['SMA_7'] = df['price'].rolling(window=7).mean()
            df['SMA_30'] = df['price'].rolling(window=30).mean()

            logger.info(
                f"Successfully fetched {len(df)} {interval} data points for {currency}"
                f" ({df['timestamp'].min()} to {df['timestamp'].max()})"
            )

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching data for {currency}: {e}")
            return None
        except ValueError as e:
            logger.error(f"Error parsing data for {currency}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {currency}: {e}")
            return None
        finally:
            # Log rate limit information if available
            if 'response' in locals():
                remaining = response.headers.get('X-RateLimit-Remaining')
                if remaining:
                    logger.info(f"Rate limit remaining: {remaining}")

def create_dash_app(tracker: CryptoTracker) -> Dash:
    """Create a beautiful and modern Dash web application"""
    app = Dash(
        __name__,
        external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
            'https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap'
        ]
    )
    
    # Add CSS for beautiful layout
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Cryptocurrency Price Tracker</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
                    color: white;
                    height: 100vh;
                    overflow: hidden;
                    font-family: 'Poppins', sans-serif;
                }
                #react-entry-point {
                    height: 100vh;
                }
                .main-container {
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                    padding: 20px;
                }
                .header {
                    text-align: center;
                    margin-bottom: 20px;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    background: linear-gradient(45deg, #00f260, #0575e6);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    animation: gradient 5s ease infinite;
                }
                @keyframes gradient {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                    100% { background-position: 0% 50%; }
                }
                .controls-container {
                    display: flex;
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .control-panel {
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 20px;
                    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    transition: transform 0.3s ease;
                }
                .control-panel:hover {
                    transform: translateY(-5px);
                }
                .price-display {
                    text-align: center;
                    margin-bottom: 20px;
                    padding: 20px;
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    animation: pulse 2s infinite;
                }
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.02); }
                    100% { transform: scale(1); }
                }
                .chart-container {
                    flex: 1;
                    min-height: 400px;
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                .alerts-container {
                    background: rgba(255, 255, 255, 0.05);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 20px;
                    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                .btn-primary {
                    background: linear-gradient(45deg, #00f260, #0575e6);
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    transition: all 0.3s ease;
                }
                .btn-primary:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(5, 117, 230, 0.4);
                }
                .Select-control, .Select-menu-outer {
                    background: rgba(255, 255, 255, 0.1) !important;
                    border: 1px solid rgba(255, 255, 255, 0.2) !important;
                    border-radius: 8px !important;
                    color: white !important;
                }
                .Select-value-label {
                    color: white !important;
                }
                .Select-option {
                    background: rgba(255, 255, 255, 0.1) !important;
                    color: white !important;
                }
                .Select-option:hover {
                    background: rgba(255, 255, 255, 0.2) !important;
                }
                input {
                    background: rgba(255, 255, 255, 0.1) !important;
                    border: 1px solid rgba(255, 255, 255, 0.2) !important;
                    border-radius: 8px !important;
                    color: white !important;
                    padding: 10px !important;
                }
                .alert {
                    background: rgba(255, 255, 255, 0.1) !important;
                    border: none !important;
                    color: white !important;
                    backdrop-filter: blur(10px);
                    transition: transform 0.3s ease;
                }
                .alert:hover {
                    transform: translateX(5px);
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Define the layout
    app.layout = html.Div(
        className="main-container",
        children=[
            # Header
            html.H1("Cryptocurrency Price Tracker", 
                   className="header text-center"),
            
            # Controls Container
            html.Div(
                className="controls-container",
                children=[
                    # Cryptocurrency Selection
                    html.Div(
                        className="control-panel flex-grow-1",
                        children=[
                            html.Label("Select Cryptocurrency:", 
                                     className="mb-2"),
                            dcc.Dropdown(
                                id='crypto-dropdown',
                                options=[
                                    {'label': c.capitalize(), 'value': c}
                                    for c in tracker.config.config['supported_cryptocurrencies']
                                ],
                                value=tracker.config.config['supported_cryptocurrencies'][0],
                                className="mb-2"
                            )
                        ]
                    ),
                    
                    # Alert Settings
                    html.Div(
                        className="control-panel flex-grow-1",
                        children=[
                            html.Label("Price Alert Settings:", 
                                     className="mb-2"),
                            html.Div(
                                className="d-flex gap-2",
                                children=[
                                    dcc.Input(
                                        id='price-target',
                                        type='number',
                                        placeholder='Target Price',
                                        className="flex-grow-1"
                                    ),
                                    dcc.Dropdown(
                                        id='condition-dropdown',
                                        options=[
                                            {'label': 'Above', 'value': 'above'},
                                            {'label': 'Below', 'value': 'below'}
                                        ],
                                        value='above',
                                        className="flex-grow-1"
                                    ),
                                    html.Button(
                                        'Set Alert',
                                        id='set-alert-button',
                                        className="btn btn-primary"
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Price Display
            html.Div(id='price-display', className="price-display"),
            
            # Chart Container
            html.Div(
                className="chart-container",
                children=[
                    dcc.Graph(
                        id='price-chart',
                        style={'height': '100%'},
                        config={'displayModeBar': True}
                    )
                ]
            ),
            
            # Alerts Container
            html.Div(
                className="alerts-container",
                children=[
                    html.H5("Active Alerts", className="mb-3"),
                    html.Div(id='alerts-display')
                ]
            ),
            
            # Interval Component
            dcc.Interval(
                id='interval-component',
                interval=tracker.config.config['update_interval'] * 1000
            )
        ]
    )

    @app.callback(
        [Output('price-display', 'children'),
         Output('price-chart', 'figure')],
        [Input('interval-component', 'n_intervals'),
         Input('crypto-dropdown', 'value')]
    )
    def update_data(n, selected_crypto):
        try:
            prices = tracker.fetch_prices([selected_crypto])
            if not prices:
                return html.Div("Error fetching prices"), {}
            
            price = prices[selected_crypto]['usd']
            price_change = prices[selected_crypto].get('usd_24h_change', 0)
            
            # Get historical data
            df = tracker.fetch_historical_data(selected_crypto)
            
            # Define figure layout
            figure_layout = {
                'template': 'plotly_dark',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'title': {
                    'text': f'{selected_crypto.capitalize()} Price History',
                    'font': {'family': 'Poppins', 'size': 24, 'color': 'white'}
                },
                'xaxis': {
                    'gridcolor': 'rgba(255,255,255,0.1)',
                    'color': 'white',
                    'title': {'text': 'Date', 'font': {'family': 'Poppins', 'color': 'white'}}
                },
                'yaxis': {
                    'gridcolor': 'rgba(255,255,255,0.1)',
                    'color': 'white',
                    'title': {'text': 'Price (USD)', 'font': {'family': 'Poppins', 'color': 'white'}}
                },
                'margin': {'t': 50, 'b': 50, 'l': 50, 'r': 50},
                'height': 500,
                'showlegend': True,
                'legend': {
                    'bgcolor': 'rgba(0,0,0,0)',
                    'font': {'family': 'Poppins', 'color': 'white'}
                }
            }

            # Create trace with gradient
            trace = {
                'x': df['timestamp'] if df is not None else [],
                'y': df['price'] if df is not None else [],
                'type': 'scatter',
                'mode': 'lines',
                'line': {
                    'color': '#00f260',
                    'width': 3,
                    'dash': 'solid'
                },
                'fill': 'tonexty',
                'fillcolor': 'rgba(0, 242, 96, 0.1)',
                'name': selected_crypto.capitalize()
            }

            # Combine trace and layout
            figure = {
                'data': [trace],
                'layout': figure_layout
            }
            
            # Format price display
            price_display = html.Div([
                html.Div(
                    f"${price:,.2f}",
                    style={
                        'fontSize': '48px',
                        'fontWeight': '600',
                        'background': 'linear-gradient(45deg, #00f260, #0575e6)',
                        'WebkitBackgroundClip': 'text',
                        'WebkitTextFillColor': 'transparent'
                    }
                ),
                html.Div(
                    f"24h Change: {price_change:+.2f}%",
                    style={
                        'color': '#00f260' if price_change >= 0 else '#ff4444',
                        'fontSize': '24px',
                        'fontWeight': '500',
                        'marginTop': '10px'
                    }
                )
            ])
            
            return price_display, figure
            
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return html.Div("Error updating data"), {}

    @app.callback(
        [Output('set-alert-button', 'children'),
         Output('alerts-display', 'children')],
        [Input('set-alert-button', 'n_clicks')],
        [State('crypto-dropdown', 'value'),
         State('price-target', 'value'),
         State('condition-dropdown', 'value')]
    )
    def handle_alerts(n_clicks, crypto, target_price, condition):
        if n_clicks and target_price:
            tracker.alert_handler.add_alert(crypto, target_price, condition)
        
        alerts = tracker.alert_handler.alerts
        alerts_display = html.Div([
            html.Div(
                f"{alert.crypto.capitalize()}: {alert.condition} ${alert.target_price:,.2f}",
                className="alert alert-info" if not alert.triggered else "alert alert-success",
                style={'marginBottom': '8px'}
            )
            for alert in alerts
        ]) if alerts else html.Div("No active alerts", className="text-muted")
        
        return 'Alert Set!' if n_clicks else 'Set Alert', alerts_display

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