# Importation of various libraries

import requests
from typing import Dict, Optional
import time


# Function to fetch for the crypto prices
def fetch_crypto_prices() -> Optional[Dict]:
    """This is a code function to fetch the current prices of the cryptocurriences from coinGecko API,
    upon a successful requests it wil return a dictionary that contains the price data or None if the request fails"""

    # The URL of the API
    url = "https://api.congecko.com/api/v3/simple/price"

    params = {
        "ids": "bitcoin, ethereum, litecoin, cardano, dogecoin",
        " vs_currencies": "usd",
        "include_24hr_change": True,
        "include_market_cap": True 
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Checking if we hit the rate limit
        if response.status_code ==429:
            print("Rate limit hit. Warning before retry...")
            time.sleep(60)
            return fetch_crypto_prices()
        
        data = response.json()
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing JSON response: {e}")
        return None
    
def format_price(price: float) -> str:
        """Format price with appropriate decimal places based on value"""
        if price >= 1000:
            return f"${price:,.2f}"
        elif price >= 1:
            return f"${price:.4f}"
        else:
            return f"${price:> 6f}"
        
if __name__ == "__main__":
            prices = fetch_crypto_prices()
            if prices:
                print("\nLIve Cryptocurrency Prices:")
                print("-" *50)
                for crypto, into in prices.items():
                    price = info['usd']
                    change = info.get('usd_24th_change', 0)
                    market_cap = info.get('usd_market_cap', 0)

                    print(f"{crypto.capitalize()}:")
                    print(f" Price: {format_price(price)}")
                    print (f" 24h Change: {change:+.2f}%")
                    print(f" Market Cap: ${market_cap:,. 0f}")
                    print("_" * 50)


