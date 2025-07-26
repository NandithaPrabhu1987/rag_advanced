"""
Dummy API Server for RAG Demonstration
=====================================

This module provides mock API endpoints that simulate real-world data sources
for demonstration purposes.
"""

import json
import time
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List
import threading
import datetime

# Mock data store
STOCK_DATA = {
    "AAPL": {
        "name": "Apple Inc.",
        "sector": "Technology",
        "price_history": [150.25, 151.30, 149.80, 152.45, 153.60],
        "market_cap": "2.5 trillion USD",
        "pe_ratio": 28.5,
        "dividend_yield": 0.6,
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "sector": "Technology",
        "price_history": [290.10, 292.40, 295.60, 291.20, 298.75],
        "market_cap": "2.3 trillion USD",
        "pe_ratio": 32.1,
        "dividend_yield": 0.8,
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "sector": "Consumer Discretionary",
        "price_history": [130.50, 132.20, 128.90, 133.60, 135.20],
        "market_cap": "1.4 trillion USD",
        "pe_ratio": 40.2,
        "dividend_yield": 0.0,
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "sector": "Communication Services",
        "price_history": [135.20, 136.80, 134.50, 138.90, 140.10],
        "market_cap": "1.8 trillion USD",
        "pe_ratio": 25.3,
        "dividend_yield": 0.5,
    },
    "TSLA": {
        "name": "Tesla, Inc.",
        "sector": "Consumer Discretionary",
        "price_history": [180.30, 185.60, 178.20, 190.40, 195.80],
        "market_cap": "600 billion USD",
        "pe_ratio": 52.8,
        "dividend_yield": 0.0,
    }
}

NEWS_DATA = [
    {
        "id": "n001",
        "title": "Tech Giants Report Strong Quarterly Earnings",
        "source": "Business Insider",
        "date": "2025-07-25",
        "content": "Major technology companies reported better-than-expected earnings for the second quarter of 2025. Apple, Microsoft, and Google all exceeded analyst expectations, driven by strong cloud services performance and increasing device sales. Amazon showed particular strength in AWS growth, while Tesla reported record vehicle deliveries despite supply chain challenges."
    },
    {
        "id": "n002",
        "title": "New AI Regulations Proposed in European Union",
        "source": "Tech Policy Journal",
        "date": "2025-07-24",
        "content": "The European Commission has proposed new regulations governing artificial intelligence applications across member states. The comprehensive framework includes stricter requirements for high-risk AI systems, particularly those used in healthcare, transportation, and financial services. Industry leaders have expressed concerns about compliance costs, while privacy advocates welcome the enhanced protections for consumer data."
    },
    {
        "id": "n003",
        "title": "Global Chip Shortage Showing Signs of Easing",
        "source": "Manufacturing Weekly",
        "date": "2025-07-23",
        "content": "After nearly three years of disruption, the global semiconductor shortage appears to be easing according to industry analysts. New manufacturing capacity coming online in Taiwan, South Korea, and the United States has helped alleviate supply constraints. Consumer electronics manufacturers report improving inventory levels, though automotive production still faces some challenges in securing specialized chips."
    }
]

WEATHER_DATA = {
    "New York": {
        "condition": "Partly Cloudy",
        "temperature": 28,
        "humidity": 65,
        "wind": "10 km/h NE",
        "forecast": [
            {"day": "Tomorrow", "condition": "Sunny", "high": 30, "low": 22},
            {"day": "Day after", "condition": "Scattered Showers", "high": 27, "low": 21}
        ]
    },
    "London": {
        "condition": "Rainy",
        "temperature": 19,
        "humidity": 80,
        "wind": "15 km/h SW",
        "forecast": [
            {"day": "Tomorrow", "condition": "Light Rain", "high": 20, "low": 16},
            {"day": "Day after", "condition": "Overcast", "high": 21, "low": 15}
        ]
    },
    "Tokyo": {
        "condition": "Clear",
        "temperature": 31,
        "humidity": 70,
        "wind": "8 km/h SE",
        "forecast": [
            {"day": "Tomorrow", "condition": "Humid", "high": 32, "low": 26},
            {"day": "Day after", "condition": "Thunderstorms", "high": 30, "low": 25}
        ]
    },
    "Sydney": {
        "condition": "Sunny",
        "temperature": 22,
        "humidity": 55,
        "wind": "12 km/h E",
        "forecast": [
            {"day": "Tomorrow", "condition": "Sunny", "high": 23, "low": 14},
            {"day": "Day after", "condition": "Partly Cloudy", "high": 21, "low": 15}
        ]
    }
}

# Regularly update stock prices to simulate real-time data
def update_stock_prices():
    while True:
        for ticker in STOCK_DATA:
            last_price = STOCK_DATA[ticker]["price_history"][-1]
            # Random price movement between -2% and +2%
            change_pct = random.uniform(-0.02, 0.02)
            new_price = round(last_price * (1 + change_pct), 2)
            STOCK_DATA[ticker]["price_history"].append(new_price)
            # Keep only the last 10 prices
            if len(STOCK_DATA[ticker]["price_history"]) > 10:
                STOCK_DATA[ticker]["price_history"] = STOCK_DATA[ticker]["price_history"][-10:]
        time.sleep(60)  # Update every minute

# Add occasional news updates
def update_news():
    while True:
        if random.random() < 0.2:  # 20% chance of new article every update cycle
            new_id = f"n{len(NEWS_DATA) + 1:03d}"
            topics = [
                "AI Innovation Accelerates in Healthcare Sector",
                "Major Cybersecurity Breach Affects Financial Institutions",
                "Renewable Energy Investments Reach Record High",
                "New Consumer Technology Trends Emerge Post-Pandemic",
                "Supply Chain Resilience Strategies Shift Global Trade Patterns"
            ]
            sources = ["Tech Review", "Financial Times", "Industry Insights", "Global Business Report", "Market Analysis Today"]
            today = datetime.date.today().strftime("%Y-%m-%d")
            
            # Generate a random article with title from topics
            title = random.choice(topics)
            source = random.choice(sources)
            content = f"In a significant development today, experts report that {title.lower()}. Industry analysts suggest this could have far-reaching implications for market dynamics and competitive positioning in the coming quarters. Several leading companies have already announced strategic initiatives in response to these changing conditions."
            
            new_article = {
                "id": new_id,
                "title": title,
                "source": source,
                "date": today,
                "content": content
            }
            NEWS_DATA.append(new_article)
            print(f"Added new article: {title}")
            
            # Keep only the latest 20 news items
            if len(NEWS_DATA) > 20:
                NEWS_DATA.pop(0)
                
        time.sleep(300)  # Check for news updates every 5 minutes

# Weather varies throughout the day
def update_weather():
    while True:
        for city in WEATHER_DATA:
            # Small random temperature fluctuations
            WEATHER_DATA[city]["temperature"] += random.uniform(-1.0, 1.0)
            WEATHER_DATA[city]["temperature"] = round(WEATHER_DATA[city]["temperature"], 1)
            
            # Randomly update conditions occasionally
            if random.random() < 0.1:  # 10% chance per update
                conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Heavy Rain", "Thunderstorms", "Clear"]
                WEATHER_DATA[city]["condition"] = random.choice(conditions)
                
            # Update humidity within reasonable bounds
            WEATHER_DATA[city]["humidity"] += random.uniform(-5, 5)
            WEATHER_DATA[city]["humidity"] = max(40, min(95, WEATHER_DATA[city]["humidity"]))
            WEATHER_DATA[city]["humidity"] = round(WEATHER_DATA[city]["humidity"])
            
        time.sleep(600)  # Update every 10 minutes

class DummyAPIHandler(BaseHTTPRequestHandler):
    """Handler for dummy API requests"""
    
    def _set_headers(self, content_type="application/json"):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
    def do_GET(self):
        """Handle GET requests"""
        # Stock market data endpoint
        if self.path.startswith('/api/stocks'):
            parts = self.path.split('/')
            if len(parts) > 3:
                ticker = parts[3]
                if ticker in STOCK_DATA:
                    self._set_headers()
                    response = {
                        "ticker": ticker,
                        "name": STOCK_DATA[ticker]["name"],
                        "sector": STOCK_DATA[ticker]["sector"],
                        "current_price": STOCK_DATA[ticker]["price_history"][-1],
                        "previous_close": STOCK_DATA[ticker]["price_history"][-2] if len(STOCK_DATA[ticker]["price_history"]) > 1 else None,
                        "market_cap": STOCK_DATA[ticker]["market_cap"],
                        "pe_ratio": STOCK_DATA[ticker]["pe_ratio"],
                        "dividend_yield": STOCK_DATA[ticker]["dividend_yield"],
                        "last_updated": datetime.datetime.now().isoformat()
                    }
                    self.wfile.write(json.dumps(response).encode())
                    return
                
            # Return all stocks if no specific ticker or invalid ticker
            self._set_headers()
            response = {}
            for ticker, data in STOCK_DATA.items():
                response[ticker] = {
                    "name": data["name"],
                    "sector": data["sector"],
                    "current_price": data["price_history"][-1],
                    "previous_close": data["price_history"][-2] if len(data["price_history"]) > 1 else None
                }
            self.wfile.write(json.dumps(response).encode())
        
        # News data endpoint
        elif self.path.startswith('/api/news'):
            parts = self.path.split('/')
            if len(parts) > 3:
                news_id = parts[3]
                for article in NEWS_DATA:
                    if article["id"] == news_id:
                        self._set_headers()
                        self.wfile.write(json.dumps(article).encode())
                        return
                
                # News not found
                self.send_response(404)
                self.end_headers()
                return
                
            # Return all news if no specific ID
            self._set_headers()
            self.wfile.write(json.dumps({
                "articles": NEWS_DATA,
                "count": len(NEWS_DATA),
                "last_updated": datetime.datetime.now().isoformat()
            }).encode())
        
        # Weather data endpoint
        elif self.path.startswith('/api/weather'):
            parts = self.path.split('/')
            if len(parts) > 3:
                city = parts[3].replace('%20', ' ')  # Handle spaces in URLs
                if city in WEATHER_DATA:
                    self._set_headers()
                    response = WEATHER_DATA[city].copy()
                    response["city"] = city
                    response["last_updated"] = datetime.datetime.now().isoformat()
                    self.wfile.write(json.dumps(response).encode())
                    return
                    
                # City not found
                self.send_response(404)
                self.end_headers()
                return
                
            # Return all cities if no specific city
            self._set_headers()
            response = {}
            for city, data in WEATHER_DATA.items():
                response[city] = {
                    "condition": data["condition"],
                    "temperature": data["temperature"],
                }
            response["last_updated"] = datetime.datetime.now().isoformat()
            self.wfile.write(json.dumps(response).encode())
            
        else:
            # Default API info endpoint
            self._set_headers()
            response = {
                "name": "Dummy API Server for RAG Demo",
                "version": "1.0",
                "endpoints": [
                    {"path": "/api/stocks", "description": "Stock market data"},
                    {"path": "/api/stocks/{ticker}", "description": "Data for a specific stock"},
                    {"path": "/api/news", "description": "Latest news articles"},
                    {"path": "/api/news/{id}", "description": "Specific news article"},
                    {"path": "/api/weather", "description": "Weather for all cities"},
                    {"path": "/api/weather/{city}", "description": "Weather for a specific city"}
                ],
                "status": "operational"
            }
            self.wfile.write(json.dumps(response).encode())

def start_server():
    """Start the API server on port 8000"""
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, DummyAPIHandler)
    print("Starting API server on port 8000...")
    
    # Start data update threads
    stock_thread = threading.Thread(target=update_stock_prices)
    stock_thread.daemon = True
    stock_thread.start()
    
    news_thread = threading.Thread(target=update_news)
    news_thread.daemon = True
    news_thread.start()
    
    weather_thread = threading.Thread(target=update_weather)
    weather_thread.daemon = True
    weather_thread.start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.")

if __name__ == "__main__":
    start_server()
