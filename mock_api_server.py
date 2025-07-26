"""
Mock API Server for Real-Time RAG demonstrations

This module provides mock API endpoints that can be used with the real-time RAG agent
to demonstrate integration with external data sources.
"""

import os
import json
import random
import datetime
from typing import Dict, List, Any, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Sample data for our mock APIs
STOCK_DATA = {
    "AAPL": {
        "name": "Apple Inc.",
        "sector": "Technology",
        "current_price": 175.34,
        "previous_close": 174.21
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "sector": "Technology",
        "current_price": 329.56,
        "previous_close": 331.05
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "sector": "Technology",
        "current_price": 151.28,
        "previous_close": 150.49
    },
    "AMZN": {
        "name": "Amazon.com, Inc.",
        "sector": "Consumer Cyclical",
        "current_price": 172.03,
        "previous_close": 171.96
    },
    "META": {
        "name": "Meta Platforms, Inc.",
        "sector": "Technology",
        "current_price": 489.01,
        "previous_close": 486.18
    }
}

NEWS_DATA = {
    "articles": [
        {
            "id": "news-001",
            "title": "Researchers Achieve New Breakthrough in Quantum Computing",
            "source": "Tech Today",
            "date": "2023-07-15",
            "content": "Scientists at the Quantum Research Institute have developed a new method for stabilizing qubits, potentially solving one of the biggest challenges in quantum computing. This breakthrough could lead to more reliable quantum computers that can maintain quantum states for longer periods, enabling more complex calculations. The team demonstrated a 10x improvement in coherence time compared to previous approaches."
        },
        {
            "id": "news-002",
            "title": "AI Model Predicts Climate Change Impact on Agriculture",
            "source": "Science Daily",
            "date": "2023-07-14",
            "content": "A new AI model developed by researchers at the Climate Research Center can predict crop yields under different climate scenarios with unprecedented accuracy. The model uses satellite imagery, weather data, and historical crop production data to help farmers adapt to changing conditions. Initial tests show the model can predict yields with 95% accuracy up to six months in advance."
        },
        {
            "id": "news-003",
            "title": "New Programming Language Designed for Biological Computing",
            "source": "Dev Weekly",
            "date": "2023-07-13",
            "content": "Researchers have unveiled BioLang, a new programming language specifically designed for computational biology and biological computing. BioLang makes it easier to model complex biological systems and design DNA computing experiments. The language includes native support for concepts like protein folding, genetic circuits, and cellular automata."
        }
    ]
}

WEATHER_DATA = {
    "New York": {
        "condition": "Partly Cloudy",
        "temperature": 24,
        "humidity": 65,
        "wind": 10
    },
    "London": {
        "condition": "Rainy",
        "temperature": 18,
        "humidity": 80,
        "wind": 15
    },
    "Tokyo": {
        "condition": "Sunny",
        "temperature": 29,
        "humidity": 70,
        "wind": 8
    },
    "Sydney": {
        "condition": "Clear",
        "temperature": 22,
        "humidity": 60,
        "wind": 12
    },
    "last_updated": datetime.datetime.now().isoformat()
}

# Helper to randomize stock prices slightly
def update_stock_prices():
    for ticker in STOCK_DATA:
        # Store previous close
        STOCK_DATA[ticker]["previous_close"] = STOCK_DATA[ticker]["current_price"]
        # Update current price with random movement (-2% to +2%)
        movement = random.uniform(-0.02, 0.02)
        STOCK_DATA[ticker]["current_price"] = round(
            STOCK_DATA[ticker]["current_price"] * (1 + movement), 2
        )

# API endpoints
@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    """Return current stock data"""
    update_stock_prices()  # Update with random movement
    return jsonify(STOCK_DATA)

@app.route('/api/news', methods=['GET'])
def get_news():
    """Return latest news articles"""
    # Could randomly update news but keeping static for demo
    return jsonify(NEWS_DATA)

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """Return current weather data"""
    # Update last updated timestamp
    WEATHER_DATA["last_updated"] = datetime.datetime.now().isoformat()
    
    # Randomly adjust temperatures slightly
    for city in WEATHER_DATA:
        if city != "last_updated":
            # Random temperature adjustment (-1 to +1 degree)
            WEATHER_DATA[city]["temperature"] += random.randint(-1, 1)
    
    return jsonify(WEATHER_DATA)

# Status endpoint
@app.route('/status', methods=['GET'])
def status():
    """API status check endpoint"""
    return jsonify({
        "status": "running",
        "timestamp": datetime.datetime.now().isoformat(),
        "endpoints": ["/api/stocks", "/api/news", "/api/weather"]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
    print(f"Mock API server running on port {port}")
