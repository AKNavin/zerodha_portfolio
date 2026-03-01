from kiteconnect import KiteConnect
from zerodha_portfolio import generate_markowitz_report

api_key = "your api key"
access_token = "your api secrect"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

portfolio = {
    "RELIANCE": 18,
    "TCS": 24,
    "INFY": 32,
    "HDFCBANK": 20,
    "ICICIBANK": 28,
    "SBIN": 45,
    "LT": 14,
    "ITC": 70,
    "HINDUNILVR": 12,
    "BHARTIARTL": 26,
    "KOTAKBANK": 16,
    "AXISBANK": 22,
    "BAJFINANCE": 8,
    "ASIANPAINT": 11,
    "MARUTI": 6,
    "SUNPHARMA": 18,
    "TITAN": 10,
    "NESTLEIND": 4,
    "ULTRACEMCO": 5,
    "M&M": 15,
    "WIPRO": 40,
    "NTPC": 52,
    "ONGC": 60,
    "POWERGRID": 58,
    "ADANIPORTS": 17,
    "TMPV": 34,
    "BAJAJFINSV": 9,
    "HCLTECH": 19,
    "COALINDIA": 48,
    "TECHM": 21,
    "INDUSINDBK": 13,
    "JSWSTEEL": 27,
    "TATASTEEL": 42,
    "DRREDDY": 7,
    "CIPLA": 23,
    "BRITANNIA": 6,
    "APOLLOHOSP": 5,
    "EICHERMOT": 4,
    "GRASIM": 12,
    "HEROMOTOCO": 9,
}

result = generate_markowitz_report(
    kite=kite,
    portfolio_quantities=portfolio,
    start_date="2022-01-01",
    end_date="2026-02-28",
    output_file_name="markowitz_report",
    risk_free_rate=0.06,
)

print("Report written:", result["output_html"])
print("Tangency Weights:", result["tangency_weights"])
