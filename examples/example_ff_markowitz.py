from kiteconnect import KiteConnect
from zerodha_portfolio import generate_fama_french_markowitz_report

api_key = "your api key"
access_token = "your api secrect"

kite = KiteConnect(api_key=api_key, timeout=30)
kite.set_access_token(access_token)

candidate_portfolio = {
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

result = generate_fama_french_markowitz_report(
    kite=kite,
    candidate_quantities=candidate_portfolio,
    start_date="2022-01-01",
    end_date="2026-02-28",
    frequency="daily",
    output_file_name="ff_markowitz_report",
    # Small + Value + Momentum tilt
    preference={"beta_smb": -1.0, "beta_hml": 1.0, "beta_wml": 1.0},
    top_n=12,
    risk_free_rate=0.06,
    # If the site link format changes, you can bypass download by pointing to a local CSV:
    # factor_file_path=r"C:\Users\aksha\Downloads\2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv",
)

print("Report written:", result["output_html"])
print("FF factor file:", result["ff_factor_file"])
print("FF max date:", result["ff_factor_max_date"])
print("Effective end date used:", result["ff_effective_end_date"])
print("FF-selected stocks:", list(result["ff_selected_weights"].keys()))
print("FF tilt analysis:", result["ff_tilt_analysis"])
