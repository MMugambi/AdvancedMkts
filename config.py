

CSV_FILE_PATH = "AugSept copy.csv"

# You can also set multiple file paths and switch between them
ALTERNATIVE_PATHS = {
    "august_september": "AugSept copy.csv",
    "full_year": "full_year_trades.csv",
    "q1_data": "Q1_2025.csv",
}

# Display settings
SHOW_DEBUG_INFO = False
AUTO_RELOAD_ON_START = False

 
TOXICITY_THRESHOLD_SECONDS = 60  # Trades held for less than this are considered "toxic"

CHART_THEME = "plotly"  # Options: "plotly", "plotly_white", "plotly_dark"