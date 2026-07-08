"""Company tickers organized by industry sector — 32 companies total."""

from __future__ import annotations

# ── Technology (12) ───────────────────────────────────────────────
tech_companies = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # NVIDIA
    "GOOGL",  # Alphabet (Google)
    "META",   # Meta (Facebook)
    "AMZN",   # Amazon
    "TSLA",   # Tesla
    "AMD",    # Advanced Micro Devices
    "INTC",   # Intel
    "CRM",    # Salesforce
    "ADBE",   # Adobe
    "ORCL",   # Oracle
]

# ── Consumer / Retail (8) ─────────────────────────────────────────
consumer_companies = [
    "WMT",    # Walmart
    "COST",   # Costco
    "HD",     # Home Depot
    "MCD",    # McDonald's
    "NKE",    # Nike
    "SBUX",   # Starbucks
    "TGT",    # Target
    "LOW",    # Lowe's
]

# ── Finance (7) ───────────────────────────────────────────────────
finance_companies = [
    "JPM",    # JPMorgan Chase
    "BAC",    # Bank of America
    "GS",     # Goldman Sachs
    "MS",     # Morgan Stanley
    "V",      # Visa
    "MA",     # Mastercard
    "BRK-B",  # Berkshire Hathaway
]

# ── Healthcare / Pharma (5) ───────────────────────────────────────
healthcare_companies = [
    "JNJ",    # Johnson & Johnson
    "PFE",    # Pfizer
    "UNH",    # UnitedHealth
    "ABBV",   # AbbVie
    "MRK",    # Merck
    "LLY",    # Eli Lilly
]

# ── Full master list ──────────────────────────────────────────────
ALL_COMPANIES = (
    tech_companies
    + consumer_companies
    + finance_companies
    + healthcare_companies
)

# Quick lookup: sector for each ticker
SECTOR_MAP: dict[str, str] = {}
for t in tech_companies:
    SECTOR_MAP[t] = "Technology"
for t in consumer_companies:
    SECTOR_MAP[t] = "Consumer"
for t in finance_companies:
    SECTOR_MAP[t] = "Finance"
for t in healthcare_companies:
    SECTOR_MAP[t] = "Healthcare"
