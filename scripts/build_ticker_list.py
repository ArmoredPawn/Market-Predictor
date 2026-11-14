"""Build the list of US stocks and ETFs that the website lets people pick from.

Source: the public "US-Stock-Symbols" dataset (github.com/rreichel3/US-Stock-Symbols),
which is rebuilt daily from the NASDAQ, NYSE and AMEX listings and includes
each company's home country.

Rules:
  * keep only companies whose country is "United States" (this drops foreign
    companies that also trade in the US, such as Alibaba or Toyota ADRs)
  * drop preferred shares, warrants, units, rights and notes
  * add a hand-picked list of popular ETFs that hold US stocks
  * sort by market value so the biggest, best-known names come first

Run from the project root:   python scripts/build_ticker_list.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_US = ROOT / "data" / "us_tickers.csv"
OUT_FOREIGN = ROOT / "data" / "non_us_tickers.csv"
SOURCE = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/{ex}/{ex}_full_tickers.json"
EXCHANGES = ("nasdaq", "nyse", "amex")

# Popular ETFs whose holdings are US stocks. (Symbol, plain-English name, category)
US_EQUITY_ETFS: list[tuple[str, str, str]] = [
    ("SPY", "SPDR S&P 500 ETF (the 500 largest US companies)", "Broad market"),
    ("VOO", "Vanguard S&P 500 ETF", "Broad market"),
    ("IVV", "iShares Core S&P 500 ETF", "Broad market"),
    ("VTI", "Vanguard Total Stock Market ETF (almost every US stock)", "Broad market"),
    ("ITOT", "iShares Core S&P Total U.S. Stock Market ETF", "Broad market"),
    ("SCHB", "Schwab U.S. Broad Market ETF", "Broad market"),
    ("QQQ", "Invesco QQQ (Nasdaq-100: big tech-heavy companies)", "Broad market"),
    ("QQQM", "Invesco Nasdaq 100 ETF", "Broad market"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF", "Broad market"),
    ("RSP", "Invesco S&P 500 Equal Weight ETF", "Broad market"),
    ("IWM", "iShares Russell 2000 ETF (small companies)", "Size"),
    ("IJR", "iShares Core S&P Small-Cap ETF", "Size"),
    ("VB", "Vanguard Small-Cap ETF", "Size"),
    ("IJH", "iShares Core S&P Mid-Cap ETF", "Size"),
    ("MDY", "SPDR S&P MidCap 400 ETF", "Size"),
    ("VO", "Vanguard Mid-Cap ETF", "Size"),
    ("VUG", "Vanguard Growth ETF", "Style"),
    ("VTV", "Vanguard Value ETF", "Style"),
    ("SCHD", "Schwab U.S. Dividend Equity ETF", "Style"),
    ("VIG", "Vanguard Dividend Appreciation ETF", "Style"),
    ("USMV", "iShares MSCI USA Min Vol Factor ETF", "Style"),
    ("QUAL", "iShares MSCI USA Quality Factor ETF", "Style"),
    ("MTUM", "iShares MSCI USA Momentum Factor ETF", "Style"),
    ("XLK", "Technology Select Sector SPDR Fund", "Sector"),
    ("XLF", "Financial Select Sector SPDR Fund", "Sector"),
    ("XLV", "Health Care Select Sector SPDR Fund", "Sector"),
    ("XLE", "Energy Select Sector SPDR Fund", "Sector"),
    ("XLY", "Consumer Discretionary Select Sector SPDR Fund", "Sector"),
    ("XLP", "Consumer Staples Select Sector SPDR Fund", "Sector"),
    ("XLI", "Industrial Select Sector SPDR Fund", "Sector"),
    ("XLU", "Utilities Select Sector SPDR Fund", "Sector"),
    ("XLB", "Materials Select Sector SPDR Fund", "Sector"),
    ("XLRE", "Real Estate Select Sector SPDR Fund", "Sector"),
    ("XLC", "Communication Services Select Sector SPDR Fund", "Sector"),
    ("VNQ", "Vanguard Real Estate ETF", "Sector"),
    ("SMH", "VanEck Semiconductor ETF", "Sector"),
    ("SOXX", "iShares Semiconductor ETF", "Sector"),
    ("ARKK", "ARK Innovation ETF", "Thematic"),
]

EXCLUDE_NAME = re.compile(
    r"warrant|\bunits?\b|\brights?\b|preferred|depositary|\bnotes?\b|debenture|%|"
    r"subordinated|trust preferred|contingent value",
    re.IGNORECASE,
)
NAME_SUFFIX = re.compile(
    r"\s*(common stock|common shares|capital stock|ordinary shares|"
    r"class [a-c] common stock|class [a-c] ordinary shares)\s*$",
    re.IGNORECASE,
)


# "Alibaba Group Holding Limited American Depositary Shares each ..." -> "Alibaba Group Holding Limited"
FOREIGN_NAME_TAIL = re.compile(
    r"\s+(American Depositary|American Depository|ADS\b|Ordinary|Class [A-Z]\b|Common|Subordinate|Sponsored|"
    r"Each representing|Representing|New York Registry|Limited Voting|Exchangeable|Shares\b).*$",
    re.IGNORECASE,
)


def clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name)).strip()
    # "Alphabet Inc. Class A Common Stock" -> "Alphabet Inc. Class A"
    m = re.match(r"^(.*?)(\s+Class [A-C])?\s+(Common Stock|Common Shares|Capital Stock|Ordinary Shares)\s*$", name, re.I)
    if m:
        name = (m.group(1) + (m.group(2) or "")).strip()
    return NAME_SUFFIX.sub("", name).strip() or name


def to_yahoo_symbol(symbol: str) -> str:
    """'BRK/B' (exchange style) -> 'BRK-B' (Yahoo Finance style)."""
    return symbol.strip().upper().replace("/", "-").replace(".", "-")


def download(exchange: str) -> list[dict]:
    url = SOURCE.format(ex=exchange)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URL
        return json.loads(resp.read().decode("utf-8"))


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for ex in EXCHANGES:
        for r in download(ex):
            r["exchange"] = ex.upper()
            rows.append(r)
    raw = pd.DataFrame(rows)
    raw["symbol"] = raw["symbol"].astype(str).str.strip()
    raw = raw[~raw["symbol"].str.contains(r"\^", regex=True)]  # preferred shares
    raw["market_cap"] = pd.to_numeric(raw["marketCap"], errors="coerce")
    raw["country"] = raw["country"].fillna("").astype(str).str.strip()
    raw["yahoo"] = raw["symbol"].map(to_yahoo_symbol)

    usable = raw[~raw["name"].astype(str).str.contains(EXCLUDE_NAME)]
    us = usable[(usable["country"] == "United States") & (usable["market_cap"] > 0)].copy()
    us["name"] = us["name"].map(clean_name)
    us["sector"] = us["sector"].fillna("").replace("", "Other")
    us["kind"] = "Stock"
    us = us.sort_values("market_cap", ascending=False).drop_duplicates("yahoo")
    us = us[["yahoo", "name", "exchange", "sector", "kind", "market_cap"]].rename(columns={"yahoo": "symbol"})

    etfs = pd.DataFrame(US_EQUITY_ETFS, columns=["symbol", "name", "sector"])
    etfs["exchange"] = "ETF"
    etfs["kind"] = "ETF"
    etfs["market_cap"] = float("nan")
    etfs = etfs[us.columns]

    # Big stocks first, then the core index ETFs, then everything else.
    combined = pd.concat([us.head(25), etfs, us.iloc[25:]], ignore_index=True)
    combined = combined.drop_duplicates("symbol")

    foreign = raw[(raw["country"] != "United States") & (raw["country"] != "")]
    foreign = foreign[~foreign["yahoo"].isin(combined["symbol"])]
    short = foreign["name"].map(clean_name).str.replace(FOREIGN_NAME_TAIL, "", regex=True).str.strip(" ,-")
    foreign = foreign.assign(name=short)[["yahoo", "name", "country"]]
    foreign = foreign.rename(columns={"yahoo": "symbol"}).drop_duplicates("symbol").sort_values("symbol")
    return combined, foreign


def main() -> int:
    us, foreign = build()
    if len(us) < 2000:
        print(f"Refusing to write: only {len(us)} US securities found (source may be broken).")
        return 1
    OUT_US.parent.mkdir(parents=True, exist_ok=True)
    us.to_csv(OUT_US, index=False, float_format="%.0f")
    foreign.to_csv(OUT_FOREIGN, index=False)
    print(f"Wrote {len(us):,} US securities to {OUT_US.relative_to(ROOT)}")
    print(f"Wrote {len(foreign):,} non-US securities to {OUT_FOREIGN.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
