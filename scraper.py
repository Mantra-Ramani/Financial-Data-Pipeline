"""
S&P 500 constituent scraper.

Pulls the constituents table from Wikipedia, validates it, and stores it in a
local SQLite database (financial_data.db). backtester.py reads the sector
information from this database.

Usage:
    python scraper.py
"""

import sqlite3
import urllib.request
from datetime import datetime, timezone

import pandas as pd
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DB_PATH = "financial_data.db"
TABLE_NAME = "sp500_companies"

# Wikipedia asks scrapers to identify themselves. Replace with your own email.
USER_AGENT = "StudentResearch/1.0 (mantraramani178@gmail.com)"


def fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_constituents(html: str) -> pd.DataFrame:
    """Parse the S&P 500 constituents table into a DataFrame."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise RuntimeError("Constituents table not found; page layout may have changed.")

    def norm(text: str) -> str:
        # lowercase and drop spaces/hyphens so "GICS sector", "GICSsector" and
        # "GICS Sub-Industry" / "gicssubindustry" all compare equal
        return "".join(ch for ch in text.lower() if ch.isalnum())

    rows = table.find_all("tr")
    headers = [norm(th.get_text(strip=True)) for th in rows[0].find_all("th")]

    records = []
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) == len(headers):
            records.append(dict(zip(headers, cells)))

    raw = pd.DataFrame(records)

    needed = {"symbol", "security", "gicssector", "gicssubindustry"}
    missing = needed - set(raw.columns)
    if missing:
        raise RuntimeError(f"Missing columns {missing}. Headers found: {list(raw.columns)}")

    df = pd.DataFrame(
        {
            "Ticker": raw["symbol"],
            "Company": raw["security"],
            "Sector": raw["gicssector"],
            "SubIndustry": raw["gicssubindustry"],
            "DateAdded": raw.get("dateadded"),
        }
    )
    # Wikipedia uses BRK.B, yfinance expects BRK-B.
    df["YahooTicker"] = df["Ticker"].str.replace(".", "-", regex=False)
    df["ScrapedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df


def validate(df: pd.DataFrame) -> None:
    """Basic data-quality checks. Raises AssertionError if something looks wrong."""
    assert 480 <= len(df) <= 520, f"Unexpected row count: {len(df)}"
    assert df["Ticker"].is_unique, "Duplicate tickers found"
    for col in ["Ticker", "Company", "Sector"]:
        assert df[col].notna().all() and (df[col] != "").all(), f"Missing values in {col}"
    assert df["Sector"].nunique() == 11, f"Expected 11 GICS sectors, got {df['Sector'].nunique()}"


def save_to_sqlite(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)


def main() -> None:
    print(f"Fetching {URL} ...")
    df = parse_constituents(fetch_html(URL))
    validate(df)
    print(f"Parsed and validated {len(df)} companies across {df['Sector'].nunique()} sectors.")
    print(df.head())
    save_to_sqlite(df)
    print(f"Saved to {DB_PATH} (table '{TABLE_NAME}').")


if __name__ == "__main__":
    main()
