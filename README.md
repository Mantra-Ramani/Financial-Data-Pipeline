# Automated Financial Data Extraction Pipeline
A Python-based quantitative data engineering pipeline designed to extract, clean, and structure alternative financial data from the web. 

**Technologies:** Python, Pandas, BeautifulSoup, SQLite  

- Bypasses basic HTTP 403 blocks using custom user-agent headers to establish secure connections.
- Parses unstructured HTML tables to extract S&P 500 constituent data (tickers, sectors, and industries).
- Loads cleaned data into a local relational SQLite database (`financial_data.db`) for downstream quantitative modeling and time-series backtesting.
