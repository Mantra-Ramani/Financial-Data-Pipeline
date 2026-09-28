import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3

print("Initializing Automated Data Extraction Pipeline...")

# 1. Fetch unstructured HTML data directly from the web
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
# 1. Fetch unstructured HTML data directly from the web
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
print(f"Connecting to {url}...")

# Create a fake browser header to bypass Wikipedia's 403 Forbidden security block
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
req = urllib.request.Request(url, headers=headers)
html = urllib.request.urlopen(req).read()
# 2. Parse the HTML using BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')
table = soup.find('table', {'id': 'constituents'})

# 3. Extract the specific table rows
tickers = []
companies = []
sectors = []

print("Parsing raw HTML and cleaning data...")
for row in table.find_all('tr')[1:]: # Skip the header row
    cols = row.find_all('td')
    tickers.append(cols[0].text.strip())
    companies.append(cols[1].text.strip())
    sectors.append(cols[2].text.strip())

# 4. Load into a structured Pandas DataFrame
df = pd.DataFrame({
    'Ticker': tickers,
    'Company': companies,
    'Sector': sectors
})

print("\n--- Extraction Successful! First 5 rows ---")
print(df.head())

# 5. Load structured dataset into a relational SQL database
# (SQLite is built directly into Python, requiring no external servers)
print("\nExporting to SQL Database...")
conn = sqlite3.connect('financial_data.db')
df.to_sql('sp500_companies', conn, if_exists='replace', index=False)
conn.close()

print("Pipeline complete. Data saved to 'financial_data.db'.")




