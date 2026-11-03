"""
Sector-level trend-following study.

Tests simple moving-average (SMA) crossover strategies on SPY and the original
nine S&P 500 sector ETFs, using sector information from the SQLite database
built by scraper.py.

Methodology
-----------
* Data: daily adjusted close prices (yfinance), 2005-2025.
* Signal: long when fast SMA > slow SMA, otherwise in cash (long/flat by default).
* No lookahead: the signal is lagged one day before it is traded.
* Costs: a proportional cost is charged on every change in position.
* Parameter selection: for each asset, the SMA pair with the best Sharpe ratio on
  the TRAIN period (2005-2017) is chosen, then evaluated on the TEST period
  (2018-2025), which was never used for selection.
* Evaluation: annualized return, volatility, Sharpe, max drawdown, trade count,
  and an OLS regression of strategy returns on buy-and-hold returns (alpha, beta).

Limitations are listed in README.md.

Usage:
    python scraper.py       # once, to build financial_data.db
    python backtester.py
"""

import sqlite3
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # render to files, no display needed
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------- configuration -----------------------------
DB_PATH = "financial_data.db"
RESULTS_DIR = Path("results")
START, END = "2005-01-01", "2025-12-31"
TRAIN_END = "2017-12-31"
COST_BPS = 5  # 5 basis points per unit of position change
ALLOW_SHORT = False  # long/flat avoids having to model borrow costs
TRADING_DAYS = 252
PARAM_GRID = [(10, 30), (20, 50), (20, 100), (30, 90), (50, 200)]

# Original nine sector ETFs have history back to 1998. XLRE (2015) and XLC (2018)
# are excluded because they do not cover the full sample.
SECTOR_ETFS = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Materials": "XLB",
}
BENCHMARK = ("S&P 500 (benchmark)", "SPY")


# ------------------------------ data loading ------------------------------
def load_sector_universe(db_path: str = DB_PATH) -> dict:
    """Return {label: etf_ticker}, using the scraped database to decide which sectors exist."""
    universe = {BENCHMARK[0]: BENCHMARK[1]}
    if not Path(db_path).exists():
        print(f"[warn] {db_path} not found; run scraper.py first. Using default sector list.")
        universe.update(SECTOR_ETFS)
        return universe

    with sqlite3.connect(db_path) as conn:
        counts = pd.read_sql(
            "SELECT Sector, COUNT(*) AS n FROM sp500_companies GROUP BY Sector", conn
        )
    for sector, n in zip(counts["Sector"], counts["n"]):
        if sector in SECTOR_ETFS:
            universe[sector] = SECTOR_ETFS[sector]
            print(f"  {sector:<26} {n:>3} constituents -> {SECTOR_ETFS[sector]}")
        else:
            print(f"  {sector:<26} {n:>3} constituents -> skipped (ETF history too short)")
    return universe


def load_prices(tickers: list) -> pd.DataFrame:
    """Download adjusted close prices. Returns a DataFrame with one column per ticker."""
    import yfinance as yf

    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.dropna(how="all")
    # Drop tickers that do not cover the sample.
    first_valid = close.apply(lambda s: s.first_valid_index())
    ok = first_valid <= pd.Timestamp(START) + pd.Timedelta(days=10)
    for t in close.columns[~ok]:
        print(f"[warn] {t} dropped: history starts {first_valid[t].date()}")
    return close.loc[:, ok]


# ------------------------------- strategy -------------------------------
def run_strategy(prices: pd.Series, fast: int, slow: int,
                 cost_bps: float = COST_BPS, allow_short: bool = ALLOW_SHORT):
    """Return (net_returns, position) for an SMA crossover strategy."""
    ret = prices.pct_change()
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()

    if allow_short:
        signal = pd.Series(np.where(fast_ma > slow_ma, 1.0, -1.0), index=prices.index)
    else:
        signal = (fast_ma > slow_ma).astype(float)
    signal[slow_ma.isna()] = 0.0

    position = signal.shift(1).fillna(0.0)  # trade the day AFTER the signal
    turnover = position.diff().abs().fillna(position.abs())
    net = position * ret - turnover * (cost_bps / 10_000)
    return net.iloc[1:], position.iloc[1:]


# ------------------------------- metrics -------------------------------
def performance(ret: pd.Series, position: pd.Series = None) -> dict:
    n = len(ret)
    equity = (1 + ret).cumprod()
    std = ret.std()
    out = {
        "ann_return": equity.iloc[-1] ** (TRADING_DAYS / n) - 1,
        "ann_vol": std * np.sqrt(TRADING_DAYS),
        "sharpe": ret.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
    }
    if position is not None:
        out["trades"] = int((position.diff().abs() > 0).sum())
    return out


def alpha_beta(strategy: pd.Series, benchmark: pd.Series) -> dict:
    """OLS of strategy returns on buy-and-hold returns (daily). Alpha is annualized."""
    y = strategy.values
    X = np.column_stack([np.ones(len(benchmark)), benchmark.values])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - 2
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {
        "alpha_ann": coef[0] * TRADING_DAYS,
        "alpha_t": coef[0] / se[0],
        "beta": coef[1],
    }


# --------------------------------- study ---------------------------------
def run_study(prices: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Run the train/test study for every asset. labels maps ticker -> display name."""
    RESULTS_DIR.mkdir(exist_ok=True)
    burn_in = max(slow for _, slow in PARAM_GRID)  # same evaluation window for every pair
    train_end = pd.Timestamp(TRAIN_END)

    rows, sweep_rows, test_curves = [], [], {}

    for ticker in prices.columns:
        px = prices[ticker].dropna()
        bh = px.pct_change().iloc[1:]
        window = bh.index[burn_in:]
        bh = bh.loc[window]

        best = None
        for fast, slow in PARAM_GRID:
            net, pos = run_strategy(px, fast, slow)
            net, pos = net.loc[window], pos.loc[window]
            tr = performance(net[net.index <= train_end])
            te = performance(net[net.index > train_end])
            sweep_rows.append({"asset": ticker, "fast": fast, "slow": slow,
                               "train_sharpe": tr["sharpe"], "test_sharpe": te["sharpe"]})
            if best is None or tr["sharpe"] > best["train_sharpe"]:
                best = {"fast": fast, "slow": slow, "train_sharpe": tr["sharpe"],
                        "net": net, "pos": pos}

        test_mask = bh.index > train_end
        s_net, s_pos = best["net"][test_mask], best["pos"][test_mask]
        b_ret = bh[test_mask]
        s_perf, b_perf = performance(s_net, s_pos), performance(b_ret)
        ab = alpha_beta(s_net, b_ret)
        test_curves[ticker] = ((1 + s_net).cumprod(), (1 + b_ret).cumprod())

        rows.append({
            "asset": ticker, "name": labels.get(ticker, ticker),
            "selected_pair": f"{best['fast']}/{best['slow']}",
            "train_sharpe": best["train_sharpe"],
            "test_sharpe_strategy": s_perf["sharpe"], "test_sharpe_buyhold": b_perf["sharpe"],
            "test_return_strategy": s_perf["ann_return"], "test_return_buyhold": b_perf["ann_return"],
            "test_maxdd_strategy": s_perf["max_drawdown"], "test_maxdd_buyhold": b_perf["max_drawdown"],
            "trades_test": s_perf["trades"],
            "alpha_ann": ab["alpha_ann"], "alpha_t": ab["alpha_t"], "beta": ab["beta"],
        })

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "results_summary.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(RESULTS_DIR / "parameter_sweep.csv", index=False)
    plot_equity_curves(test_curves, labels)
    return results


def plot_equity_curves(curves: dict, labels: dict) -> None:
    n = len(curves)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (ticker, (strat, bh)) in zip(axes, curves.items()):
        ax.plot(bh.index, bh.values, label="Buy & hold", color="gray")
        ax.plot(strat.index, strat.values, label="SMA strategy (net)", color="tab:blue")
        ax.set_title(f"{labels.get(ticker, ticker)} ({ticker})", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Out-of-sample equity curves (test period: after {TRAIN_END}, growth of $1)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "equity_curves_test.png", dpi=150)
    plt.close(fig)


def print_summary(results: pd.DataFrame) -> None:
    show = results[["asset", "selected_pair", "train_sharpe", "test_sharpe_strategy",
                    "test_sharpe_buyhold", "test_maxdd_strategy", "test_maxdd_buyhold",
                    "alpha_ann", "alpha_t", "beta"]]
    print("\n=== Out-of-sample results (test period) ===")
    print(show.round(3).to_string(index=False))

    beat_sharpe = (results["test_sharpe_strategy"] > results["test_sharpe_buyhold"]).sum()
    smaller_dd = (results["test_maxdd_strategy"] > results["test_maxdd_buyhold"] + 0.005).sum()
    sig_alpha = (results["alpha_t"].abs() > 2).sum()
    n = len(results)
    print(f"\nStrategy beat buy-and-hold Sharpe in {beat_sharpe}/{n} assets.")
    print(f"Strategy had a smaller max drawdown in {smaller_dd}/{n} assets.")
    print(f"Alpha statistically significant (|t|>2) in {sig_alpha}/{n} assets.")
    print(f"Results saved in {RESULTS_DIR}/")


def main() -> None:
    print("Loading sector universe from database...")
    universe = load_sector_universe()
    labels = {ticker: name for name, ticker in universe.items()}
    print("\nDownloading prices...")
    prices = load_prices(list(universe.values()))
    results = run_study(prices, labels)
    print_summary(results)


if __name__ == "__main__":
    main()
