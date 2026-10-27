import json
import os
import math
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "portfolio_config.json")
OUT_JSON_PATH = os.path.join(BASE_DIR, "data", "market_data.json")
OUT_JS_PATH = os.path.join(BASE_DIR, "data", "market_data.js")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)


def calc_metrics(daily_returns, benchmark_returns=None, rf_annual=0.042):
    """
    Calculate annualized return, volatility, Sharpe ratio, Sortino ratio, MDD, Beta, Alpha.
    daily_returns: pd.Series of daily pct changes
    """
    s = daily_returns.dropna()
    if len(s) < 5:
        return {
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "volatility_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "mdd_pct": 0.0,
            "beta": 1.0,
            "win_rate_pct": 0.0,
        }

    cum = (1 + s).cumprod()
    total_return = float(cum.iloc[-1] - 1.0)
    n_days = len(s)
    cagr = float((1.0 + total_return) ** (252.0 / max(n_days, 1)) - 1.0) if total_return > -1.0 else -1.0

    vol = float(s.std() * math.sqrt(252))
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    excess = s - rf_daily

    sharpe = float((excess.mean() * 252) / vol) if vol > 1e-8 else 0.0

    downside = s[s < rf_daily] - rf_daily
    downside_std = float(math.sqrt((downside ** 2).sum() / len(s)) * math.sqrt(252))
    sortino = float((excess.mean() * 252) / downside_std) if downside_std > 1e-8 else 0.0

    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = float(dd.min())

    beta = 1.0
    if benchmark_returns is not None:
        aligned = pd.concat([s, benchmark_returns], axis=1, sort=False).dropna()
        if len(aligned) > 10:
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
            if cov[1, 1] > 1e-10:
                beta = float(cov[0, 1] / cov[1, 1])

    win_rate = float((s > 0).mean() * 100.0)

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "mdd_pct": round(mdd * 100, 2),
        "beta": round(beta, 2),
        "win_rate_pct": round(win_rate, 1),
    }


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    holdings = config["holdings"]
    benchmarks = config["benchmarks"]
    accounts = config["accounts"]

    holding_tickers = sorted(list(set(h["ticker"] for h in holdings)))
    # Note: State Street renamed SPLG to SPYM in Yahoo Finance. Map SPLG -> SPYM for Yahoo queries.
    YF_TICKER_MAP = {"SPLG": "SPYM"}
    benchmark_tickers = [b["ticker"] for b in benchmarks]
    yf_holding_tickers = [YF_TICKER_MAP.get(t, t) for t in holding_tickers]
    all_yf_tickers = sorted(list(set(yf_holding_tickers + benchmark_tickers + ["KRW=X"])))

    print(f"Fetching market data for {len(all_yf_tickers)} tickers: {all_yf_tickers}")

    # Download 2 years of daily history to cover 1D, 1M, 3M (Quarter), YTD, 1Y
    hist = yf.download(all_yf_tickers, period="2y", interval="1d", auto_adjust=True, progress=False)

    close_df = hist["Close"] if "Close" in hist else hist
    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame()

    # Copy mapped tickers back to original ticker names (e.g. SPYM -> SPLG)
    for orig_t, yf_t in YF_TICKER_MAP.items():
        if yf_t in close_df.columns:
            close_df[orig_t] = close_df[yf_t]

    all_tickers = sorted(list(set(holding_tickers + benchmark_tickers + ["KRW=X"])))

    # Forward fill missing values across different market holidays (US vs KR vs Crypto)
    close_df = close_df.ffill().bfill()

    # Exchange rate USD/KRW
    usdkrw = 1380.0
    if "KRW=X" in close_df.columns and not close_df["KRW=X"].dropna().empty:
        usdkrw = float(close_df["KRW=X"].dropna().iloc[-1])
        if usdkrw < 900 or usdkrw > 2000:
            usdkrw = 1380.0

    # Fetch dividend history for holding tickers
    dividend_info = {}
    for ticker in holding_tickers:
        try:
            yf_sym = YF_TICKER_MAP.get(ticker, ticker)
            t_obj = yf.Ticker(yf_sym)
            divs = t_obj.dividends
            if divs is not None and len(divs) > 0:
                # Filter last 365 days from the latest date in divs or now
                divs.index = divs.index.tz_localize(None)
                latest_div_date = close_df.index[-1].tz_localize(None) if close_df.index.tz is not None else close_df.index[-1]
                one_year_ago = latest_div_date - pd.Timedelta(days=365)
                recent_divs = divs[divs.index >= one_year_ago]
                # If recent_divs is empty, take last 4 payouts or last year of available divs
                if len(recent_divs) == 0:
                    last_avail = divs.index[-1]
                    recent_divs = divs[divs.index >= (last_avail - pd.Timedelta(days=365))]

                annual_div_per_share = float(recent_divs.sum())
                # Monthly distribution pattern (which months this ticker pays)
                monthly_by_month = {m: 0.0 for m in range(1, 13)}
                for dt, val in recent_divs.items():
                    monthly_by_month[dt.month] += float(val)

                # Frequency estimation
                freq = len(recent_divs)
                freq_label = "분기배당" if 3 <= freq <= 5 else ("월배당" if freq >= 10 else ("연배당/반기" if freq >= 1 else "무배당"))

                dividend_info[ticker] = {
                    "annual_div_usd": round(annual_div_per_share, 4),
                    "monthly_pattern": {str(k): round(v, 4) for k, v in monthly_by_month.items()},
                    "frequency": freq_label,
                    "payout_months": sorted(list(set(dt.month for dt in recent_divs.index))),
                }
            else:
                dividend_info[ticker] = {
                    "annual_div_usd": 0.0,
                    "monthly_pattern": {str(m): 0.0 for m in range(1, 13)},
                    "frequency": "무배당",
                    "payout_months": [],
                }
        except Exception as e:
            print(f"Warning: Dividend fetch error for {ticker}: {e}")
            dividend_info[ticker] = {
                "annual_div_usd": 0.0,
                "monthly_pattern": {str(m): 0.0 for m in range(1, 13)},
                "frequency": "무배당",
                "payout_months": [],
            }

    # Current prices & daily changes for each ticker
    ticker_quotes = {}
    for ticker in all_tickers:
        if ticker in close_df.columns:
            s = close_df[ticker].dropna()
            if len(s) >= 2:
                curr = float(s.iloc[-1])
                prev = float(s.iloc[-2])
                chg_pct = ((curr / prev) - 1.0) * 100.0 if prev > 0 else 0.0
                # Also 1M, 3M, YTD, 1Y return for individual ticker
                idx_1m = max(0, len(s) - 22)
                idx_3m = max(0, len(s) - 63)
                idx_1y = max(0, len(s) - 252)
                cur_year = s.index[-1].year
                ytd_s = s[s.index.year == cur_year]
                ytd_base = float(s[s.index.year < cur_year].iloc[-1]) if len(s[s.index.year < cur_year]) > 0 else float(ytd_s.iloc[0])

                ticker_quotes[ticker] = {
                    "price": round(curr, 4),
                    "prev_close": round(prev, 4),
                    "change_1d_pct": round(chg_pct, 2),
                    "change_1m_pct": round(((curr / float(s.iloc[idx_1m])) - 1.0) * 100.0, 2),
                    "change_3m_pct": round(((curr / float(s.iloc[idx_3m])) - 1.0) * 100.0, 2),
                    "change_ytd_pct": round(((curr / ytd_base) - 1.0) * 100.0, 2),
                    "change_1y_pct": round(((curr / float(s.iloc[idx_1y])) - 1.0) * 100.0, 2),
                }

    # Build Portfolio Historical Daily Series (last ~252 trading days)
    # Note: SPLG had a ticker/data history in yfinance; if SPLG is missing or NaN in recent yfinance (since SPLG or SPYM), fallback safely
    for h in holdings:
        t = h["ticker"]
        if t not in close_df.columns or close_df[t].isna().all():
            # Fallback if ticker symbol changed or missing
            if t == "SPLG" and "^GSPC" in close_df.columns:
                close_df[t] = close_df["^GSPC"] / 85.0
            else:
                close_df[t] = h.get("avg_price", 100.0)

    # Filter to standard trading days (using S&P500 ^GSPC calendar as baseline so weekend crypto noise doesn't distort stock portfolio daily vol)
    if "^GSPC" in hist["Close"]:
        valid_dates = hist["Close"]["^GSPC"].dropna().index
        trading_df = close_df.loc[valid_dates].ffill().bfill()
    else:
        trading_df = close_df.ffill().bfill()

    # Keep last 260 trading days (~1 year + buffer)
    trading_df = trading_df.iloc[-260:]

    total_cash_usd = sum(float(a["cash_usd"]) for a in accounts)

    # Compute daily portfolio market value (equities + cash)
    port_equity_series = pd.Series(0.0, index=trading_df.index)
    for h in holdings:
        t = h["ticker"]
        shares = float(h["shares"])
        if t in trading_df.columns:
            port_equity_series += trading_df[t] * shares

    port_total_series = port_equity_series + total_cash_usd

    # Enrich holdings with live market values, weights, dividend estimates
    enriched_holdings = []
    total_equity_usd = float(port_equity_series.iloc[-1])
    total_assets_usd = total_equity_usd + total_cash_usd

    for h in holdings:
        t = h["ticker"]
        shares = float(h["shares"])
        quote = ticker_quotes.get(t, {
            "price": h.get("avg_price", 100.0),
            "prev_close": h.get("avg_price", 100.0),
            "change_1d_pct": 0.0,
            "change_1m_pct": 0.0,
            "change_3m_pct": 0.0,
            "change_ytd_pct": 0.0,
            "change_1y_pct": 0.0,
        })
        price = float(quote["price"])
        market_val_usd = shares * price
        avg_price = float(h.get("avg_price", price))
        cost_basis_usd = shares * avg_price
        profit_usd = market_val_usd - cost_basis_usd
        profit_pct = ((price / avg_price) - 1.0) * 100.0 if avg_price > 0 else 0.0

        d_info = dividend_info.get(t, {"annual_div_usd": 0.0, "monthly_pattern": {str(m): 0.0 for m in range(1, 13)}, "frequency": "무배당", "payout_months": []})
        ann_div_per_share = float(d_info["annual_div_usd"])
        div_yield_pct = (ann_div_per_share / price * 100.0) if price > 0 else 0.0
        annual_div_total_usd = shares * ann_div_per_share
        yield_on_cost_pct = (ann_div_per_share / avg_price * 100.0) if avg_price > 0 else 0.0

        enriched_holdings.append({
            **h,
            "current_price_usd": round(price, 2),
            "market_value_usd": round(market_val_usd, 2),
            "cost_basis_usd": round(cost_basis_usd, 2),
            "profit_usd": round(profit_usd, 2),
            "profit_pct": round(profit_pct, 2),
            "weight_of_total_pct": round((market_val_usd / total_assets_usd) * 100.0, 2) if total_assets_usd > 0 else 0.0,
            "weight_of_equity_pct": round((market_val_usd / total_equity_usd) * 100.0, 2) if total_equity_usd > 0 else 0.0,
            "change_1d_pct": quote["change_1d_pct"],
            "change_1m_pct": quote["change_1m_pct"],
            "change_3m_pct": quote["change_3m_pct"],
            "change_ytd_pct": quote["change_ytd_pct"],
            "change_1y_pct": quote["change_1y_pct"],
            "dividend_per_share_usd": round(ann_div_per_share, 4),
            "dividend_yield_pct": round(div_yield_pct, 2),
            "yield_on_cost_pct": round(yield_on_cost_pct, 2),
            "annual_dividend_usd": round(annual_div_total_usd, 2),
            "dividend_frequency": d_info["frequency"],
            "payout_months": d_info["payout_months"],
            "monthly_div_per_share": d_info["monthly_pattern"],
        })

    # Period performance helper for Portfolio & Benchmarks
    # Periods: 1D (일간), 1M (월간 - 21 trading days), 3M (분기별 - 63 trading days), YTD (연초대비), 1Y (연간 - 252 trading days)
    n_len = len(trading_df)
    cur_year = trading_df.index[-1].year
    ytd_mask = trading_df.index.year < cur_year
    ytd_start_idx = int(ytd_mask.sum() - 1) if ytd_mask.any() else 0

    period_indices = {
        "1D": max(0, n_len - 2),
        "1M": max(0, n_len - 22),
        "3M": max(0, n_len - 64),
        "YTD": max(0, ytd_start_idx),
        "1Y": max(0, n_len - 252),
    }

    # Build time series & risk metrics per period for Portfolio + Benchmarks (BOTH USD and KRW basis)
    periods_data = {}
    fx_series_full = trading_df["KRW=X"] if "KRW=X" in trading_df.columns else pd.Series(usdkrw, index=trading_df.index)
    fx_series_full = fx_series_full.replace(0, usdkrw).ffill().bfill()

    port_total_krw_series = port_total_series * fx_series_full

    for p_key, start_idx in period_indices.items():
        sub_df = trading_df.iloc[start_idx:]
        sub_fx = fx_series_full.iloc[start_idx:]
        sub_port_usd = port_total_series.iloc[start_idx:]
        sub_port_krw = port_total_krw_series.iloc[start_idx:]

        dates_str = [d.strftime("%Y-%m-%d") for d in sub_df.index]

        # 1) USD Basis Portfolio
        port_norm_usd = ((sub_port_usd / float(sub_port_usd.iloc[0])) - 1.0) * 100.0
        port_values_usd = [round(float(v), 2) for v in sub_port_usd]
        port_norm_usd_list = [round(float(v), 2) for v in port_norm_usd]

        # 2) KRW Basis Portfolio (reflecting daily historical USD/KRW exchange rate)
        port_norm_krw = ((sub_port_krw / float(sub_port_krw.iloc[0])) - 1.0) * 100.0
        port_values_krw = [round(float(v), 0) for v in sub_port_krw]
        port_norm_krw_list = [round(float(v), 2) for v in port_norm_krw]

        snp_usd_series = sub_df["^GSPC"] if "^GSPC" in sub_df.columns else sub_port_usd
        snp_krw_series = snp_usd_series * sub_fx

        port_metrics_usd = calc_metrics(sub_port_usd.pct_change(), snp_usd_series.pct_change())
        port_metrics_usd["total_return_pct"] = round(float(port_norm_usd.iloc[-1]), 2)
        port_metrics_usd["profit_usd"] = round(float(sub_port_usd.iloc[-1] - sub_port_usd.iloc[0]), 2)
        port_metrics_usd["profit_krw"] = round(float(sub_port_krw.iloc[-1] - sub_port_krw.iloc[0]), 0)

        port_metrics_krw = calc_metrics(sub_port_krw.pct_change(), snp_krw_series.pct_change())
        port_metrics_krw["total_return_pct"] = round(float(port_norm_krw.iloc[-1]), 2)
        port_metrics_krw["profit_usd"] = round(float(sub_port_usd.iloc[-1] - sub_port_usd.iloc[0]), 2)
        port_metrics_krw["profit_krw"] = round(float(sub_port_krw.iloc[-1] - sub_port_krw.iloc[0]), 0)

        bench_series_usd = {}
        bench_metrics_usd = {}
        bench_series_krw = {}
        bench_metrics_krw = {}

        for b in benchmarks:
            b_id = b["id"]
            b_tk = b["ticker"]
            if b_tk in sub_df.columns:
                raw_s = sub_df[b_tk]
                is_krw_native = b_tk.endswith(".KS") or b_tk.endswith(".KQ")

                if is_krw_native:
                    # Native KRW asset (e.g. KOSPI 200): KRW series is raw, USD series divides by daily FX
                    b_s_krw = raw_s
                    b_s_usd = raw_s / sub_fx
                else:
                    # Native USD asset (BTC, S&P500, NDX, SOXX): USD series is raw, KRW series multiplies by daily FX
                    b_s_usd = raw_s
                    b_s_krw = raw_s * sub_fx

                b_norm_usd = ((b_s_usd / float(b_s_usd.iloc[0])) - 1.0) * 100.0
                bench_series_usd[b_id] = [round(float(v), 2) for v in b_norm_usd]
                m_usd = calc_metrics(b_s_usd.pct_change(), snp_usd_series.pct_change())
                m_usd["total_return_pct"] = round(float(b_norm_usd.iloc[-1]), 2)
                bench_metrics_usd[b_id] = m_usd

                b_norm_krw = ((b_s_krw / float(b_s_krw.iloc[0])) - 1.0) * 100.0
                bench_series_krw[b_id] = [round(float(v), 2) for v in b_norm_krw]
                m_krw = calc_metrics(b_s_krw.pct_change(), snp_krw_series.pct_change())
                m_krw["total_return_pct"] = round(float(b_norm_krw.iloc[-1]), 2)
                bench_metrics_krw[b_id] = m_krw

        periods_data[p_key] = {
            "dates": dates_str,
            "portfolio_values_usd": port_values_usd,
            "portfolio_values_krw": port_values_krw,
            "portfolio_return_series": port_norm_usd_list,
            "portfolio_return_series_usd": port_norm_usd_list,
            "portfolio_return_series_krw": port_norm_krw_list,
            "portfolio_metrics": port_metrics_usd,
            "portfolio_metrics_usd": port_metrics_usd,
            "portfolio_metrics_krw": port_metrics_krw,
            "benchmark_series": bench_series_usd,
            "benchmark_series_usd": bench_series_usd,
            "benchmark_series_krw": bench_series_krw,
            "benchmark_metrics": bench_metrics_usd,
            "benchmark_metrics_usd": bench_metrics_usd,
            "benchmark_metrics_krw": bench_metrics_krw,
        }

    # Monthly Dividend Breakdown (1~12월)
    monthly_dividends_usd = {str(m): 0.0 for m in range(1, 13)}
    monthly_dividend_details = {str(m): [] for m in range(1, 13)}

    for eh in enriched_holdings:
        shares = float(eh["shares"])
        pat = eh["monthly_div_per_share"]
        for m_str, dps in pat.items():
            amt = shares * float(dps)
            if amt > 0.001:
                monthly_dividends_usd[m_str] = round(monthly_dividends_usd[m_str] + amt, 2)
                monthly_dividend_details[m_str].append({
                    "ticker": eh["ticker"],
                    "account": eh["account"],
                    "amount_usd": round(amt, 2),
                })

    total_annual_div_usd = round(sum(monthly_dividends_usd.values()), 2)
    portfolio_div_yield_total_pct = round((total_annual_div_usd / total_assets_usd) * 100.0, 2) if total_assets_usd > 0 else 0.0
    portfolio_div_yield_equity_pct = round((total_annual_div_usd / total_equity_usd) * 100.0, 2) if total_equity_usd > 0 else 0.0

    # Yield to Maturity / Expected Annual Total Return + Dividend Projection (YTM / 예상 연환산 기대수익 + 배당)
    port_1y_metrics = periods_data["1Y"]["portfolio_metrics"]
    estimated_ytm_total_pct = round(port_1y_metrics["cagr_pct"] + portfolio_div_yield_total_pct, 2)

    actual_div_history = config.get("actual_dividend_history", [])
    actual_div_total_usd = round(sum(float(x["amount_usd"]) for x in actual_div_history), 2)
    actual_div_2026_usd = round(sum(float(x["amount_usd"]) for x in actual_div_history if x["date"].startswith("2026")), 2)

    output = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_date": trading_df.index[-1].strftime("%Y-%m-%d"),
        "usdkrw": round(usdkrw, 2),
        "summary": {
            "total_assets_usd": round(total_assets_usd, 2),
            "total_equity_usd": round(total_equity_usd, 2),
            "total_cash_usd": round(total_cash_usd, 2),
            "total_cost_basis_usd": round(sum(eh["cost_basis_usd"] for eh in enriched_holdings) + total_cash_usd, 2),
            "total_profit_usd": round(sum(eh["profit_usd"] for eh in enriched_holdings), 2),
            "total_annual_dividend_usd": total_annual_div_usd,
            "monthly_avg_dividend_usd": round(total_annual_div_usd / 12.0, 2),
            "dividend_yield_total_pct": portfolio_div_yield_total_pct,
            "dividend_yield_equity_pct": portfolio_div_yield_equity_pct,
            "estimated_ytm_pct": estimated_ytm_total_pct,
            "actual_cumulative_dividend_usd": actual_div_total_usd,
            "actual_2026_dividend_usd": actual_div_2026_usd,
        },
        "accounts": accounts,
        "holdings": enriched_holdings,
        "benchmarks": benchmarks,
        "monthly_dividends_usd": monthly_dividends_usd,
        "monthly_dividend_details": monthly_dividend_details,
        "actual_dividend_history": actual_div_history,
        "periods": periods_data,
    }

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    with open(OUT_JS_PATH, "w", encoding="utf-8") as f:
        f.write("window.INITIAL_MARKET_DATA = ")
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write(";\n")

    print(f"Successfully updated market data -> {OUT_JSON_PATH}")
    print(f"Total Assets: ${total_assets_usd:,.2f} (KRW {total_assets_usd * usdkrw:,.0f})")
    print(f"Annual Dividend: ${total_annual_div_usd:,.2f} ({portfolio_div_yield_total_pct}% of total assets)")


if __name__ == "__main__":
    main()
