import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import yfinance as yf
import panel as pn
import plotly.express as px
import plotly.graph_objects as go

pn.extension("tabulator", "plotly")

TRADING_DAYS = 252
DEFAULT_TICKER_LIST = [
    "AARTIIND.NS", "ABBOTINDIA.NS", "ABB.NS", "ABCAPITAL.NS", "BALRAMCHIN.NS",
    "AMBUJACEM.NS", "ADANIENT.NS", "ABFRL.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS",
    "APOLLOTYRE.NS", "ACC.NS", "BSOFT.NS", "BATAINDIA.NS", "ASHOKLEY.NS",
    "ASIANPAINT.NS", "ATUL.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS",
    "BAJFINANCE.NS", "BALKRISIND.NS", "BEL.NS", "BHARATFORG.NS", "BIOCON.NS",
    "BOSCHLTD.NS", "BPCL.NS", "BHEL.NS", "BRITANNIA.NS", "CANFINHOME.NS",
    "CHAMBLFERT.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS",
    "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS", "CUMMINSIND.NS", "DABUR.NS",
    "DALBHARAT.NS", "DEEPAKNTR.NS", "DELTACORP.NS", "DIVISLAB.NS", "DLF.NS",
    "DRREDDY.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS",
    "GLENMARK.NS", "GMRINFRA.NS", "GNFC.NS", "GODREJCP.NS", "GODREJPROP.NS",
    "GUJGASLTD.NS", "DIXON.NS", "HAL.NS", "HAVELLS.NS", "HCLTECH.NS",
    "HDFCAMC.NS", "HINDALCO.NS", "HINDCOPPER.NS", "IDEA.NS", "IEX.NS",
    "HDFCBANK.NS", "INDHOTEL.NS", "HEROMOTOCO.NS", "HINDPETRO.NS", "HINDUNILVR.NS",
    "IBULHSGFIN.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "INDIACEM.NS",
    "IDFC.NS", "IDFCFIRSTB.NS", "INDIAMART.NS", "INDIGO.NS", "INDUSINDBK.NS",
    "INFY.NS", "IOC.NS", "IPCALAB.NS", "INDUSTOWER.NS", "IRCTC.NS", "ITC.NS",
    "JINDALSTEL.NS", "JKCEMENT.NS", "JSWSTEEL.NS", "JUBLFOOD.NS", "L&TFH.NS",
    "LAURUSLABS.NS", "LALPATHLAB.NS", "LT.NS", "LTIM.NS", "LTTS.NS", "LUPIN.NS",
    "M&M.NS", "M&MFIN.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS",
    "MCDOWELL-N.NS", "MCX.NS", "METROPOLIS.NS", "MGL.NS", "MOTHERSON.NS",
    "MPHASIS.NS", "NAUKRI.NS", "NESTLEIND.NS", "NMDC.NS", "NTPC.NS",
    "OBEROIRLTY.NS", "ONGC.NS", "PAGEIND.NS", "PEL.NS", "PETRONET.NS",
    "PFC.NS", "PIDILITIND.NS", "PIIND.NS", "POLYCAB.NS", "PVRINOX.NS",
    "RECLTD.NS", "RELIANCE.NS", "SBILIFE.NS", "SHREECEM.NS", "SIEMENS.NS",
    "SRF.NS", "SUNPHARMA.NS", "SUNTV.NS", "OFSS.NS", "SYNGENE.NS",
    "TATACHEM.NS", "PERSISTENT.NS", "PNB.NS", "TATACOMM.NS", "TATACONSUM.NS",
    "TATAMOTORS.NS", "SAIL.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "UBL.NS", "VEDL.NS", "ZEEL.NS",
    "ZYDUSLIFE.NS", "ALKEM.NS", "ASTRAL.NS", "AUROPHARMA.NS", "TORNTPHARM.NS",
    "BANDHANBNK.NS", "TVSMOTOR.NS", "BANKBARODA.NS", "BHARTIARTL.NS", "CHOLAFIN.NS",
    "CUB.NS", "GRASIM.NS", "HDFCLIFE.NS", "IGL.NS", "KOTAKBANK.NS", "MRF.NS",
    "MUTHOOTFIN.NS", "EICHERMOT.NS", "GRANULES.NS", "RAMCOCEM.NS", "RBLBANK.NS",
    "SBIN.NS", "SHRIRAMFIN.NS", "VOLTAS.NS", "WIPRO.NS", "AUBANK.NS",
    "BERGEPAINT.NS", "CANBK.NS", "ULTRACEMCO.NS", "NAVINFLUOR.NS", "SBICARD.NS",
    "UPL.NS", "POWERGRID.NS"
]

# -----------------------------
# CONFIGS
# -----------------------------
BENCHMARK = "^NSEI"
BATCH_SIZE = 25
MIN_OBS = 120  # min 120 trading days

# -----------------------------
# HELPERS
# -----------------------------
def annual_rf_to_daily(annual_rf):
    return (1 + annual_rf) ** (1 / TRADING_DAYS) - 1

def drawdown_stats(returns):
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    max_dd = drawdown.min()
    years = len(returns) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    return cagr, max_dd

def compute_risk_metrics(returns, annual_rf):
    ann_return = returns.mean() * TRADING_DAYS
    ann_vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

    daily_rf = annual_rf_to_daily(annual_rf)
    downside = returns[returns < daily_rf] - daily_rf
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS) if len(downside) else np.nan

    sharpe = (ann_return - annual_rf) / ann_vol if pd.notna(ann_vol) and ann_vol != 0 else np.nan
    sortino = (ann_return - annual_rf) / downside_dev if pd.notna(downside_dev) and downside_dev != 0 else np.nan

    cagr, max_dd = drawdown_stats(returns)
    calmar = cagr / abs(max_dd) if pd.notna(max_dd) and max_dd != 0 else np.nan

    return ann_return, ann_vol, sharpe, sortino, cagr, max_dd, calmar

def batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def extract_close_matrix(raw):
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"].copy()
        else:
            return pd.DataFrame()
    else:
        if "Close" in raw.columns:
            close = raw[["Close"]].copy()
            close.columns = ["SINGLE"]
        elif "Adj Close" in raw.columns:
            close = raw[["Adj Close"]].copy()
            close.columns = ["SINGLE"]
        else:
            return pd.DataFrame()

    if isinstance(close, pd.Series):
        close = close.to_frame()

    return close

def bulk_download_closes(tickers, period="1y", interval="1d", batch_size=25):
    close_frames = []
    failed_symbols = []

    for batch in batched(tickers, batch_size):
        try:
            raw = yf.download(
                tickers=batch,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False
            )

            close = extract_close_matrix(raw)

            if close.empty:
                failed_symbols.extend(batch)
                continue

            available = set(close.columns.tolist())
            for ticker in batch:
                if ticker not in available:
                    failed_symbols.append(ticker)
                else:
                    series = close[ticker]
                    if series.dropna().empty:
                        failed_symbols.append(ticker)

            valid_cols = [c for c in close.columns if c not in failed_symbols and not close[c].dropna().empty]
            if valid_cols:
                close_frames.append(close[valid_cols])

        except Exception:
            failed_symbols.extend(batch)

    if close_frames:
        close_df = pd.concat(close_frames, axis=1)
        close_df = close_df.loc[:, ~close_df.columns.duplicated()]
    else:
        close_df = pd.DataFrame()

    failed_symbols = sorted(set(failed_symbols))
    return close_df, failed_symbols

def analyze_stocks_bulk(tickers, benchmark, annual_rf, period="1y", interval="1d", batch_size=25):
    bm = yf.download(
        benchmark,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if bm.empty or "Close" not in bm.columns:
        raise ValueError(f"Could not fetch benchmark data for {benchmark}")

    market_close = bm["Close"]
    if isinstance(market_close, pd.DataFrame):
        market_close = market_close.iloc[:, 0]

    market_close = market_close.dropna()
    market_returns = market_close.pct_change(fill_method=None).dropna()
    market_returns.name = "market"

    market_return = market_returns.mean() * TRADING_DAYS
    market_var = market_returns.var()

    close_df, failed = bulk_download_closes(
        tickers=tickers,
        period=period,
        interval=interval,
        batch_size=batch_size
    )

    if close_df.empty:
        return pd.DataFrame(), pd.DataFrame({"Ticker": failed, "Reason": "No price data"}), market_return

    returns_df = close_df.pct_change(fill_method=None).dropna(how="all")

    rows = []
    for ticker in returns_df.columns:
        stock_returns = returns_df[ticker].dropna()
        if stock_returns.empty:
            continue

        aligned = pd.concat([stock_returns.rename("stock"), market_returns], axis=1).dropna()
        if aligned.empty or pd.isna(market_var) or market_var == 0:
            continue

        cov = np.cov(aligned["stock"], aligned["market"])[0][1]
        beta = cov / market_var
        expected_return = annual_rf + beta * (market_return - annual_rf)

        ann_return, ann_vol, sharpe, sortino, cagr, max_dd, calmar = compute_risk_metrics(stock_returns, annual_rf)

        rows.append({
            "Ticker": ticker,
            "Beta": beta,
            "Expected Return": expected_return,
            "Actual Return": ann_return,
            "Gap": ann_return - expected_return,
            "Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "CAGR": cagr,
            "Max Drawdown": max_dd,
            "Calmar Ratio": calmar,
            "Observations": len(stock_returns),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Gap", ascending=False).set_index("Ticker")

    failed_df = pd.DataFrame({
        "Ticker": sorted(set(failed)),
        "Reason": "No Yahoo price data"
    })

    return df, failed_df, market_return

def make_sml_plot(df, market_return, annual_rf, n=10):
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template="plotly_white", title="No data")
        return fig

    plot_df = df.dropna(subset=["Beta", "Actual Return"]).copy()
    top_df = plot_df.sort_values("Gap", ascending=False).head(n)
    bottom_df = plot_df.sort_values("Gap", ascending=True).head(n)

    x_line = np.linspace(plot_df["Beta"].min() - 0.2, plot_df["Beta"].max() + 0.2, 100)
    y_line = annual_rf + (market_return - annual_rf) * x_line

    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        name="SML",
        line=dict(color="red", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=plot_df["Beta"],
        y=plot_df["Actual Return"],
        mode="markers",
        name="All Stocks",
        text=plot_df.index,
        marker=dict(color="lightgray", size=8),
        hovertemplate="<b>%{text}</b><br>Beta=%{x:.2f}<br>Actual=%{y:.2%}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=top_df["Beta"],
        y=top_df["Actual Return"],
        mode="markers+text",
        text=top_df.index,
        textposition="top center",
        name=f"Top {n} by Gap",
        marker=dict(color="green", size=12, line=dict(color="black", width=1))
    ))

    fig.add_trace(go.Scatter(
        x=bottom_df["Beta"],
        y=bottom_df["Actual Return"],
        mode="markers+text",
        text=bottom_df.index,
        textposition="bottom center",
        name=f"Bottom {n} by Gap",
        marker=dict(color="crimson", size=12, line=dict(color="black", width=1))
    ))

    fig.update_layout(
        template="plotly_white",
        height=650,
        title="Security Market Line — Ranked by Gap",
        xaxis_title="Beta",
        yaxis_title="Annualized Return"
    )
    return fig

def make_bar_plot(df, metric="Sharpe Ratio", n=10):
    if df.empty or metric not in df.columns:
        return go.Figure(layout=dict(template="plotly_white", title="No data"))

    plot_df = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(n).reset_index()
    fig = px.bar(
        plot_df,
        x="Ticker",
        y=metric,
        color=metric,
        template="plotly_white",
        title=f"Top {n} Stocks by {metric}"
    )
    fig.update_layout(height=420)
    return fig

def format_display(df):
    out = df.copy()
    pct_cols = ["Expected Return", "Actual Return", "Gap", "Volatility", "CAGR", "Max Drawdown"]
    num_cols = ["Beta", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]

    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    for c in num_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

    return out.reset_index()

# -----------------------------
# WIDGETS
# -----------------------------
rf_input = pn.widgets.FloatInput(name="Risk-Free Rate", value=0.07, step=0.005)
period_input = pn.widgets.Select(name="Period", options=["6mo", "1y", "2y", "5y"], value="1y")
interval_input = pn.widgets.Select(name="Interval", options=["1d", "1wk"], value="1d")
min_obs_input = pn.widgets.IntSlider(name="Min Observations", start=30, end=252, value=MIN_OBS)
topn_input = pn.widgets.IntSlider(name="Top/Bottom N", start=5, end=20, value=10)

status = pn.pane.Alert("Loading data...", alert_type="warning")
summary = pn.pane.Markdown("")
failed_view = pn.widgets.Tabulator(pd.DataFrame(), pagination="local", page_size=10, height=220)
result_view = pn.widgets.Tabulator(pd.DataFrame(), pagination="local", page_size=20, height=420, sizing_mode="stretch_width")
sml_plot = pn.pane.Plotly(height=650)
bar_plot = pn.pane.Plotly(height=420)
download_button = pn.widgets.FileDownload(label="Download Results CSV", button_type="success", visible=False)

def update_download(df):
    csv_bytes = df.reset_index().to_csv(index=False).encode()
    download_button.file = csv_bytes
    download_button.filename = "sml_results_panel.csv"
    download_button.visible = True

def refresh_dashboard():
    try:
        df, failed_df, market_return = analyze_stocks_bulk(
            tickers=DEFAULT_TICKER_LIST,
            benchmark=BENCHMARK,
            annual_rf=rf_input.value,
            period=period_input.value,
            interval=interval_input.value,
            batch_size=BATCH_SIZE
        )

        if df.empty:
            status.object = "No valid stock data found."
            status.alert_type = "danger"
            result_view.value = pd.DataFrame()
            failed_view.value = failed_df
            sml_plot.object = go.Figure()
            bar_plot.object = go.Figure()
            summary.object = ""
            download_button.visible = False
            return

        filtered = df[df["Observations"] >= min_obs_input.value].copy()

        result_view.value = format_display(filtered)
        failed_view.value = failed_df
        sml_plot.object = make_sml_plot(filtered, market_return, rf_input.value, topn_input.value)
        bar_plot.object = make_bar_plot(filtered, "Sharpe Ratio", topn_input.value)

        summary.object = (
            f"### Dashboard Summary\n"
            f"- Stock universe: **{len(DEFAULT_TICKER_LIST)}**\n"
            f"- Successful tickers: **{len(df)}**\n"
            f"- Filtered tickers: **{len(filtered)}**\n"
            f"- Failed tickers: **{len(failed_df)}**\n"
            f"- Benchmark annualized return: **{market_return:.2%}**"
        )

        update_download(filtered)
        status.object = "Analysis complete."
        status.alert_type = "success"

    except Exception as e:
        status.object = f"Error: {e}"
        status.alert_type = "danger"
        download_button.visible = False

for w in [rf_input, period_input, interval_input, min_obs_input, topn_input]:
    w.param.watch(refresh_dashboard, "value")

sidebar = pn.WidgetBox(
    "## Controls",
    pn.pane.Markdown("Using fixed NSE universe (no CSV needed)."),
    rf_input,
    period_input,
    interval_input,
    min_obs_input,
    topn_input,
    download_button,
    width=340
)

main = pn.Column(
    "## CAPM + Risk Ratios Dashboard",
    status,
    summary,
    pn.Row(sml_plot),
    pn.Row(bar_plot),
    "### Results",
    result_view,
    "### Failed Tickers",
    failed_view
)

dashboard = pn.template.FastListTemplate(
    title="NSE CAPM Risk Dashboard",
    sidebar=[sidebar],
    main=[main],
    accent_base_color="#0f766e",
    header_background="#0f766e"
)

# Run once on load
refresh_dashboard()
dashboard.servable()

if __name__ == "__main__":
    pn.serve(
        dashboard,
        show=True,
        port=5006,
        address="localhost",
        websocket_origin=["localhost:5006", "127.0.0.1:5006"]
    )