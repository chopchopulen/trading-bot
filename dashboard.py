from flask import Flask
import alpaca_trade_api as tradeapi
import pandas as pd
import csv
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

api = tradeapi.REST(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    os.environ["ALPACA_BASE_URL"]
)
app = Flask(__name__)

STARTING_CASH = 100000
LOG_FILE = "trades_log.csv"

def get_spy_return():
    try:
        bars = api.get_bars(
            "SPY",
            tradeapi.rest.TimeFrame.Day,
            start=(datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d"),
            end=datetime.now().strftime("%Y-%m-%d"),
            feed="iex"
        ).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs("SPY", level="symbol")
        start_price = bars["close"].iloc[0]
        end_price = bars["close"].iloc[-1]
        return ((end_price - start_price) / start_price) * 100
    except:
        return 0

@app.route("/")
def dashboard():
    account = api.get_account()
    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)
    total_return = ((portfolio_value - STARTING_CASH) / STARTING_CASH) * 100
    spy_return = get_spy_return()
    alpha = total_return - spy_return

    positions = api.list_positions()
    total_pnl = sum(float(p.unrealized_pl) for p in positions)

    # Load trade log
    trades = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            reader = csv.DictReader(f)
            trades = list(reader)[-20:]
            trades.reverse()

    return_color = "#00c853" if total_return >= 0 else "#ff1744"
    alpha_color = "#00c853" if alpha >= 0 else "#ff1744"
    pnl_color = "#00c853" if total_pnl >= 0 else "#ff1744"

    positions_html = ""
    for p in positions:
        pnl = float(p.unrealized_pl)
        pnl_pct = float(p.unrealized_plpc) * 100
        color = "#00c853" if pnl >= 0 else "#ff1744"
        positions_html += f"""
        <tr>
            <td><strong>{p.symbol}</strong></td>
            <td>{p.qty}</td>
            <td>${float(p.avg_entry_price):.2f}</td>
            <td>${float(p.current_price):.2f}</td>
            <td style="color:{color}">${pnl:+.2f}</td>
            <td style="color:{color}">{pnl_pct:+.2f}%</td>
        </tr>"""

    trades_html = ""
    for t in trades:
        color = "#00c853" if t["action"] == "BUY" else "#ff1744"
        trades_html += f"""
        <tr>
            <td>{t["timestamp"][:16]}</td>
            <td><strong>{t["stock"]}</strong></td>
            <td style="color:{color}">{t["action"]}</td>
            <td>${float(t["price"]):.2f}</td>
        </tr>"""

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Bot Dashboard</title>
        <meta http-equiv="refresh" content="30">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; padding: 30px; }}
            h1 {{ font-size: 28px; margin-bottom: 5px; }}
            .subtitle {{ color: #8b949e; margin-bottom: 30px; font-size: 14px; }}
            .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 30px; }}
            .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; }}
            .card .label {{ color: #8b949e; font-size: 13px; margin-bottom: 8px; }}
            .card .value {{ font-size: 26px; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; background: #161b22;
                     border: 1px solid #30363d; border-radius: 12px; overflow: hidden; }}
            th {{ background: #21262d; padding: 12px 16px; text-align: left;
                  color: #8b949e; font-size: 13px; }}
            td {{ padding: 12px 16px; border-top: 1px solid #21262d; font-size: 14px; }}
            tr:hover td {{ background: #1c2128; }}
            .section-title {{ font-size: 18px; font-weight: bold; margin: 30px 0 12px; }}
            .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px;
                      font-size: 12px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>🤖 Trading Bot Dashboard</h1>
        <div class="subtitle">Auto-refreshes every 30 seconds · Last updated: {datetime.now().strftime("%b %d %Y, %I:%M %p")}</div>

        <div class="cards">
            <div class="card">
                <div class="label">Portfolio Value</div>
                <div class="value">${portfolio_value:,.2f}</div>
            </div>
            <div class="card">
                <div class="label">Cash Available</div>
                <div class="value">${cash:,.2f}</div>
            </div>
            <div class="card">
                <div class="label">Total Return</div>
                <div class="value" style="color:{return_color}">{total_return:+.2f}%</div>
            </div>
            <div class="card">
                <div class="label">Alpha vs S&P 500</div>
                <div class="value" style="color:{alpha_color}">{alpha:+.2f}%</div>
            </div>
        </div>

        <div class="section-title">📈 Open Positions
            <span style="color:{pnl_color}; font-size:15px; margin-left:10px">
                Total PnL: ${total_pnl:+,.2f}
            </span>
        </div>
        <table>
            <tr>
                <th>Symbol</th><th>Shares</th><th>Avg Entry</th>
                <th>Current Price</th><th>PnL ($)</th><th>PnL (%)</th>
            </tr>
            {positions_html if positions_html else '<tr><td colspan="6" style="text-align:center;color:#8b949e">No open positions</td></tr>'}
        </table>

        <div class="section-title">📋 Recent Trades</div>
        <table>
            <tr><th>Time</th><th>Stock</th><th>Action</th><th>Price</th></tr>
            {trades_html if trades_html else '<tr><td colspan="4" style="text-align:center;color:#8b949e">No trades yet</td></tr>'}
        </table>
    </body>
    </html>
    """

if __name__ == "__main__":
    print("Dashboard running at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.")
    app.run(debug=False)