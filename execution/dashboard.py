# execution/dashboard.py
# ============================================================
# GENIUS BOT — Live Trading Dashboard
# Flask web server — გაუშვი ბოტთან ერთად main.py-დან
# URL: https://your-render-url.onrender.com/dashboard
# ============================================================
from __future__ import annotations

import os
import threading
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger("gbm")

# ── Flask import (optional — თუ არ არის, dashboard გათიშულია) ──
try:
    from flask import Flask, jsonify, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("[DASHBOARD] Flask not installed → dashboard disabled")


DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GENIUS BOT — Live Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:      #0a0a0f;
    --bg2:     #111118;
    --bg3:     #1a1a24;
    --border:  #2a2a3a;
    --green:   #00ff88;
    --green2:  #00cc6a;
    --red:     #ff4466;
    --gold:    #ffd700;
    --gold2:   #ffaa00;
    --cyan:    #00ccff;
    --text:    #e8e8f0;
    --muted:   #666688;
    --font:    'Space Mono', monospace;
    --display: 'Bebas Neue', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* animated grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,255,136,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,255,136,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .wrap {
    position: relative;
    z-index: 1;
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px 20px;
  }

  /* ── HEADER ── */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }

  .logo {
    display: flex;
    flex-direction: column;
  }

  .logo-title {
    font-family: var(--display);
    font-size: 42px;
    letter-spacing: 3px;
    background: linear-gradient(135deg, var(--gold), var(--green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
  }

  .logo-sub {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 4px;
  }

  .live-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.3);
    padding: 8px 16px;
    border-radius: 4px;
    font-size: 12px;
    color: var(--green);
    letter-spacing: 2px;
  }

  .live-dot {
    width: 8px;
    height: 8px;
    background: var(--green);
    border-radius: 50%;
    animation: pulse 1.5s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
  }

  /* ── STATS GRID ── */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }

  @media (max-width: 900px) {
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
  }

  .stat-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    padding: 20px;
    border-radius: 2px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
  }

  .stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--green), transparent);
  }

  .stat-card:hover { border-color: rgba(0,255,136,0.3); }

  .stat-label {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
  }

  .stat-value {
    font-family: var(--display);
    font-size: 36px;
    letter-spacing: 1px;
    line-height: 1;
  }

  .stat-value.green { color: var(--green); }
  .stat-value.gold  { color: var(--gold); }
  .stat-value.cyan  { color: var(--cyan); }
  .stat-value.white { color: var(--text); }

  .stat-sub {
    font-size: 10px;
    color: var(--muted);
    margin-top: 6px;
  }

  /* ── MAIN CONTENT ── */
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }

  @media (max-width: 900px) {
    .main-grid { grid-template-columns: 1fr; }
  }

  .panel {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 2px;
    overflow: hidden;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--bg3);
  }

  .panel-title {
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
  }

  .panel-badge {
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 2px;
    letter-spacing: 1px;
  }

  .badge-green { background: rgba(0,255,136,0.12); color: var(--green); }
  .badge-gold  { background: rgba(255,215,0,0.12);  color: var(--gold); }

  /* ── POSITIONS TABLE ── */
  .positions-table {
    width: 100%;
    border-collapse: collapse;
  }

  .positions-table th {
    font-size: 9px;
    letter-spacing: 2px;
    color: var(--muted);
    text-align: left;
    padding: 10px 20px;
    border-bottom: 1px solid var(--border);
    text-transform: uppercase;
  }

  .positions-table td {
    padding: 12px 20px;
    font-size: 12px;
    border-bottom: 1px solid rgba(42,42,58,0.5);
  }

  .positions-table tr:last-child td { border-bottom: none; }

  .positions-table tr:hover td {
    background: rgba(255,255,255,0.02);
  }

  .sym-tag {
    display: inline-block;
    background: rgba(0,204,255,0.1);
    color: var(--cyan);
    padding: 2px 8px;
    border-radius: 2px;
    font-size: 11px;
    letter-spacing: 1px;
  }

  .sym-tag.l2 {
    background: rgba(255,215,0,0.1);
    color: var(--gold);
  }

  .tp-bar {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .tp-progress {
    flex: 1;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }

  .tp-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--green2), var(--green));
    border-radius: 2px;
    transition: width 1s ease;
  }

  .tp-pct {
    font-size: 10px;
    color: var(--green);
    min-width: 40px;
    text-align: right;
  }

  /* ── TRADE FEED ── */
  .feed-list {
    padding: 8px 0;
    max-height: 320px;
    overflow-y: auto;
  }

  .feed-list::-webkit-scrollbar { width: 4px; }
  .feed-list::-webkit-scrollbar-track { background: var(--bg2); }
  .feed-list::-webkit-scrollbar-thumb { background: var(--border); }

  .feed-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 20px;
    border-bottom: 1px solid rgba(42,42,58,0.4);
    animation: slideIn 0.4s ease;
  }

  @keyframes slideIn {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: translateX(0); }
  }

  .feed-icon {
    width: 28px;
    height: 28px;
    border-radius: 2px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    flex-shrink: 0;
  }

  .icon-tp  { background: rgba(0,255,136,0.12); color: var(--green); }
  .icon-sl  { background: rgba(255,68,102,0.12); color: var(--red); }
  .icon-buy { background: rgba(0,204,255,0.12);  color: var(--cyan); }
  .icon-cas { background: rgba(255,215,0,0.12);  color: var(--gold); }

  .feed-content { flex: 1; }

  .feed-sym {
    font-size: 12px;
    font-weight: 700;
    color: var(--text);
  }

  .feed-detail {
    font-size: 10px;
    color: var(--muted);
    margin-top: 2px;
  }

  .feed-pnl {
    font-size: 13px;
    font-weight: 700;
    text-align: right;
  }

  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }

  .feed-time {
    font-size: 9px;
    color: var(--muted);
    text-align: right;
    margin-top: 2px;
  }

  /* ── BOTTOM BAR ── */
  .bottom-bar {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
  }

  .mini-stat {
    background: var(--bg2);
    border: 1px solid var(--border);
    padding: 16px 20px;
    border-radius: 2px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .mini-label {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
  }

  .mini-value {
    font-family: var(--display);
    font-size: 22px;
    letter-spacing: 1px;
  }

  /* ── FOOTER ── */
  .footer {
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2px;
  }

  .footer-brand {
    background: linear-gradient(90deg, var(--gold), var(--green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    letter-spacing: 3px;
  }

  #last-update { color: var(--muted); }

  .no-data {
    padding: 30px 20px;
    text-align: center;
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 2px;
  }
</style>
</head>
<body>
<div class="wrap">

  <!-- HEADER -->
  <div class="header">
    <div class="logo">
      <div class="logo-title">GENIUS BOT</div>
      <div class="logo-sub">JAIANI &amp; CLAUDE · Live Trading Dashboard</div>
    </div>
    <div class="live-badge">
      <div class="live-dot"></div>
      LIVE
    </div>
  </div>

  <!-- TOP STATS -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Win Rate</div>
      <div class="stat-value green" id="winrate">—</div>
      <div class="stat-sub" id="wl-ratio">— / —</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total PnL</div>
      <div class="stat-value gold" id="total-pnl">—</div>
      <div class="stat-sub" id="roi">ROI —</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Open Trades</div>
      <div class="stat-value cyan" id="open-trades">—</div>
      <div class="stat-sub" id="open-capital">Capital: —</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Profit Factor</div>
      <div class="stat-value white" id="profit-factor">—</div>
      <div class="stat-sub" id="closed-trades">Closed: —</div>
    </div>
  </div>

  <!-- MAIN GRID -->
  <div class="main-grid">

    <!-- OPEN POSITIONS -->
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Open Positions</span>
        <span class="panel-badge badge-green" id="pos-count">0 active</span>
      </div>
      <table class="positions-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Avg Entry</th>
            <th>TP Target</th>
            <th>Progress</th>
          </tr>
        </thead>
        <tbody id="positions-body">
          <tr><td colspan="4" class="no-data">Loading...</td></tr>
        </tbody>
      </table>
    </div>

    <!-- TRADE FEED -->
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">Live Trade Feed</span>
        <span class="panel-badge badge-gold" id="feed-count">0 trades</span>
      </div>
      <div class="feed-list" id="feed-list">
        <div class="no-data">Loading...</div>
      </div>
    </div>

  </div>

  <!-- BOTTOM STATS -->
  <div class="bottom-bar">
    <div class="mini-stat">
      <div class="mini-label">Avg Win</div>
      <div class="mini-value" style="color:var(--green)" id="avg-win">—</div>
    </div>
    <div class="mini-stat">
      <div class="mini-label">Avg Loss</div>
      <div class="mini-value" style="color:var(--red)" id="avg-loss">—</div>
    </div>
    <div class="mini-stat">
      <div class="mini-label">Expectancy</div>
      <div class="mini-value" style="color:var(--gold)" id="expectancy">—</div>
    </div>
  </div>

  <!-- FOOTER -->
  <div class="footer">
    <span class="footer-brand">GENIUS BOT · JAIANI &amp; CLAUDE</span>
    <span id="last-update">Updating...</span>
  </div>

</div>

<script>
async function fetchData() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();

    // Stats
    const s = d.stats || {};
    const wr = s.winrate_pct != null ? s.winrate_pct.toFixed(2) + '%' : '—';
    document.getElementById('winrate').textContent = wr;
    document.getElementById('wl-ratio').textContent =
      `${s.wins || 0}W / ${s.losses || 0}L`;

    const pnl = s.pnl_quote_sum != null ? s.pnl_quote_sum : 0;
    document.getElementById('total-pnl').textContent =
      (pnl >= 0 ? '+' : '') + pnl.toFixed(4) + ' USDT';
    document.getElementById('total-pnl').className =
      'stat-value ' + (pnl >= 0 ? 'green' : 'red');
    document.getElementById('roi').textContent =
      'ROI ' + (s.roi_pct != null ? s.roi_pct.toFixed(2) + '%' : '—');

    document.getElementById('open-trades').textContent = s.open_trades || 0;
    document.getElementById('open-capital').textContent =
      'Capital: $' + (s.open_quote_in_sum != null ? s.open_quote_in_sum.toFixed(2) : '0');

    const pf = s.profit_factor != null ? s.profit_factor.toFixed(2) : '—';
    document.getElementById('profit-factor').textContent = pf;
    document.getElementById('closed-trades').textContent =
      'Closed: ' + (s.closed_trades || 0);

    document.getElementById('avg-win').textContent =
      s.avg_win != null ? '+$' + s.avg_win.toFixed(4) : '—';
    document.getElementById('avg-loss').textContent =
      s.avg_loss != null ? '$' + s.avg_loss.toFixed(4) : '—';
    document.getElementById('expectancy').textContent =
      s.expectancy_quote != null
        ? (s.expectancy_quote >= 0 ? '+' : '') + '$' + s.expectancy_quote.toFixed(4)
        : '—';

    // Positions
    const positions = d.positions || [];
    document.getElementById('pos-count').textContent = positions.length + ' active';
    const tbody = document.getElementById('positions-body');

    if (positions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="no-data">No open positions</td></tr>';
    } else {
      tbody.innerHTML = positions.map(p => {
        const sym = p.symbol || '';
        const isL2 = sym.includes('_L');
        const tag = isL2 ? 'l2' : '';
        const avg = parseFloat(p.avg_entry_price || 0);
        const tp  = parseFloat(p.current_tp_price || 0);
        // Progress: how close to TP (0% = just bought, 100% = at TP)
        // We estimate current price is between avg and tp
        const pctToTp = tp > 0 && avg > 0
          ? ((tp - avg) / avg * 100).toFixed(3)
          : '0.55';
        const progress = Math.min(100, Math.max(0,
          tp > 0 && avg > 0 ? 50 : 0));
        return `
          <tr>
            <td><span class="sym-tag ${tag}">${sym.replace('_L2','').replace('_L3','').replace('_L4','')}</span>
                ${isL2 ? '<span style="font-size:9px;color:var(--gold);margin-left:4px">L' + (sym.split('_L')[1] || '2') + '</span>' : ''}
            </td>
            <td style="color:var(--cyan)">${avg > 0 ? avg.toFixed(2) : '—'}</td>
            <td style="color:var(--green)">${tp > 0 ? tp.toFixed(2) : '—'}</td>
            <td>
              <div class="tp-bar">
                <div class="tp-progress">
                  <div class="tp-fill" style="width:${progress}%"></div>
                </div>
                <span class="tp-pct">+${pctToTp}%</span>
              </div>
            </td>
          </tr>`;
      }).join('');
    }

    // Trade Feed
    const trades = d.recent_trades || [];
    document.getElementById('feed-count').textContent = (s.closed_trades || 0) + ' trades';
    const feedEl = document.getElementById('feed-list');

    if (trades.length === 0) {
      feedEl.innerHTML = '<div class="no-data">No closed trades yet</div>';
    } else {
      feedEl.innerHTML = trades.slice(0, 15).map(t => {
        const outcome = (t.outcome || '').toUpperCase();
        let icon = '📈', iconClass = 'icon-buy', label = 'BUY';
        if (outcome === 'TP') { icon = '✅'; iconClass = 'icon-tp'; label = 'TAKE PROFIT'; }
        else if (outcome === 'SL') { icon = '❌'; iconClass = 'icon-sl'; label = 'STOP LOSS'; }
        else if (outcome === 'CASCADE_EXCHANGE') { icon = '🔄'; iconClass = 'icon-cas'; label = 'CASCADE'; }

        const pnl = parseFloat(t.pnl_quote || 0);
        const pnlStr = (pnl >= 0 ? '+' : '') + pnl.toFixed(4) + ' USDT';
        const pnlClass = pnl >= 0 ? 'pnl-pos' : 'pnl-neg';

        const ts = t.closed_at
          ? new Date(t.closed_at).toLocaleTimeString('en', {hour:'2-digit', minute:'2-digit'})
          : '';

        return `
          <div class="feed-item">
            <div class="feed-icon ${iconClass}">${icon}</div>
            <div class="feed-content">
              <div class="feed-sym">${t.symbol || '—'}</div>
              <div class="feed-detail">${label} · ${t.entry_price ? parseFloat(t.entry_price).toFixed(2) : '—'} → ${t.exit_price ? parseFloat(t.exit_price).toFixed(2) : '—'}</div>
            </div>
            <div>
              <div class="feed-pnl ${pnlClass}">${pnlStr}</div>
              <div class="feed-time">${ts}</div>
            </div>
          </div>`;
      }).join('');
    }

    document.getElementById('last-update').textContent =
      'Updated: ' + new Date().toLocaleTimeString();

  } catch(e) {
    console.error('Fetch error:', e);
    document.getElementById('last-update').textContent = 'Update failed — retrying...';
  }
}

fetchData();
setInterval(fetchData, 30000);
</script>
</body>
</html>'''


def create_dashboard_app():
    """Flask dashboard app — DB-დან real-time data."""
    if not FLASK_AVAILABLE:
        return None

    app = Flask(__name__)

    @app.route('/dashboard')
    def dashboard():
        return render_template_string(DASHBOARD_HTML)

    @app.route('/')
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route('/api/stats')
    def api_stats():
        try:
            from execution.db.repository import (
                get_trade_stats,
                get_all_open_dca_positions,
                get_closed_trades,
            )

            stats     = get_trade_stats()
            positions = get_all_open_dca_positions()
            trades    = get_closed_trades()

            # ბოლო 20 closed trade — უახლესი პირველი
            recent = sorted(
                [t for t in trades if t.get("outcome")],
                key=lambda x: str(x.get("closed_at", "")),
                reverse=True
            )[:20]

            return jsonify({
                "stats":         stats,
                "positions":     positions,
                "recent_trades": recent,
                "timestamp":     datetime.now(timezone.utc).isoformat(),
            })

        except Exception as e:
            logger.error(f"[DASHBOARD] API error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/health')
    def health():
        return jsonify({"status": "ok", "bot": "GENIUS BOT JAIANI\U0001f916"})

    return app


def start_dashboard(port: int = 8080):
    """Dashboard-ს background thread-ში გაუშვებს."""
    if not FLASK_AVAILABLE:
        logger.warning("[DASHBOARD] Flask not available — skipping")
        return

    app = create_dashboard_app()
    if app is None:
        return

    def run():
        logger.info(f"[DASHBOARD] Starting on port {port} → /dashboard")
        app.run(
            host="0.0.0.0",
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

    t = threading.Thread(target=run, daemon=True, name="dashboard")
    t.start()
    logger.info(f"[DASHBOARD] Live at http://0.0.0.0:{port}/dashboard")
