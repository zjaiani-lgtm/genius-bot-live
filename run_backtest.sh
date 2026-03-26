#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# GENIUS BOT — Backtest Runner for Render Shell
# გამოყენება: bash run_backtest.sh [mode] [symbols]
# ═══════════════════════════════════════════════════════════════

set -e
cd /opt/render/project/src

echo "═══════════════════════════════════════════"
echo "  GENIUS BOT BACKTEST RUNNER"
echo "═══════════════════════════════════════════"
echo ""

# ── Dependencies ─────────────────────────────
echo "📦 Installing dependencies..."
pip install numpy pandas ccxt openpyxl scipy \
    --break-system-packages -q \
    --index-url https://pypi.org/simple/ \
    2>/dev/null || pip install numpy pandas ccxt openpyxl scipy -q 2>/dev/null
echo "✅ Dependencies ready"
echo ""

# ── Mode ─────────────────────────────────────
MODE="${1:-ohlcv}"
SYMBOLS="${2:-BTC/USDT ETH/USDT BNB/USDT}"

echo "🚀 Mode: $MODE"
echo "📊 Symbols: $SYMBOLS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Run backtest ──────────────────────────────
LOG_FILE="/var/data/backtest_$(date +%Y%m%d_%H%M%S).log"

python backtest.py --mode "$MODE" --symbols $SYMBOLS 2>&1 | tee "$LOG_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Done! Results saved:"
echo "   Log:   $LOG_FILE"

# Copy results to /var/data
[ -f backtest_v3_results.xlsx ] && \
    cp backtest_v3_results.xlsx /var/data/ && \
    echo "   Excel: /var/data/backtest_v3_results.xlsx"

[ -f backtest_v3_report.html ] && \
    cp backtest_v3_report.html /var/data/ && \
    echo "   HTML:  /var/data/backtest_v3_report.html"

echo "═══════════════════════════════════════════"
