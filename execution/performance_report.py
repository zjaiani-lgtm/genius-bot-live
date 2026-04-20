from execution.db.db import init_db
from execution.db.repository import get_trade_stats


def main():
    init_db()
    s = get_trade_stats()

    print("=== GENIUS BOT MAN | PERFORMANCE REPORT ===")
    print(f"Closed trades:   {s['closed_trades']}")
    print(f"Wins / Losses:   {s['wins']} / {s['losses']}")
    print(f"Winrate %:       {s['winrate_pct']:.2f}")
    print(f"ROI %:           {s['roi_pct']:.2f}")
    print(f"PnL sum (quote): {s['pnl_quote_sum']:.4f}")
    print(f"Quote in sum:    {s['quote_in_sum']:.4f}")
    print(f"Profit factor:   {s['profit_factor']:.3f}")
    print(f"Gross profit:    {s['gross_profit']:.4f}")
    print(f"Gross loss:      {s['gross_loss']:.4f}")
    print(f"Avg win:         {s['avg_win']:.4f}")
    print(f"Avg loss:        {s['avg_loss']:.4f}")
    print(f"Expectancy:      {s['expectancy_quote']:.4f}")
    print(f"Open trades:     {s['open_trades']}")
    print(f"Open quote in:   {s['open_quote_in_sum']:.4f}")


if __name__ == "__main__":
    main()
