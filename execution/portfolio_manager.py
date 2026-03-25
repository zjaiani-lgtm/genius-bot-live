# execution/portfolio_manager.py
# ============================================================
# GENIUS BOT — Multi-Bot Portfolio Manager
# ============================================================
# არქიტექტურა:
#   PortfolioManager — კაპიტალის განაწილება + პროფილების მართვა
#   BotProfile        — aggressive / moderate / conservative
#   CapitalAllocator  — per-bot quote limit გამოთვლა
#
# გამოყენება .env-ში:
#   PORTFOLIO_ENABLED=true
#   PORTFOLIO_PROFILE=moderate          # aggressive / moderate / conservative
#   PORTFOLIO_TOTAL_CAPITAL=150         # სრული USDT ბიუჯეტი ყველა bot-სთვის
#   PORTFOLIO_BOT_IDS=bot1,bot2,bot3    # ბოტების ID-ები
#   PORTFOLIO_BOT1_PROFILE=aggressive
#   PORTFOLIO_BOT2_PROFILE=moderate
#   PORTFOLIO_BOT3_PROFILE=conservative
#
# Per-Symbol isolation: უკვე გვაქვს (MAX_POSITIONS_PER_SYMBOL=1) ✅
# ============================================================

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("portfolio_manager")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _ef(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, AttributeError):
        return default

def _ei(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, AttributeError):
        return default

def _eb(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")

def _es(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# ─────────────────────────────────────────────
# PROFILE DEFINITIONS
# ─────────────────────────────────────────────

@dataclass
class BotProfile:
    """
    სტრატეგიის პროფილი ერთი ბოტისთვის.

    aggressive:   მეტი trade, მეტი capital, ფართო SL
    moderate:     ბალანსირებული (default genius-bot settings)
    conservative: ნაკლები trade, მცირე capital, ვიწრო SL
    """
    name: str

    # Capital allocation (% of total portfolio budget)
    capital_share: float        # 0.0..1.0 — ამ bot-ს მიენიჭება total × share

    # Trade sizing
    quote_size_bull: float      # USDT per BULL trade
    quote_size_uncertain: float # USDT per UNCERTAIN trade

    # Risk
    sl_pct: float               # Stop Loss %
    tp_pct: float               # Take Profit %
    max_open_trades: int        # ერთდროული ღია trade-ების მაქსიმუმი
    max_trades_per_day: int
    max_consecutive_losses: int
    max_daily_loss_pct: float   # % of bot's allocated capital

    # Regime filters
    allow_uncertain: bool       # UNCERTAIN regime-ზე trade?
    min_ai_score: float         # AI score minimum

    # Extra labels
    description: str = ""


# Built-in profiles — ENV-ით override შესაძლებელია
PROFILES: Dict[str, BotProfile] = {
    "aggressive": BotProfile(
        name="aggressive",
        capital_share=0.40,
        quote_size_bull=20.0,
        quote_size_uncertain=12.0,
        sl_pct=0.80,
        tp_pct=1.2,
        max_open_trades=6,
        max_trades_per_day=40,
        max_consecutive_losses=3,
        max_daily_loss_pct=4.0,
        allow_uncertain=True,
        min_ai_score=0.44,
        description="მეტი trade, მეტი capital, UNCERTAIN დაშვებული",
    ),
    "moderate": BotProfile(
        name="moderate",
        capital_share=0.35,
        quote_size_bull=15.0,
        quote_size_uncertain=7.0,
        sl_pct=0.70,
        tp_pct=1.0,
        max_open_trades=5,
        max_trades_per_day=30,
        max_consecutive_losses=2,
        max_daily_loss_pct=2.5,
        allow_uncertain=True,
        min_ai_score=0.52,
        description="ბალანსირებული — default GENIUS settings",
    ),
    "conservative": BotProfile(
        name="conservative",
        capital_share=0.25,
        quote_size_bull=10.0,
        quote_size_uncertain=5.0,
        sl_pct=0.55,
        tp_pct=0.8,
        max_open_trades=3,
        max_trades_per_day=15,
        max_consecutive_losses=2,
        max_daily_loss_pct=1.5,
        allow_uncertain=False,
        min_ai_score=0.60,
        description="ნაკლები trade, მჭიდრო SL, მხოლოდ BULL",
    ),
}


# ─────────────────────────────────────────────
# CAPITAL ALLOCATOR
# ─────────────────────────────────────────────

@dataclass
class CapitalAllocator:
    """
    სრული portfolio budget-ის გამოთვლა per-bot allocation-ით.

    total_capital: სრული USDT ბიუჯეტი (ყველა ბოტი ერთად)
    bots: bot_id → BotProfile
    """
    total_capital: float
    bots: Dict[str, BotProfile] = field(default_factory=dict)

    def allocate(self) -> Dict[str, float]:
        """
        Returns: {bot_id: allocated_usdt}
        shares-ის ჯამი > 1.0-ზე → proportional normalization.
        """
        total_share = sum(p.capital_share for p in self.bots.values())
        if total_share <= 0:
            return {bot_id: 0.0 for bot_id in self.bots}

        result = {}
        for bot_id, profile in self.bots.items():
            norm_share = profile.capital_share / total_share
            result[bot_id] = round(self.total_capital * norm_share, 2)

        logger.info(
            f"[PORTFOLIO] capital allocated: total={self.total_capital:.2f} USDT | "
            + " | ".join(f"{bid}={amt:.2f}" for bid, amt in result.items())
        )
        return result

    def max_quote_for_bot(self, bot_id: str, regime: str = "BULL") -> float:
        """
        ამ ბოტის ერთი trade-ის მაქსიმუმი — profile-ის მიხედვით.
        """
        profile = self.bots.get(bot_id)
        if not profile:
            return 0.0
        if regime == "BULL":
            return profile.quote_size_bull
        return profile.quote_size_uncertain

    def can_trade(self, bot_id: str, regime: str, open_trades: int,
                  consecutive_losses: int, daily_loss_usdt: float) -> Tuple[bool, str]:
        """
        ამ ბოტს შეუძლია trade? → (True/False, reason)
        """
        profile = self.bots.get(bot_id)
        if not profile:
            return False, f"BOT_NOT_FOUND:{bot_id}"

        if regime == "UNCERTAIN" and not profile.allow_uncertain:
            return False, f"UNCERTAIN_BLOCKED_BY_PROFILE:{profile.name}"

        if open_trades >= profile.max_open_trades:
            return False, f"MAX_OPEN_TRADES:{open_trades}>={profile.max_open_trades}"

        if consecutive_losses >= profile.max_consecutive_losses:
            return False, f"CONSEC_LOSSES:{consecutive_losses}>={profile.max_consecutive_losses}"

        allocation = self.allocate().get(bot_id, 0.0)
        if allocation > 0:
            daily_loss_pct = (daily_loss_usdt / allocation) * 100.0
            if daily_loss_pct >= profile.max_daily_loss_pct:
                return False, f"DAILY_LOSS:{daily_loss_pct:.2f}%>={profile.max_daily_loss_pct}%"

        return True, "OK"


# ─────────────────────────────────────────────
# PORTFOLIO MANAGER
# ─────────────────────────────────────────────

class PortfolioManager:
    """
    Multi-Bot Portfolio-ს სრული მართვა.

    გამოყენება:
        pm = PortfolioManager.from_env()
        pm.print_summary()

        ok, reason = pm.can_bot_trade("bot1", regime="BULL", ...)
        quote = pm.get_quote_for_bot("bot1", regime="BULL")
    """

    def __init__(
        self,
        enabled: bool,
        total_capital: float,
        bots: Dict[str, BotProfile],
    ):
        self.enabled = enabled
        self.total_capital = total_capital
        self.bots = bots
        self.allocator = CapitalAllocator(
            total_capital=total_capital,
            bots=bots,
        )
        self._allocation_cache: Optional[Dict[str, float]] = None

    # ─────────────────────────────────
    # Factory
    # ─────────────────────────────────

    @classmethod
    def from_env(cls) -> "PortfolioManager":
        """
        .env-დან PortfolioManager-ის შექმნა.

        .env მინიმუმი:
            PORTFOLIO_ENABLED=true
            PORTFOLIO_TOTAL_CAPITAL=150
            PORTFOLIO_BOT_IDS=bot1,bot2,bot3
            PORTFOLIO_BOT1_PROFILE=aggressive
            PORTFOLIO_BOT2_PROFILE=moderate
            PORTFOLIO_BOT3_PROFILE=conservative

        Optional per-bot override:
            PORTFOLIO_BOT1_QUOTE_BULL=25
            PORTFOLIO_BOT1_SL_PCT=0.9
        """
        enabled       = _eb("PORTFOLIO_ENABLED", "false")
        total_capital = _ef("PORTFOLIO_TOTAL_CAPITAL", 150.0)
        bot_ids_str   = _es("PORTFOLIO_BOT_IDS", "bot1,bot2,bot3")
        bot_ids       = [b.strip() for b in bot_ids_str.split(",") if b.strip()]

        bots: Dict[str, BotProfile] = {}
        for bot_id in bot_ids:
            key        = bot_id.upper()
            prof_name  = _es(f"PORTFOLIO_{key}_PROFILE", "moderate").lower()
            base       = PROFILES.get(prof_name, PROFILES["moderate"])

            # per-bot ENV overrides (optional)
            profile = BotProfile(
                name=prof_name,
                capital_share=_ef(f"PORTFOLIO_{key}_CAPITAL_SHARE",  base.capital_share),
                quote_size_bull=_ef(f"PORTFOLIO_{key}_QUOTE_BULL",   base.quote_size_bull),
                quote_size_uncertain=_ef(f"PORTFOLIO_{key}_QUOTE_UNC", base.quote_size_uncertain),
                sl_pct=_ef(f"PORTFOLIO_{key}_SL_PCT",                base.sl_pct),
                tp_pct=_ef(f"PORTFOLIO_{key}_TP_PCT",                base.tp_pct),
                max_open_trades=_ei(f"PORTFOLIO_{key}_MAX_OPEN",     base.max_open_trades),
                max_trades_per_day=_ei(f"PORTFOLIO_{key}_MAX_DAILY", base.max_trades_per_day),
                max_consecutive_losses=_ei(f"PORTFOLIO_{key}_MAX_CONSEC", base.max_consecutive_losses),
                max_daily_loss_pct=_ef(f"PORTFOLIO_{key}_MAX_LOSS_PCT",  base.max_daily_loss_pct),
                allow_uncertain=_eb(f"PORTFOLIO_{key}_ALLOW_UNCERTAIN",
                                    "true" if base.allow_uncertain else "false"),
                min_ai_score=_ef(f"PORTFOLIO_{key}_MIN_AI",          base.min_ai_score),
                description=base.description,
            )
            bots[bot_id] = profile

        pm = cls(enabled=enabled, total_capital=total_capital, bots=bots)
        if enabled:
            logger.info(f"[PORTFOLIO] initialized: {len(bots)} bots, total={total_capital:.2f} USDT")
        return pm

    # ─────────────────────────────────
    # Core API
    # ─────────────────────────────────

    def get_allocation(self) -> Dict[str, float]:
        if self._allocation_cache is None:
            self._allocation_cache = self.allocator.allocate()
        return self._allocation_cache

    def get_profile(self, bot_id: str) -> Optional[BotProfile]:
        return self.bots.get(bot_id)

    def can_bot_trade(
        self,
        bot_id: str,
        regime: str,
        open_trades: int = 0,
        consecutive_losses: int = 0,
        daily_loss_usdt: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        True/False + reason. PortfolioManager გათიშულია → always True.
        """
        if not self.enabled:
            return True, "PORTFOLIO_DISABLED"
        return self.allocator.can_trade(
            bot_id=bot_id,
            regime=regime,
            open_trades=open_trades,
            consecutive_losses=consecutive_losses,
            daily_loss_usdt=daily_loss_usdt,
        )

    def get_quote_for_bot(self, bot_id: str, regime: str = "BULL") -> float:
        """
        ამ ბოტის trade-ის quote amount — profile-ის მიხედვით.
        PortfolioManager გათიშულია → 0.0 (caller იყენებს default-ს).
        """
        if not self.enabled:
            return 0.0
        return self.allocator.max_quote_for_bot(bot_id, regime)

    def get_sl_pct(self, bot_id: str, fallback: float = 0.70) -> float:
        profile = self.bots.get(bot_id)
        return profile.sl_pct if profile else fallback

    def get_tp_pct(self, bot_id: str, fallback: float = 1.0) -> float:
        profile = self.bots.get(bot_id)
        return profile.tp_pct if profile else fallback

    def get_min_ai_score(self, bot_id: str, fallback: float = 0.52) -> float:
        profile = self.bots.get(bot_id)
        return profile.min_ai_score if profile else fallback

    # ─────────────────────────────────
    # Summary / Debug
    # ─────────────────────────────────

    def print_summary(self):
        print("\n" + "=" * 60)
        print("  GENIUS BOT — PORTFOLIO MANAGER SUMMARY")
        print("=" * 60)
        print(f"  Status  : {'ENABLED ✅' if self.enabled else 'DISABLED ⛔'}")
        print(f"  Capital : {self.total_capital:.2f} USDT")
        print(f"  Bots    : {len(self.bots)}")
        print("-" * 60)

        allocation = self.get_allocation()
        for bot_id, profile in self.bots.items():
            alloc = allocation.get(bot_id, 0.0)
            print(f"  [{bot_id}]  profile={profile.name:<13} alloc={alloc:.2f} USDT")
            print(f"         quote_bull={profile.quote_size_bull}  quote_unc={profile.quote_size_uncertain}")
            print(f"         SL={profile.sl_pct}%  TP={profile.tp_pct}%  max_open={profile.max_open_trades}")
            print(f"         uncertain={'✅' if profile.allow_uncertain else '⛔'}  min_ai={profile.min_ai_score}")
            print()
        print("=" * 60 + "\n")

    def to_dict(self) -> dict:
        allocation = self.get_allocation()
        return {
            "enabled": self.enabled,
            "total_capital": self.total_capital,
            "bots": {
                bot_id: {
                    "profile": p.name,
                    "allocation_usdt": allocation.get(bot_id, 0.0),
                    "quote_bull": p.quote_size_bull,
                    "quote_uncertain": p.quote_size_uncertain,
                    "sl_pct": p.sl_pct,
                    "tp_pct": p.tp_pct,
                    "max_open_trades": p.max_open_trades,
                    "allow_uncertain": p.allow_uncertain,
                    "min_ai_score": p.min_ai_score,
                }
                for bot_id, p in self.bots.items()
            },
        }


# ─────────────────────────────────────────────
# SINGLETON — main.py იყენებს
# ─────────────────────────────────────────────

_portfolio_manager: Optional[PortfolioManager] = None


def get_portfolio_manager() -> PortfolioManager:
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = PortfolioManager.from_env()
    return _portfolio_manager


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # .env გარეშე test
    os.environ.update({
        "PORTFOLIO_ENABLED": "true",
        "PORTFOLIO_TOTAL_CAPITAL": "150",
        "PORTFOLIO_BOT_IDS": "bot1,bot2,bot3",
        "PORTFOLIO_BOT1_PROFILE": "aggressive",
        "PORTFOLIO_BOT2_PROFILE": "moderate",
        "PORTFOLIO_BOT3_PROFILE": "conservative",
    })

    pm = PortfolioManager.from_env()
    pm.print_summary()

    # can_trade tests
    tests = [
        # (bot_id, regime, open_trades, consec_losses, daily_loss_usdt, expected)
        ("bot1", "BULL",      0, 0, 0.0,   True),    # normal bull ✓
        ("bot2", "UNCERTAIN", 0, 0, 0.0,   True),    # moderate: uncertain ok ✓
        ("bot3", "UNCERTAIN", 0, 0, 0.0,   False),   # conservative: uncertain blocked ✓
        ("bot2", "BULL",      4, 0, 0.0,   True),    # moderate open=4 < max=5 ✓
        ("bot2", "BULL",      5, 0, 0.0,   False),   # moderate open=5 >= max=5 ✓
        ("bot2", "BULL",      6, 0, 0.0,   False),   # moderate open=6 >= max=5 ✓
        ("bot1", "BULL",      0, 3, 0.0,   False),   # aggressive consec=3 >= max=3 ✓
        ("bot3", "BULL",      0, 0, 0.4,   True),    # conservative alloc=37.5, loss=0.4 = 1.06% < 1.5% ✓
        ("bot3", "BULL",      0, 0, 0.6,   False),   # conservative alloc=37.5, loss=0.6 = 1.6% > 1.5% ✓
    ]

    print("can_trade tests:")
    all_ok = True
    for bot_id, regime, open_t, consec, loss, expected in tests:
        ok, reason = pm.can_bot_trade(bot_id, regime, open_t, consec, loss)
        status = "✅" if ok == expected else "❌"
        if ok != expected:
            all_ok = False
        print(f"  {status} {bot_id} {regime:10} open={open_t} consec={consec} → {ok} ({reason})")

    print(f"\n{'✅ ALL TESTS PASSED' if all_ok else '❌ SOME TESTS FAILED'}")
