"""Движок бэктеста: симуляция сделок, портфельные ограничения, статистика."""
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from indicators import pct, sma
from strategy import DAY_MS, Params, RawSignal, evaluate_bar

log = logging.getLogger("backtest")

H4_MS = 14_400_000


@dataclass
class Trade:
    symbol: str
    entry_ts: int
    exit_ts: int
    entry: float
    exit: float
    sl: float
    tp: float
    sl_pct: float
    tp_pct: float
    rr: float
    outcome: str              # tp | sl | timeout
    pnl_pct: float            # с учётом комиссий и проскальзывания
    hours: float
    mae_pct: float            # худшая просадка внутри сделки
    mfe_pct: float            # лучший ход в плюс
    drop_3d: float
    rsi_4h: float
    atr_pct: float
    notes: int
    btc_up: bool = True       # BTC выше своей 50-дневной средней


def simulate_symbol(symbol: str, daily: List[dict], h4: List[dict],
                    p: Params, from_ts: int = 0) -> List[Trade]:
    """Проходит по 4ч свечам, ищет входы и доводит каждую сделку до исхода."""
    trades: List[Trade] = []
    if len(h4) < 120 or len(daily) < 95:
        return trades

    max_bars = max(1, int(p.track_hours * 3600_000 / H4_MS))
    cooldown_ms = p.cooldown_hours * 3600_000
    daily_ts = [d["ts"] for d in daily]
    last_signal_ts = 0
    i = 60
    while i < len(h4) - 1:
        bar_ts = h4[i]["ts"]
        if from_ts and bar_ts < from_ts:
            i += 1
            continue
        if bar_ts - last_signal_ts < cooldown_ms:
            i += 1
            continue

        sig = evaluate_bar(symbol, daily, h4, i, p, daily_ts)
        if not sig:
            i += 1
            continue
        last_signal_ts = bar_ts

        trade = _resolve(sig, h4, i, max_bars, p)
        if trade:
            trades.append(trade)
        i += 1
    return trades


def _resolve(sig: RawSignal, h4: List[dict], i: int, max_bars: int,
             p: Params) -> Optional[Trade]:
    """Ведёт сделку вперёд по свечам до тейка, стопа или истечения срока.

    Если внутри одной свечи задеты и стоп, и тейк, засчитываем стоп —
    по минутным данным порядок неизвестен, поэтому берём худший вариант.
    """
    entry = sig.entry
    worst = entry
    best = entry
    end = min(len(h4) - 1, i + max_bars)
    for j in range(i + 1, end + 1):
        bar = h4[j]
        worst = min(worst, bar["low"])
        best = max(best, bar["high"])
        hit_sl = bar["low"] <= sig.sl
        hit_tp = bar["high"] >= sig.tp
        if hit_sl:
            return _make_trade(sig, bar["ts"], sig.sl, "sl", worst, best, p)
        if hit_tp:
            return _make_trade(sig, bar["ts"], sig.tp, "tp", worst, best, p)
    # срок вышел
    last = h4[end]
    return _make_trade(sig, last["ts"], last["close"], "timeout", worst, best, p)


def _make_trade(sig: RawSignal, exit_ts: int, exit_price: float,
                outcome: str, worst: float, best: float, p: Params) -> Trade:
    gross = (exit_price - sig.entry) / sig.entry * 100.0
    costs = p.fee_pct * 2 + p.slippage_pct
    return Trade(
        symbol=sig.symbol, entry_ts=sig.ts, exit_ts=exit_ts,
        entry=sig.entry, exit=exit_price, sl=sig.sl, tp=sig.tp,
        sl_pct=sig.sl_pct, tp_pct=sig.tp_pct, rr=sig.rr,
        outcome=outcome, pnl_pct=gross - costs,
        hours=(exit_ts - sig.ts) / 3600_000.0,
        mae_pct=(worst - sig.entry) / sig.entry * 100.0,
        mfe_pct=(best - sig.entry) / sig.entry * 100.0,
        drop_3d=sig.drop_3d, rsi_4h=sig.rsi_4h, atr_pct=sig.atr_pct,
        notes=sig.notes)


def btc_regime_map(btc_daily: List[dict]) -> Dict[int, bool]:
    """Для каждого дня: BTC выше своей 50-дневной средней или ниже."""
    out: Dict[int, bool] = {}
    closes = [d["close"] for d in btc_daily]
    for idx, d in enumerate(btc_daily):
        if idx < 50:
            continue
        m = sma(closes[:idx + 1], 50)
        if m:
            out[d["ts"] - (d["ts"] % DAY_MS)] = d["close"] >= m
    return out


def apply_portfolio(trades: List[Trade], p: Params) -> List[Trade]:
    """Ограничивает число одновременно открытых сделок."""
    trades = sorted(trades, key=lambda t: t.entry_ts)
    open_until: List[int] = []
    kept: List[Trade] = []
    for t in trades:
        open_until = [ts for ts in open_until if ts > t.entry_ts]
        if len(open_until) >= p.max_concurrent:
            continue
        kept.append(t)
        open_until.append(t.exit_ts)
    return kept


def equity_curve(trades: List[Trade], p: Params,
                 start_equity: float = 1000.0) -> List[dict]:
    """Кривая капитала при фиксированном риске на сделку."""
    eq = start_equity
    curve = [{"ts": trades[0].entry_ts if trades else 0, "equity": eq}]
    for t in sorted(trades, key=lambda x: x.exit_ts):
        if t.sl_pct <= 0:
            continue
        mult = 1.0 + (p.risk_pct / 100.0) * (t.pnl_pct / t.sl_pct)
        eq *= max(0.0, mult)
        curve.append({"ts": t.exit_ts, "equity": eq})
    return curve


def stats(trades: List[Trade], p: Params) -> dict:
    if not trades:
        return {"trades": 0, "winrate": 0.0, "expectancy": 0.0,
                "profit_factor": 0.0, "total_pct": 0.0, "equity_mult": 1.0,
                "max_dd": 0.0, "avg_hours": 0.0, "tp": 0, "sl": 0,
                "timeout": 0, "avg_win": 0.0, "avg_loss": 0.0,
                "best": 0.0, "worst": 0.0, "avg_mae": 0.0, "avg_mfe": 0.0,
                "by_regime": {}, "by_symbol": [], "monthly": []}

    tp = sum(1 for t in trades if t.outcome == "tp")
    sl = sum(1 for t in trades if t.outcome == "sl")
    to = sum(1 for t in trades if t.outcome == "timeout")
    pnls = [t.pnl_pct for t in trades]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]

    curve = equity_curve(trades, p)
    eqs = [c["equity"] for c in curve]
    peak = eqs[0]
    max_dd = 0.0
    for e in eqs:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak * 100.0)

    # разбивка по режиму рынка
    by_regime: Dict[str, dict] = {}
    for label, subset in (("BTC в росте", [t for t in trades if t.btc_up]),
                          ("BTC в падении",
                           [t for t in trades if not t.btc_up])):
        if not subset:
            by_regime[label] = {"trades": 0, "winrate": 0.0,
                                "expectancy": 0.0}
            continue
        w = sum(1 for t in subset if t.pnl_pct > 0)
        by_regime[label] = {
            "trades": len(subset),
            "winrate": w / len(subset) * 100.0,
            "expectancy": sum(t.pnl_pct for t in subset) / len(subset)}

    # по монетам
    per: Dict[str, list] = {}
    for t in trades:
        per.setdefault(t.symbol, []).append(t)
    by_symbol = sorted(
        [{"symbol": s,
          "trades": len(v),
          "winrate": sum(1 for t in v if t.pnl_pct > 0) / len(v) * 100.0,
          "total": sum(t.pnl_pct for t in v)}
         for s, v in per.items()],
        key=lambda x: x["total"], reverse=True)

    # по месяцам
    months: Dict[str, list] = {}
    for t in trades:
        key = time.strftime("%Y-%m", time.gmtime(t.entry_ts / 1000))
        months.setdefault(key, []).append(t.pnl_pct)
    monthly = [{"month": k, "trades": len(v), "total": sum(v)}
               for k, v in sorted(months.items())]

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(trades),
        "tp": tp, "sl": sl, "timeout": to,
        "winrate": len(wins) / len(trades) * 100.0,
        "expectancy": sum(pnls) / len(pnls),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else 999.0,
        "total_pct": sum(pnls),
        "equity_mult": eqs[-1] / eqs[0] if eqs and eqs[0] else 1.0,
        "max_dd": max_dd,
        "avg_hours": sum(t.hours for t in trades) / len(trades),
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "best": max(pnls), "worst": min(pnls),
        "avg_mae": sum(t.mae_pct for t in trades) / len(trades),
        "avg_mfe": sum(t.mfe_pct for t in trades) / len(trades),
        "by_regime": by_regime,
        "by_symbol": by_symbol[:25],
        "monthly": monthly,
    }


def breakeven_winrate(p: Params, avg_sl_pct: float = None) -> float:
    """Какой винрейт нужен, чтобы выйти в ноль при текущих цели и стопе."""
    sl = avg_sl_pct if avg_sl_pct else p.sl_fixed_pct
    costs = p.fee_pct * 2 + p.slippage_pct
    win = p.tp_pct - costs
    loss = sl + costs
    if win + loss <= 0:
        return 100.0
    return loss / (win + loss) * 100.0
