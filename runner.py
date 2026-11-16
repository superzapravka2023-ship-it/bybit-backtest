"""Связка: загрузка истории → симуляция → статистика. Плюс подбор параметров."""
import asyncio
import logging
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Optional

from backtest import (Trade, apply_portfolio, btc_regime_map, equity_curve,
                      breakeven_winrate, simulate_symbol, stats)
from data import BybitHistory, DAY_MS
from strategy import Params, day_start

log = logging.getLogger("runner")

WARMUP_DAILY = 220          # дней истории до начала периода для индикаторов
WARMUP_H4_DAYS = 40


async def load_symbol(hist: BybitHistory, symbol: str, start_ms: int,
                      end_ms: int) -> Optional[tuple]:
    """Качает (или берёт из кэша) дневные и 4-часовые свечи с запасом."""
    d_start = start_ms - WARMUP_DAILY * DAY_MS
    h_start = start_ms - WARMUP_H4_DAYS * DAY_MS
    daily = await hist.ensure(symbol, "D", d_start, end_ms)
    if len(daily) < 120:
        return None
    h4 = await hist.ensure(symbol, "240", h_start, end_ms)
    if len(h4) < 150:
        return None
    return daily, h4


async def load_many(hist: BybitHistory, symbols: List[str], start_ms: int,
                    end_ms: int, progress: Callable = None) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    total = len(symbols)
    for n, sym in enumerate(symbols, 1):
        if progress:
            progress("Загрузка истории", n, total, sym)
        try:
            data = await load_symbol(hist, sym, start_ms, end_ms)
        except Exception as e:  # noqa: BLE001
            log.warning("Не загрузилась %s: %s", sym, e)
            continue
        if data:
            out[sym] = data
        await asyncio.sleep(0)
    return out


def tag_regime(trades: List[Trade], regime: Dict[int, bool]):
    for t in trades:
        t.btc_up = regime.get(day_start(t.entry_ts), True)


def simulate_all(loaded: Dict[str, tuple], p: Params, start_ms: int,
                 regime: Dict[int, bool],
                 progress: Callable = None) -> List[Trade]:
    trades: List[Trade] = []
    total = len(loaded)
    for n, (sym, (daily, h4)) in enumerate(loaded.items(), 1):
        if progress:
            progress("Симуляция", n, total, sym)
        trades.extend(simulate_symbol(sym, daily, h4, p, from_ts=start_ms))
    tag_regime(trades, regime)
    return apply_portfolio(trades, p)


def period_bounds(months: int) -> tuple:
    end = int(time.time() * 1000)
    start = end - int(months * 30.44 * DAY_MS)
    return start, end


async def run_backtest(hist: BybitHistory, symbols: List[str], months: int,
                       p: Params, progress: Callable = None) -> dict:
    start_ms, end_ms = period_bounds(months)

    if progress:
        progress("Загрузка BTC", 0, 1, "BTCUSDT")
    btc = await hist.ensure("BTCUSDT", "D",
                            start_ms - WARMUP_DAILY * DAY_MS, end_ms)
    regime = btc_regime_map(btc)

    loaded = await load_many(hist, symbols, start_ms, end_ms, progress)
    if not loaded:
        return {"error": "Не удалось загрузить историю ни по одной монете",
                "stats": stats([], p), "trades": []}

    trades = simulate_all(loaded, p, start_ms, regime, progress)
    st = stats(trades, p)
    avg_sl = (sum(t.sl_pct for t in trades) / len(trades)) if trades else 0.0
    st["avg_sl_pct"] = avg_sl
    st["breakeven_winrate"] = breakeven_winrate(p, avg_sl or p.sl_fixed_pct)
    st["edge"] = st["winrate"] - st["breakeven_winrate"]
    curve = equity_curve(trades, p) if trades else []
    return {
        "stats": st,
        "curve": curve,
        "trades": [asdict(t) for t in sorted(
            trades, key=lambda x: x.entry_ts, reverse=True)],
        "symbols_loaded": len(loaded),
        "symbols_requested": len(symbols),
        "period": {"start": start_ms, "end": end_ms, "months": months},
        "params": asdict(p),
    }


async def run_sweep(hist: BybitHistory, symbols: List[str], months: int,
                    base: Params, grid: dict,
                    progress: Callable = None) -> dict:
    """Перебор сетки параметров по уже скачанным данным."""
    start_ms, end_ms = period_bounds(months)
    if progress:
        progress("Загрузка BTC", 0, 1, "BTCUSDT")
    btc = await hist.ensure("BTCUSDT", "D",
                            start_ms - WARMUP_DAILY * DAY_MS, end_ms)
    regime = btc_regime_map(btc)

    loaded = await load_many(hist, symbols, start_ms, end_ms, progress)
    if not loaded:
        return {"error": "Не удалось загрузить историю", "results": []}

    tps = grid.get("tp_pct") or [base.tp_pct]
    drops = grid.get("min_drop_3d") or [base.min_drop_3d]
    rsis = grid.get("rsi_max_4h") or [base.rsi_max_4h]
    atrs = grid.get("min_atr_distance") or [base.min_atr_distance]
    combos = [(a, b, c, d) for a in tps for b in drops
              for c in rsis for d in atrs]

    results = []
    total = len(combos)
    for n, (tp, drop, rsi_max, atr_d) in enumerate(combos, 1):
        if progress:
            progress("Перебор параметров", n, total,
                     f"TP {tp}% · падение {drop}% · RSI {rsi_max} · "
                     f"стоп {atr_d} ATR")
        p = base.copy_with(tp_pct=tp, min_drop_3d=drop, rsi_max_4h=rsi_max,
                           min_atr_distance=atr_d)
        trades = simulate_all(loaded, p, start_ms, regime)
        st = stats(trades, p)
        avg_sl = (sum(t.sl_pct for t in trades) / len(trades)
                  if trades else 0.0)
        st["avg_sl_pct"] = avg_sl
        st["breakeven_winrate"] = breakeven_winrate(
            p, avg_sl or p.sl_fixed_pct)
        st["edge"] = st["winrate"] - st["breakeven_winrate"]
        results.append({
            "tp_pct": tp, "min_drop_3d": drop, "rsi_max_4h": rsi_max,
            "min_atr_distance": atr_d,
            "trades": st["trades"], "winrate": st["winrate"],
            "breakeven_winrate": st["breakeven_winrate"],
            "edge": st["edge"], "expectancy": st["expectancy"],
            "profit_factor": st["profit_factor"],
            "equity_mult": st["equity_mult"], "max_dd": st["max_dd"],
            "avg_hours": st["avg_hours"],
        })
        await asyncio.sleep(0)

    # сортируем по ожиданию на сделку, но только там, где сделок достаточно
    results.sort(key=lambda r: (r["trades"] >= 20, r["expectancy"]),
                 reverse=True)
    return {"results": results, "symbols_loaded": len(loaded),
            "period": {"months": months}}
