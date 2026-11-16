"""Правила входа — те же, что в боевом боте, но с параметрами снаружи.

Главное требование: на каждом шаге используются только данные, которые
были бы доступны в этот момент в реальности. Дневная свеча текущего дня
собирается из уже прошедших 4-часовых свечей, а не берётся готовой из
истории — иначе бэктест подглядывал бы в будущее.
"""
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from indicators import (atr, bullish_divergence, fib_levels,
                        lower_wick_ratio, pct, rsi, rsi_series, sma,
                        swing_low)

DAY_MS = 86_400_000


@dataclass
class Params:
    # цель и риск
    tp_pct: float = 5.0
    sl_mode: str = "atr"          # atr | fixed
    sl_fixed_pct: float = 5.0
    sl_atr_mult: float = 0.8
    min_atr_distance: float = 1.5   # стоп не ближе стольких ATR от входа
    max_sl_pct: float = 7.0
    min_sl_pct: float = 2.0
    min_rr: float = 1.1

    # форма входа
    min_drop_3d: float = 14.0
    min_drop_1d: float = 8.0
    max_drop_3d: float = 45.0
    shock_ratio: float = 0.42

    # отсев умирающих
    max_drop_30d: float = 55.0
    max_drop_90d: float = 70.0

    # живость монеты
    min_atr_pct_4h: float = 1.2
    max_atr_pct_4h: float = 9.0
    min_range_14d: float = 12.0
    low_freshness_days: int = 3

    # разворот
    rsi_max_4h: float = 42.0
    rsi_max_1d: float = 45.0
    min_volume_spike: float = 1.6
    min_notes: int = 2

    # сопровождение
    track_hours: int = 72
    cooldown_hours: int = 18
    max_concurrent: int = 3

    # деньги
    fee_pct: float = 0.055         # комиссия одной стороны
    risk_pct: float = 2.0          # риск на сделку, % от эквити
    slippage_pct: float = 0.03

    def copy_with(self, **kw) -> "Params":
        d = {**self.__dict__, **kw}
        return Params(**d)


@dataclass
class RawSignal:
    symbol: str
    ts: int
    entry: float
    sl: float
    tp: float
    sl_pct: float
    tp_pct: float
    rr: float
    drop_1d: float
    drop_3d: float
    drop_30d: float
    rsi_4h: float
    rsi_1d: float
    atr_pct: float
    vol_spike: float
    notes: int
    bar_index: int


def day_start(ts: int) -> int:
    return ts - (ts % DAY_MS)


def build_daily_view(daily: List[dict], h4: List[dict], i: int,
                     daily_ts: List[int] = None) -> List[dict]:
    """Дневные свечи, какими они были бы видны в момент закрытия 4ч свечи i.

    Завершённые дни берём из истории, текущий день собираем из тех 4ч
    свечей, которые уже закрылись. Так мы не видим будущего внутри дня.
    Берём хвост в 200 дней — ровно столько отдаёт биржа боевому боту,
    поэтому индикаторы считаются на той же глубине, что и в реальности.
    """
    now_ts = h4[i]["ts"]
    today = day_start(now_ts)
    if daily_ts is not None:
        k = bisect_left(daily_ts, today)
        out = list(daily[max(0, k - 199):k])
    else:
        out = [d for d in daily if d["ts"] < today][-199:]

    # собираем частичную свечу текущего дня
    o = h = l = c = None
    vol = 0.0
    j = i
    while j >= 0 and day_start(h4[j]["ts"]) == today:
        bar = h4[j]
        h = bar["high"] if h is None else max(h, bar["high"])
        l = bar["low"] if l is None else min(l, bar["low"])
        o = bar["open"]
        if c is None:
            c = bar["close"]
        vol += bar["volume"]
        j -= 1
    if c is not None:
        out.append({"ts": today, "open": o, "high": h, "low": l,
                    "close": c, "volume": vol})
    return out


def evaluate_bar(symbol: str, daily: List[dict], h4: List[dict], i: int,
                 p: Params, daily_ts: List[int] = None) -> Optional[RawSignal]:
    """Проверяет условия входа на закрытии 4ч свечи с индексом i."""
    if i < 60:
        return None
    dv = build_daily_view(daily, h4, i, daily_ts)
    if len(dv) < 95:
        return None

    dc = [d["close"] for d in dv]
    dh = [d["high"] for d in dv]
    dl = [d["low"] for d in dv]
    dvol = [d["volume"] for d in dv]
    price = h4[i]["close"]
    if price <= 0:
        return None

    def close_ago(n: int) -> float:
        idx = len(dc) - 1 - n
        return dc[idx] if idx >= 0 else dc[0]

    drop_1d = pct(close_ago(1), price)
    drop_3d = pct(close_ago(3), price)
    drop_30d = pct(close_ago(30), price)
    drop_90d = pct(close_ago(90), price) if len(dc) > 91 else drop_30d

    if not (drop_3d <= -p.min_drop_3d or drop_1d <= -p.min_drop_1d):
        return None
    if drop_3d < -p.max_drop_3d:
        return None

    if drop_30d >= -1.0:
        shock = 1.0
    else:
        shock = max(0.0, min(1.0, drop_3d / drop_30d))
    if shock < p.shock_ratio:
        return None

    if drop_30d < -p.max_drop_30d or drop_90d < -p.max_drop_90d:
        return None

    look14 = min(14, len(dl))
    lo14, hi14 = min(dl[-look14:]), max(dh[-look14:])
    if pct(lo14, hi14) < p.min_range_14d:
        return None

    look20 = min(20, len(dl))
    _, low_age = swing_low(dl[-look20:], look20)
    if low_age > p.low_freshness_days:
        return None

    rsi_1d = rsi(dc, 14) or 50.0
    if rsi_1d > p.rsi_max_1d:
        return None

    vol_avg = sum(dvol[-21:-1]) / max(1, len(dvol[-21:-1]))
    recent_vol = max(dvol[-4:]) if len(dvol) >= 4 else dvol[-1]
    vol_spike = (recent_vol / vol_avg) if vol_avg > 0 else 0.0

    # ---------------------------- 4-часовой таймфрейм --------------------
    window = h4[max(0, i - 199):i + 1]
    hh = [b["high"] for b in window]
    hl = [b["low"] for b in window]
    hc = [b["close"] for b in window]

    a = atr(hh, hl, hc, 14)
    if not a:
        return None
    atr_pct = a / price * 100.0
    if atr_pct < p.min_atr_pct_4h or atr_pct > p.max_atr_pct_4h:
        return None

    rs = rsi_series(hc, 14)
    rsi_4h = rs[-1] if rs and rs[-1] is not None else 50.0
    if rsi_4h > p.rsi_max_4h:
        return None

    notes = 0
    last = h4[i]
    prev = h4[i - 1]
    if last["close"] > last["open"]:
        notes += 1
    if lower_wick_ratio(last["open"], last["high"], last["low"],
                        last["close"]) >= 0.35:
        notes += 1
    if last["low"] > prev["low"]:
        notes += 1
    rs_prev = rs[-3] if len(rs) >= 3 and rs[-3] is not None else rsi_4h
    if rsi_4h > rs_prev + 1.0:
        notes += 1
    if bullish_divergence(hc, rs, 30):
        notes += 1
    if vol_spike >= p.min_volume_spike:
        notes += 1
    if notes < p.min_notes:
        return None

    # ------------------------------- уровни -------------------------------
    entry = price
    look5 = min(5, len(dl))
    recent_low = min(dl[-look5:])
    drop_high = max(dh[-(look5 + 1):])

    if p.sl_mode == "fixed":
        sl = entry * (1 - p.sl_fixed_pct / 100.0)
        sl_pct = p.sl_fixed_pct
    else:
        # под лоем, но не ближе min_atr_distance ATR от входа:
        # иначе при входе у самого дна стоп попадает в рыночный шум
        # и его сносит первой же свечой
        sl = min(recent_low - a * p.sl_atr_mult,
                 entry - a * p.min_atr_distance)
        sl_pct = abs(pct(entry, sl))
    if sl_pct > p.max_sl_pct:
        sl = entry * (1 - p.max_sl_pct / 100.0)
        sl_pct = p.max_sl_pct
    if sl_pct < p.min_sl_pct:
        sl = entry * (1 - p.min_sl_pct / 100.0)
        sl_pct = p.min_sl_pct

    tp = entry * (1 + p.tp_pct / 100.0)
    fibs = fib_levels(recent_low, drop_high)
    room = pct(entry, fibs.get("38.2%", entry))
    if room < p.tp_pct * 0.6:
        return None

    rr = (tp - entry) / (entry - sl) if entry > sl else 0.0
    if rr < p.min_rr:
        return None

    return RawSignal(
        symbol=symbol, ts=h4[i]["ts"], entry=entry, sl=sl, tp=tp,
        sl_pct=sl_pct, tp_pct=p.tp_pct, rr=rr,
        drop_1d=drop_1d, drop_3d=drop_3d, drop_30d=drop_30d,
        rsi_4h=rsi_4h, rsi_1d=rsi_1d, atr_pct=atr_pct,
        vol_spike=vol_spike, notes=notes, bar_index=i)
