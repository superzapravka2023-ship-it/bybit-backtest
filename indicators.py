"""Индикаторы и работа со свечами — чистый Python, без numpy/pandas."""
from typing import List, Optional, Tuple


def sma(values: List[float], length: int) -> Optional[float]:
    if len(values) < length or length <= 0:
        return None
    return sum(values[-length:]) / length


def ema(values: List[float], length: int) -> Optional[float]:
    if not values or len(values) < length or length <= 0:
        return None
    k = 2.0 / (length + 1.0)
    val = sum(values[:length]) / length
    for v in values[length:]:
        val = v * k + val * (1.0 - k)
    return val


def ema_series(values: List[float], length: int) -> List[Optional[float]]:
    """Полный ряд EMA — нужен для графика."""
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < length or length <= 0:
        return out
    k = 2.0 / (length + 1.0)
    val = sum(values[:length]) / length
    out[length - 1] = val
    for i in range(length, len(values)):
        val = values[i] * k + val * (1.0 - k)
        out[i] = val
    return out


def rsi(closes: List[float], length: int = 14) -> Optional[float]:
    s = rsi_series(closes, length)
    return s[-1] if s and s[-1] is not None else None


def rsi_series(closes: List[float], length: int = 14) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(closes)
    if len(closes) < length + 1:
        return out
    gains = losses = 0.0
    for i in range(1, length + 1):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    avg_gain = gains / length
    avg_loss = losses / length

    def to_rsi(ag: float, al: float) -> float:
        if al == 0:
            return 100.0
        return 100.0 - (100.0 / (1.0 + ag / al))

    out[length] = to_rsi(avg_gain, avg_loss)
    for i in range(length + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        gain = d if d > 0 else 0.0
        loss = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        out[i] = to_rsi(avg_gain, avg_loss)
    return out


def atr(highs: List[float], lows: List[float], closes: List[float],
        length: int = 14) -> Optional[float]:
    n = min(len(highs), len(lows), len(closes))
    if n < length + 1:
        return None
    trs = []
    for i in range(1, n):
        trs.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    val = sum(trs[:length]) / length
    for tr in trs[length:]:
        val = (val * (length - 1) + tr) / length
    return val


def pct(a: float, b: float) -> float:
    """Изменение в процентах от a к b."""
    if a == 0:
        return 0.0
    return (b - a) / a * 100.0


def swing_low(lows: List[float], lookback: int = 20) -> Tuple[float, int]:
    """Минимум за lookback свечей и сколько свечей назад он был."""
    if not lows:
        return 0.0, -1
    window = lows[-lookback:] if len(lows) >= lookback else lows[:]
    low = min(window)
    idx_from_end = len(window) - 1 - window.index(low)
    return low, idx_from_end


def swing_high(highs: List[float], lookback: int = 20) -> Tuple[float, int]:
    if not highs:
        return 0.0, -1
    window = highs[-lookback:] if len(highs) >= lookback else highs[:]
    high = max(window)
    idx_from_end = len(window) - 1 - window.index(high)
    return high, idx_from_end


def body_ratio(o: float, h: float, l: float, c: float) -> float:
    """Доля тела свечи в полном диапазоне."""
    rng = h - l
    if rng <= 0:
        return 0.0
    return abs(c - o) / rng


def lower_wick_ratio(o: float, h: float, l: float, c: float) -> float:
    """Доля нижней тени — признак выкупа снизу."""
    rng = h - l
    if rng <= 0:
        return 0.0
    return (min(o, c) - l) / rng


def bullish_divergence(closes: List[float], rsis: List[Optional[float]],
                       lookback: int = 30) -> bool:
    """Цена сделала более низкий минимум, а RSI — более высокий."""
    n = len(closes)
    if n < lookback or len(rsis) != n:
        return False
    seg_c = closes[-lookback:]
    seg_r = rsis[-lookback:]
    half = lookback // 2
    c1 = min(seg_c[:half])
    i1 = seg_c[:half].index(c1)
    c2 = min(seg_c[half:])
    i2 = half + seg_c[half:].index(c2)
    r1, r2 = seg_r[i1], seg_r[i2]
    if r1 is None or r2 is None:
        return False
    return c2 < c1 and r2 > r1 + 2.0


def fib_levels(low: float, high: float) -> dict:
    """Уровни восстановления падения от high к low."""
    rng = high - low
    if rng <= 0:
        return {}
    return {
        "23.6%": low + rng * 0.236,
        "38.2%": low + rng * 0.382,
        "50.0%": low + rng * 0.5,
        "61.8%": low + rng * 0.618,
    }
