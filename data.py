"""Загрузка исторических свечей Bybit с кэшем в SQLite.

Свечи качаются постранично и складываются в базу. Повторный прогон
бэктеста по той же монете и периоду уже не обращается к бирже —
это делает подбор параметров мгновенным.
"""
import asyncio
import json
import logging
import os
import sqlite3
import time
from typing import Dict, List, Optional

import aiohttp

log = logging.getLogger("data")

DAY_MS = 86_400_000
INTERVAL_MS = {"1": 60_000, "5": 300_000, "15": 900_000, "60": 3_600_000,
               "240": 14_400_000, "D": DAY_MS}


class BybitError(Exception):
    pass


class DataStore:
    def __init__(self, data_dir: str = "."):
        try:
            os.makedirs(data_dir, exist_ok=True)
        except OSError:
            data_dir = "."
        self.path = os.path.join(data_dir, "history.db")
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            ts INTEGER NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, interval, ts)
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE INDEX IF NOT EXISTS idx_candles ON candles(symbol, interval, ts);
        """)
        self.conn.commit()
        log.info("Кэш истории: %s", self.path)

    def save(self, symbol: str, interval: str, candles: List[dict]):
        if not candles:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO candles(symbol,interval,ts,open,high,low,"
            "close,volume) VALUES(?,?,?,?,?,?,?,?)",
            [(symbol, interval, c["ts"], c["open"], c["high"], c["low"],
              c["close"], c["volume"]) for c in candles])
        self.conn.commit()

    def load(self, symbol: str, interval: str, start_ms: int,
             end_ms: int) -> List[dict]:
        rows = self.conn.execute(
            "SELECT ts,open,high,low,close,volume FROM candles "
            "WHERE symbol=? AND interval=? AND ts>=? AND ts<=? ORDER BY ts",
            (symbol, interval, start_ms, end_ms)).fetchall()
        return [dict(r) for r in rows]

    def coverage(self, symbol: str, interval: str) -> tuple:
        row = self.conn.execute(
            "SELECT MIN(ts) a, MAX(ts) b, COUNT(*) n FROM candles "
            "WHERE symbol=? AND interval=?", (symbol, interval)).fetchone()
        if not row or not row["n"]:
            return (0, 0, 0)
        return (row["a"], row["b"], row["n"])

    def cached_symbols(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT symbol FROM candles ORDER BY symbol").fetchall()
        return [r["symbol"] for r in rows]

    def clear(self, symbol: Optional[str] = None):
        if symbol:
            self.conn.execute("DELETE FROM candles WHERE symbol=?", (symbol,))
        else:
            self.conn.execute("DELETE FROM candles")
        self.conn.commit()

    def size_mb(self) -> float:
        try:
            return os.path.getsize(self.path) / 1e6
        except OSError:
            return 0.0


class BybitHistory:
    def __init__(self, store: DataStore, base_url: str = None,
                 delay_ms: int = 120):
        self.store = store
        self.base = (base_url or os.getenv("BYBIT_URL",
                                           "https://api.bybit.com")).rstrip("/")
        self.delay = delay_ms / 1000.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._instruments: Dict[str, dict] = {}
        self._tickers: Dict[str, dict] = {}

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30))
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, path: str, params: dict, retries: int = 3):
        url = self.base + path
        last = None
        for attempt in range(retries):
            try:
                sess = await self.session()
                async with sess.get(url, params=params) as r:
                    status = r.status
                    raw = await r.text()
                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    low = raw.lower()
                    hint = ""
                    if status == 403 or "denied" in low or "cloudfront" in low:
                        hint = (" | блокировка по региону: Bybit закрывает API "
                                "для IP из США и Китая, нужен регион Frankfurt")
                    raise BybitError(f"HTTP {status}, не JSON"
                                     f"{hint}: {' '.join(raw.split())[:150]}")
                code = int(data.get("retCode", -1))
                if code == 0:
                    return data.get("result", {})
                if code == 10006 and attempt < retries - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise BybitError(f"[{code}] {data.get('retMsg')}")
            except BybitError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last = e
                if attempt < retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
        raise BybitError(f"сеть: {last}")

    # ----------------------------------------------------------- справочники
    async def instruments(self) -> Dict[str, dict]:
        if self._instruments:
            return self._instruments
        out, cursor = {}, None
        for _ in range(20):
            params = {"category": "linear", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            res = await self._get("/v5/market/instruments-info", params)
            for it in res.get("list", []):
                if (it.get("quoteCoin") != "USDT"
                        or it.get("status") != "Trading"
                        or it.get("contractType") != "LinearPerpetual"):
                    continue
                out[it["symbol"]] = {
                    "symbol": it["symbol"],
                    "launch_ms": int(it.get("launchTime", 0) or 0)}
            cursor = res.get("nextPageCursor")
            if not cursor:
                break
        self._instruments = out
        return out

    async def tickers(self) -> Dict[str, dict]:
        if self._tickers:
            return self._tickers
        res = await self._get("/v5/market/tickers", {"category": "linear"})
        out = {}
        for t in res.get("list", []):
            try:
                out[t["symbol"]] = {
                    "last": float(t.get("lastPrice") or 0),
                    "turnover24h": float(t.get("turnover24h") or 0)}
            except (TypeError, ValueError):
                continue
        self._tickers = out
        return out

    async def top_symbols(self, limit: int, min_turnover: float,
                          min_age_days: int, blacklist: List[str]) -> List[str]:
        """Самые ликвидные монеты, подходящие по возрасту."""
        instruments = await self.instruments()
        tickers = await self.tickers()
        now = int(time.time() * 1000)
        rows = []
        for sym, inst in instruments.items():
            if sym in blacklist:
                continue
            t = tickers.get(sym)
            if not t or t["turnover24h"] < min_turnover:
                continue
            launch = inst.get("launch_ms", 0)
            if launch and (now - launch) < min_age_days * DAY_MS:
                continue
            rows.append((sym, t["turnover24h"]))
        rows.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in rows[:limit]]

    # ------------------------------------------------------- скачивание свечей
    async def fetch_range(self, symbol: str, interval: str,
                          start_ms: int, end_ms: int) -> List[dict]:
        """Качает свечи за период постранично, от свежих к старым."""
        step = INTERVAL_MS.get(interval, DAY_MS)
        out: Dict[int, dict] = {}
        cursor_end = end_ms
        for _ in range(60):
            res = await self._get("/v5/market/kline", {
                "category": "linear", "symbol": symbol,
                "interval": interval, "limit": 1000,
                "start": start_ms, "end": cursor_end})
            rows = res.get("list", []) or []
            if not rows:
                break
            got = 0
            for row in rows:
                try:
                    ts = int(row[0])
                    if ts in out:
                        continue
                    out[ts] = {"ts": ts, "open": float(row[1]),
                               "high": float(row[2]), "low": float(row[3]),
                               "close": float(row[4]), "volume": float(row[5])}
                    got += 1
                except (ValueError, IndexError):
                    continue
            oldest = min(int(r[0]) for r in rows)
            if oldest <= start_ms or got == 0:
                break
            cursor_end = oldest - step
            await asyncio.sleep(self.delay)
        return [out[k] for k in sorted(out)]

    async def ensure(self, symbol: str, interval: str, start_ms: int,
                     end_ms: int) -> List[dict]:
        """Отдаёт свечи из кэша, докачивая недостающее."""
        step = INTERVAL_MS.get(interval, DAY_MS)
        have = self.store.load(symbol, interval, start_ms, end_ms)
        expected = max(1, (end_ms - start_ms) // step)
        # 85% покрытия считаем достаточным: биржа могла не торговать часть дней
        if have and len(have) >= expected * 0.85:
            return have
        fresh = await self.fetch_range(symbol, interval, start_ms, end_ms)
        if fresh:
            self.store.save(symbol, interval, fresh)
        return self.store.load(symbol, interval, start_ms, end_ms)
