"""Веб-приложение бэктестера: форма, фоновые задания, выдача результатов."""
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict

from aiohttp import web

from data import BybitError, BybitHistory, DataStore
from runner import run_backtest, run_sweep
from strategy import Params

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)-9s %(message)s",
    stream=sys.stdout)
log = logging.getLogger("app")

PORT = int(os.getenv("PORT", "10000"))
DATA_DIR = os.getenv("DATA_DIR", ".")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_BLACKLIST = [s.strip().upper() for s in os.getenv(
    "BLACKLIST",
    "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,USDCUSDT,"
    "USDEUSDT,TONUSDT,ADAUSDT,TRXUSDT").split(",") if s.strip()]

store = DataStore(DATA_DIR)
history = BybitHistory(store)
JOBS: Dict[str, Dict[str, Any]] = {}


def new_job() -> str:
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {"status": "running", "stage": "Старт", "done": 0,
                 "total": 1, "message": "", "result": None, "error": None,
                 "started": time.time()}
    # чистим старые задания
    for k in list(JOBS):
        if time.time() - JOBS[k]["started"] > 3600 and k != jid:
            JOBS.pop(k, None)
    return jid


def progress_for(jid: str):
    def cb(stage: str, done: int, total: int, message: str = ""):
        j = JOBS.get(jid)
        if j:
            j.update(stage=stage, done=done, total=max(total, 1),
                     message=message)
    return cb


def params_from(body: dict) -> Params:
    p = Params()
    for key in p.__dict__:
        if key in body and body[key] not in (None, ""):
            cur = getattr(p, key)
            try:
                setattr(p, key, type(cur)(body[key]))
            except (TypeError, ValueError):
                pass
    return p


async def resolve_symbols(body: dict) -> list:
    """Какие монеты тестировать: конкретная, топ-N или список."""
    raw = (body.get("symbol") or "").strip().upper()
    mode = (body.get("mode") or "single").strip()

    if mode == "top":
        limit = int(body.get("top_n") or 40)
        return await history.top_symbols(
            limit=limit,
            min_turnover=float(body.get("min_turnover") or 20_000_000),
            min_age_days=int(body.get("min_age_days") or 120),
            blacklist=DEFAULT_BLACKLIST)

    if not raw:
        raise ValueError("не указан тикер")
    out = []
    for part in raw.replace(";", ",").split(","):
        s = part.strip().upper()
        if not s:
            continue
        if not s.endswith("USDT"):
            s += "USDT"
        out.append(s)
    return out


# ------------------------------------------------------------------ задания
async def job_backtest(jid: str, body: dict):
    try:
        symbols = await resolve_symbols(body)
        p = params_from(body)
        months = int(body.get("months") or 6)
        res = await run_backtest(history, symbols, months, p,
                                 progress_for(jid))
        JOBS[jid].update(status="done", result=res)
    except BybitError as e:
        JOBS[jid].update(status="error", error=str(e))
    except ValueError as e:
        JOBS[jid].update(status="error", error=str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("Бэктест упал: %s", e)
        JOBS[jid].update(status="error", error=f"{type(e).__name__}: {e}")


async def job_sweep(jid: str, body: dict):
    try:
        symbols = await resolve_symbols(body)
        p = params_from(body)
        months = int(body.get("months") or 6)

        def nums(key, default):
            raw = body.get(key)
            if not raw:
                return default
            if isinstance(raw, list):
                vals = raw
            else:
                vals = str(raw).replace(";", ",").split(",")
            out = []
            for v in vals:
                try:
                    out.append(float(str(v).strip()))
                except (TypeError, ValueError):
                    continue
            return out or default

        grid = {
            "tp_pct": nums("grid_tp", [3.0, 5.0, 7.0]),
            "min_drop_3d": nums("grid_drop", [10.0, 14.0, 18.0]),
            "rsi_max_4h": nums("grid_rsi", [38.0, 42.0, 47.0]),
            "min_atr_distance": nums("grid_atr", [p.min_atr_distance]),
        }
        res = await run_sweep(history, symbols, months, p, grid,
                              progress_for(jid))
        JOBS[jid].update(status="done", result=res)
    except BybitError as e:
        JOBS[jid].update(status="error", error=str(e))
    except ValueError as e:
        JOBS[jid].update(status="error", error=str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("Перебор упал: %s", e)
        JOBS[jid].update(status="error", error=f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ роуты
async def index(_req):
    path = os.path.join(BASE_DIR, "ui.html")
    try:
        with open(path, encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/html")
    except OSError:
        return web.Response(text="ui.html не найден", status=500)


async def api_backtest(req):
    body = await req.json()
    jid = new_job()
    asyncio.create_task(job_backtest(jid, body))
    return web.json_response({"job": jid})


async def api_sweep(req):
    body = await req.json()
    jid = new_job()
    asyncio.create_task(job_sweep(jid, body))
    return web.json_response({"job": jid})


async def api_job(req):
    jid = req.match_info["jid"]
    job = JOBS.get(jid)
    if not job:
        return web.json_response({"error": "задание не найдено"}, status=404)
    payload = {k: v for k, v in job.items() if k != "result"}
    if job["status"] == "done":
        payload["result"] = job["result"]
    return web.json_response(payload)


async def api_defaults(_req):
    return web.json_response({
        "params": asdict(Params()),
        "blacklist": DEFAULT_BLACKLIST,
        "cache": {"symbols": len(store.cached_symbols()),
                  "size_mb": round(store.size_mb(), 1)},
    })


async def api_cache_clear(req):
    body = await req.json() if req.can_read_body else {}
    store.clear(body.get("symbol"))
    return web.json_response({"ok": True,
                              "symbols": len(store.cached_symbols())})


async def health(_req):
    running = sum(1 for j in JOBS.values() if j["status"] == "running")
    return web.json_response({
        "status": "ok",
        "jobs_running": running,
        "cached_symbols": len(store.cached_symbols()),
        "cache_mb": round(store.size_mb(), 1),
    })


def make_app() -> web.Application:
    app = web.Application(client_max_size=4 * 1024 * 1024)
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/api/defaults", api_defaults)
    app.router.add_post("/api/backtest", api_backtest)
    app.router.add_post("/api/sweep", api_sweep)
    app.router.add_get("/api/job/{jid}", api_job)
    app.router.add_post("/api/cache/clear", api_cache_clear)
    return app


async def on_cleanup(_app):
    await history.close()


if __name__ == "__main__":
    application = make_app()
    application.on_cleanup.append(on_cleanup)
    log.info("Бэктестер слушает порт %d", PORT)
    web.run_app(application, host="0.0.0.0", port=PORT, print=None)
