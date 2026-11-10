"""
data/futu_gateway.py - 富途 OpenD L2 毫秒級即時串流與 10 檔擺盤包裝
===================================================================================
Provides high-speed TCP socket connectivity to FutuOpenD, millisecond-level Level 1/Level 2
quotes, 10-level bid/ask market depth parsing, and seamless fallback to Yahoo Finance.
"""

from __future__ import annotations

import socket
from datetime import datetime
from typing import Any

import numpy as np
try:
    import yfinance as yf
except ImportError:
    yf = None

from config.settings import (
    HK_DEFAULT_LOT_SIZES,
    get_candidate_tickers,
    get_hk_tick_size,
    normalize_hk_ticker,
    round_to_hk_tick,
)

try:
    import futu as ft
    FUTU_INSTALLED = True
except ImportError:
    ft = None
    FUTU_INSTALLED = False


def to_futu_ticker(ticker: str) -> str:
    """Convert standard HK ticker (e.g. 0700.HK, 5.HK, ^HSI) to Futu format (HK.00700, HK.800000)."""
    raw = str(ticker).strip().upper()
    if raw in ("^HSI", "HSI", "HK.800000", "800000"):
        return "HK.800000"
    if raw.startswith("HK."):
        return raw
    clean_code = raw.replace(".HK", "")
    digits = clean_code.lstrip("0")
    if not digits.isdigit():
        return "HK." + clean_code
    num = int(digits)
    return f"HK.{num:05d}"


def from_futu_ticker(futu_code: str) -> str:
    """Convert Futu standard ticker (e.g. HK.00700) back to Yahoo format (0700.HK)."""
    raw = str(futu_code).strip().upper()
    if raw in ("HK.800000", "HK.HSI"):
        return "^HSI"
    if raw.startswith("HK."):
        digits = raw.replace("HK.", "").lstrip("0")
        if digits.isdigit():
            num = int(digits)
            return f"{num:04d}.HK" if num < 10000 else f"{num:05d}.HK"
    return raw


def check_futu_opend_connection(host: str = "127.0.0.1", port: int = 11111, timeout: float = 0.5) -> bool:
    """Test TCP socket connection to FutuOpenD gateway."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def get_futu_realtime_quote(ticker: str, host: str = "127.0.0.1", port: int = 11111) -> dict[str, Any] | None:
    """
    Fetch millisecond-level real-time Level 2 quote from Futu OpenD if connected.
    Returns standardized quote dictionary or None if offline.
    """
    if not FUTU_INSTALLED or not check_futu_opend_connection(host, port):
        return None
    futu_code = to_futu_ticker(ticker)
    ctx = None
    try:
        ctx = ft.OpenQuoteContext(host=host, port=int(port))
        ctx.subscribe([futu_code], [ft.SubType.QUOTE], is_first_push=False)
        ret, df = ctx.get_stock_quote([futu_code])
        if ret == ft.RET_OK and not df.empty:
            row = df.iloc[0]
            last_p = float(row.get("last_price", 0.0))
            bid_p = float(row.get("bid_price", 0.0))
            ask_p = float(row.get("ask_price", 0.0))
            spread = max(0.0, ask_p - bid_p) if (ask_p > 0 and bid_p > 0) else get_hk_tick_size(last_p)
            spread_bps = (spread / (last_p + 1e-9)) * 10000.0
            return {
                "ticker": ticker,
                "futu_code": futu_code,
                "price": last_p,
                "bid": bid_p,
                "ask": ask_p,
                "spread": spread,
                "spread_bps": spread_bps,
                "spread_ticks": int(round(spread / get_hk_tick_size(last_p))),
                "volume": int(row.get("volume", 0)),
                "turnover": float(row.get("turnover", 0.0)),
                "lot_size": int(row.get("lot_size", HK_DEFAULT_LOT_SIZES.get(normalize_hk_ticker(ticker), 100))),
                "source": "🟢 富途牛牛 Level 2 實盤即時串流",
                "update_time": str(row.get("data_time", datetime.now().strftime("%H:%M:%S"))),
            }
    except Exception as e:
        print(f"Futu quote error: {e}")
    finally:
        if ctx:
            try:
                ctx.close()
            except Exception:
                pass
    return None


def get_futu_order_book(
    ticker: str,
    num: int = 10,
    host: str = "127.0.0.1",
    port: int = 11111,
    spot_price_fallback: float = 0.0,
    lot_size_fallback: int = 100,
) -> dict[str, Any]:
    """
    Fetch 10-level Market Depth (Bid 1~10 & Ask 1~10) with order counts and volumes.
    If FutuOpenD is connected, retrieves live L2 market depth.
    If FutuOpenD is offline, generates realistic HKEX statutory-tick aligned market depth.
    """
    futu_code = to_futu_ticker(ticker)

    if FUTU_INSTALLED and check_futu_opend_connection(host, port):
        ctx = None
        try:
            ctx = ft.OpenQuoteContext(host=host, port=int(port))
            ctx.subscribe([futu_code], [ft.SubType.ORDER_BOOK], is_first_push=False)
            ret, data = ctx.get_order_book(futu_code, num=num)
            if ret == ft.RET_OK and isinstance(data, dict):
                bids_raw = data.get("Bid", [])
                asks_raw = data.get("Ask", [])

                bids = []
                for idx, item in enumerate(bids_raw[:num]):
                    p = float(item[0]) if len(item) > 0 else 0.0
                    v = int(item[1]) if len(item) > 1 else 0
                    o = int(item[2]) if len(item) > 2 else 1
                    bids.append({"level": idx + 1, "price": p, "volume": v, "orders": o})

                asks = []
                for idx, item in enumerate(asks_raw[:num]):
                    p = float(item[0]) if len(item) > 0 else 0.0
                    v = int(item[1]) if len(item) > 1 else 0
                    o = int(item[2]) if len(item) > 2 else 1
                    asks.append({"level": idx + 1, "price": p, "volume": v, "orders": o})

                if bids and asks:
                    bid_1 = bids[0]["price"]
                    ask_1 = asks[0]["price"]
                    spread = max(0.001, ask_1 - bid_1)
                    mid_price = (bid_1 + ask_1) / 2.0
                    total_bid_v = sum(b["volume"] for b in bids)
                    total_ask_v = sum(a["volume"] for a in asks)
                    obi = (total_bid_v - total_ask_v) / max(1, total_bid_v + total_ask_v)
                    b1_v = bids[0]["volume"]
                    a1_v = asks[0]["volume"]
                    micro_p = (bid_1 * a1_v + ask_1 * b1_v) / max(1, b1_v + a1_v) if (b1_v + a1_v) > 0 else mid_price

                    return {
                        "ticker": ticker,
                        "futu_code": futu_code,
                        "is_live": True,
                        "source": "🟢 富途牛牛 Level 2 實盤即時擺盤",
                        "bids": bids,
                        "asks": asks,
                        "bid_1": bid_1,
                        "ask_1": ask_1,
                        "spread": spread,
                        "spread_bps": (spread / (mid_price + 1e-9)) * 10000.0,
                        "spread_ticks": int(round(spread / get_hk_tick_size(mid_price))),
                        "mid_price": mid_price,
                        "micro_price": micro_p,
                        "total_bid_vol": total_bid_v,
                        "total_ask_vol": total_ask_v,
                        "order_book_imbalance": obi,
                        "update_time": datetime.now().strftime("%H:%M:%S.%f")[:12],
                    }
            # Fallback Level 1 quote if L2 permission not enabled
            try:
                ret_q, df_q = ctx.get_stock_quote([futu_code])
                if ret_q == ft.RET_OK and not df_q.empty:
                    row_q = df_q.iloc[0]
                    last_p = float(row_q.get("last_price", 0.0))
                    bid_1 = float(row_q.get("bid_price", 0.0))
                    ask_1 = float(row_q.get("ask_price", 0.0))
                    b1_v = int(row_q.get("bid_vol", 0))
                    a1_v = int(row_q.get("ask_vol", 0))
                    lot_sz = int(row_q.get("lot_size", lot_size_fallback))

                    if bid_1 > 0 and ask_1 > 0:
                        spread = max(0.001, ask_1 - bid_1)
                        mid_price = (bid_1 + ask_1) / 2.0
                        tick = get_hk_tick_size(mid_price)

                        l1_bids = [{"level": 1, "price": bid_1, "volume": b1_v, "orders": max(1, b1_v // lot_sz)}]
                        l1_asks = [{"level": 1, "price": ask_1, "volume": a1_v, "orders": max(1, a1_v // lot_sz)}]

                        for i in range(2, num + 1):
                            bp = round_to_hk_tick(bid_1 - (i - 1) * tick)
                            ap = round_to_hk_tick(ask_1 + (i - 1) * tick)
                            sim_bv = int(b1_v * (0.85 ** (i - 1)))
                            sim_av = int(a1_v * (0.85 ** (i - 1)))
                            l1_bids.append({"level": i, "price": bp, "volume": sim_bv, "orders": max(1, sim_bv // lot_sz)})
                            l1_asks.append({"level": i, "price": ap, "volume": sim_av, "orders": max(1, sim_av // lot_sz)})

                        total_bid_v = sum(b["volume"] for b in l1_bids)
                        total_ask_v = sum(a["volume"] for a in l1_asks)
                        obi = (b1_v - a1_v) / max(1, b1_v + a1_v)
                        micro_p = (bid_1 * a1_v + ask_1 * b1_v) / max(1, b1_v + a1_v)

                        return {
                            "ticker": ticker,
                            "futu_code": futu_code,
                            "is_live": True,
                            "is_level2": False,
                            "quote_level": "Level 1 (買賣一檔實盤即時串流)",
                            "source": "🟢 富途牛牛 Level 1 實盤即時報價 (買一/賣一 0 延遲)",
                            "bids": l1_bids,
                            "asks": l1_asks,
                            "bid_1": bid_1,
                            "ask_1": ask_1,
                            "spread": spread,
                            "spread_bps": (spread / (mid_price + 1e-9)) * 10000.0,
                            "spread_ticks": int(round(spread / tick)),
                            "mid_price": mid_price,
                            "micro_price": micro_p,
                            "total_bid_vol": total_bid_v,
                            "total_ask_vol": total_ask_v,
                            "order_book_imbalance": obi,
                            "update_time": str(row_q.get("data_time", datetime.now().strftime("%H:%M:%S"))),
                        }
            except Exception as e_q:
                print(f"Futu Level 1 fallback error: {e_q}")
        except Exception as e:
            print(f"Futu order book error: {e}")
        finally:
            if ctx:
                try:
                    ctx.close()
                except Exception:
                    pass

    # Fallback / Scaffolding Mode: synthesize 10-level HKEX tick-aligned order book
    ref_p = float(spot_price_fallback) if spot_price_fallback > 0 else 100.0
    tick = get_hk_tick_size(ref_p)
    lot = max(100, int(lot_size_fallback))

    sim_bids = []
    sim_asks = []
    base_vol = lot * 25
    for i in range(1, num + 1):
        bp = round_to_hk_tick(ref_p - (i - 1) * tick)
        ap = round_to_hk_tick(ref_p + i * tick)
        b_vol = int(base_vol * (1.0 + 0.15 * i + 0.08 * (i % 3)))
        a_vol = int(base_vol * (1.0 + 0.12 * i + 0.05 * (i % 2)))
        sim_bids.append({"level": i, "price": bp, "volume": b_vol, "orders": max(1, int(b_vol / (lot * 3)))})
        sim_asks.append({"level": i, "price": ap, "volume": a_vol, "orders": max(1, int(a_vol / (lot * 3)))})

    bid_1 = sim_bids[0]["price"]
    ask_1 = sim_asks[0]["price"]
    spread = max(tick, ask_1 - bid_1)
    mid_price = (bid_1 + ask_1) / 2.0
    total_bid_v = sum(b["volume"] for b in sim_bids)
    total_ask_v = sum(a["volume"] for a in sim_asks)
    obi = (total_bid_v - total_ask_v) / max(1, total_bid_v + total_ask_v)

    return {
        "ticker": ticker,
        "futu_code": futu_code,
        "is_live": False,
        "source": "⚪ 備援模式 / 示範擺盤 (請啟動本地 FutuOpenD 以載入 Level 2 實盤串流)",
        "bids": sim_bids,
        "asks": sim_asks,
        "bid_1": bid_1,
        "ask_1": ask_1,
        "spread": spread,
        "spread_bps": (spread / (mid_price + 1e-9)) * 10000.0,
        "spread_ticks": 1,
        "mid_price": mid_price,
        "micro_price": (bid_1 + ask_1) / 2.0,
        "total_bid_vol": total_bid_v,
        "total_ask_vol": total_ask_v,
        "order_book_imbalance": obi,
        "update_time": datetime.now().strftime("%H:%M:%S"),
    }


def fetch_realtime_spot_price(ticker: str, futu_host: str = "127.0.0.1", futu_port: int = 11111) -> float | None:
    """
    Fetch the latest real-time / near-real-time spot quote.
    Priority 1: Futu OpenD Level 2 real-time streaming feed (if connected).
    Priority 2: Yahoo Finance fast_info / regularMarketPrice fallback.
    """
    if FUTU_INSTALLED and check_futu_opend_connection(futu_host, futu_port):
        f_quote = get_futu_realtime_quote(ticker, host=futu_host, port=futu_port)
        if f_quote and f_quote.get("last_price", 0) > 0:
            return float(f_quote["last_price"])
    for cand in get_candidate_tickers(ticker):
        try:
            tk = yf.Ticker(cand)
            fast = getattr(tk, "fast_info", None)
            if fast is not None:
                p = getattr(fast, "last_price", None) or getattr(fast, "regular_market_price", None)
                if p is not None and isinstance(p, (int, float, np.floating, np.integer)) and not np.isnan(float(p)) and float(p) > 0.05:
                    return float(p)
            info = getattr(tk, "info", None)
            if info and isinstance(info, dict):
                p = info.get("regularMarketPrice") or info.get("currentPrice")
                if p is not None and isinstance(p, (int, float, np.floating, np.integer)) and not np.isnan(float(p)) and float(p) > 0.05:
                    return float(p)
        except Exception:
            continue
    return None