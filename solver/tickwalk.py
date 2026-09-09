#!/usr/bin/env python3
"""
tickwalk.py -- full multi-segment V3 swap walk (the lead investigator).

The single-segment replay in scan.py assumes constant liquidity through
the swap. Real V3 swaps WALK: when the price reaches the next initialized
tick, active liquidity changes by that tick's liquidityNet and the walk
continues. A swap that starts above a position, passes entirely through
it, and ends below it returns to the SAME active L -- invisible to the
single-segment model, obvious to the walk.

This implements the exact Pool.sol swap loop + SwapMath.computeSwapStep
for exactInput and exactOutput, then replays the recorded LEAD from
solver/scanlog/eth.jsonl against the pool's live tick data.

Caveat (printed, honest): tick state is read at LATEST, the swap happened
hours ago; if LPs changed in between, the walk sees a different world.
"""
import math
import sys
import time
from bisect import bisect_right

sys.path.insert(0, ".")
from solver.mainnet import rpc                       # noqa: E402
from solver.v3math import (                          # noqa: E402
    Q96, get_amount0_delta, get_amount1_delta,
    get_next_sqrt_price_from_amount0_rounding_up,
    get_sqrt_ratio_at_tick, TickMathConstants)

FEE_ONE = 1_000_000
MIN_SQRT = TickMathConstants.MIN_SQRT_RATIO

# the LEAD under investigation (from solver/scanlog/eth.jsonl)
LEAD = dict(
    tx="0x8752ff39750d744d4791427de2846e9f2c34d3fe4dae2172efca72e99ba1dac9",
    pool="0x677c5f13227c6750852afdaa89122b6327da47ef",
    sqrtP=252973450403891274200115746817714256,
    L=17499655183695341504765,
    gross=356265645278920,
    out_chain=3410143565605548314056500602,
    target=237534322845280923174603440573333207,
    fee=100,
)


def ceil_div(a, b):
    return -((-a) // b)


def muldiv_ru(a, b, d):
    return (a * b + d - 1) // d


def tick_of(sqrtP):
    """Largest tick t with getSqrtRatioAtTick(t) <= sqrtP (exact)."""
    est = int(2 * math.log(sqrtP / Q96) / math.log(1.0001))
    t = est
    while get_sqrt_ratio_at_tick(t) > sqrtP:
        t -= 1
    while get_sqrt_ratio_at_tick(t + 1) <= sqrtP:
        t += 1
    return t


def fetch_ticks(pool, t_lo, t_hi):
    """Initialized ticks (with liquidityNet) in [t_lo, t_hi] from the
    tick bitmaps. State as of LATEST -- see module caveat."""
    words = {}
    for w in range(t_lo >> 8, (t_hi >> 8) + 1):
        r = rpc("eth_call", [{"to": pool,
                              "data": "0x5339c296"
                                      + (w & 0xFFFF).to_bytes(32, "big").hex()},
                             "latest"])
        words[w] = int(r, 16)
        time.sleep(0.12)
    ticks = {}
    for w, bm in words.items():
        for i in range(256):
            if (bm >> i) & 1:
                t = w * 256 + i
                if t_lo <= t <= t_hi:
                    ticks[t] = None
    for t in ticks:
        r = rpc("eth_call", [{"to": pool,
                              "data": "0xf30dba93"
                                      + t.to_bytes(32, "big", signed=True).hex()},
                             "latest"])
        raw = r[2:]
        gross_liq = int(raw[0:64], 16)
        net = int.from_bytes(bytes.fromhex(raw[64:128]), "big", signed=True)
        ticks[t] = (gross_liq, net)
        time.sleep(0.12)
    return ticks


def capture_tick_context(pool, sqrtP_from, sqrtP_to, call):
    """Snapshot the pool's tick structure around a price interval, NOW.

    This is the archive-wall breaker: when the scanner classifies a LEAD
    at block time, the tick state is still live and queryable -- capture
    it immediately, because the crossed positions may be burned later
    (exactly what happened to the first LEAD we ever found).
    `call` is an (method, params) -> result RPC function.
    """
    def rpc_call(method, params):
        return call(method, params)

    try:
        t_from = tick_of(max(sqrtP_from, sqrtP_to))
        t_to = tick_of(min(sqrtP_from, sqrtP_to))
        t_lo, t_hi = t_to - 64, t_from + 64
        ticks = {}
        for w in range(t_lo >> 8, (t_hi >> 8) + 1):
            r = rpc_call("eth_call", [{"to": pool,
                                       "data": "0x5339c296"
                                               + (w & 0xFFFF).to_bytes(
                                                   32, "big").hex()},
                                      "latest"])
            bm = int(r, 16)
            for i in range(256):
                if (bm >> i) & 1:
                    t = w * 256 + i
                    if t_lo <= t <= t_hi:
                        ticks[t] = None
        for t in ticks:
            r = rpc_call("eth_call", [{"to": pool,
                                       "data": "0xf30dba93"
                                               + t.to_bytes(32, "big",
                                                            signed=True).hex()},
                                      "latest"])
            raw = r[2:]
            ticks[t] = [int(raw[0:64], 16),
                        int.from_bytes(bytes.fromhex(raw[64:128]),
                                       "big", signed=True)]
        slot0 = rpc_call("eth_call", [{"to": pool, "data": "0x3850c7bd"},
                                       "latest"])[2:]
        liq = int(rpc_call("eth_call", [{"to": pool,
                                         "data": "0x1a686502"},
                                        "latest"]), 16)
        return dict(tick_range=[t_lo, t_hi],
                    ticks={str(t): ticks[t] for t in sorted(ticks)},
                    slot0_sqrtP=int(slot0[0:64], 16),
                    liquidity=liq)
    except Exception as e:                              # noqa: BLE001
        return dict(error=f"capture failed: {type(e).__name__}: {e}")


def walk_exact_in(sqrtP, L, ticks, gross, fee):
    """The Pool.sol swap loop, exactInput, zeroForOne (token0 in)."""
    remaining = gross
    out_total = 0
    steps = []
    ts = sorted(ticks)                      # ascending; we walk DOWN
    while remaining > 0 and sqrtP > MIN_SQRT:
        # next initialized tick at-or-below the current price
        ratios = [get_sqrt_ratio_at_tick(t) for t in ts]
        # largest t with ratio(t) <= sqrtP
        idx = bisect_right(ratios, sqrtP) - 1
        if idx >= 0:
            tick_next, sqrt_target = ts[idx], ratios[idx]
        else:
            tick_next, sqrt_target = None, MIN_SQRT

        # --- computeSwapStep (exactIn, zeroForOne) ---
        less_fee = remaining * (FEE_ONE - fee) // FEE_ONE
        in_target = get_amount0_delta(sqrt_target, sqrtP, L, True)
        if less_fee >= in_target:
            sqrt_next = sqrt_target
        else:
            sqrt_next = get_next_sqrt_price_from_amount0_rounding_up(
                sqrtP, L, less_fee)
        hit = sqrt_next == sqrt_target
        amount_in = in_target if hit else get_amount0_delta(
            sqrt_next, sqrtP, L, True)
        amount_out = get_amount1_delta(sqrt_next, sqrtP, L, False)
        if not hit:
            fee_amt = remaining - amount_in      # fee on the final segment
        else:
            fee_amt = muldiv_ru(amount_in, fee, FEE_ONE - fee)
        remaining -= amount_in + fee_amt
        out_total += amount_out
        steps.append(dict(tick=tick_next, sqrt_next=sqrt_next,
                          amount_in=amount_in, amount_out=amount_out,
                          fee=fee_amt, L=L))
        if hit and tick_next is not None:
            L += -ticks[tick_next][1]            # zeroForOne negates net
        sqrtP = sqrt_next
        if not hit:
            break
    return sqrtP, L, out_total, remaining, steps


def walk_exact_out(sqrtP, L, ticks, out_want, fee):
    """The Pool.sol swap loop, exactOutput, zeroForOne (token1 out)."""
    remaining_out = out_want
    gross_in = 0
    steps = []
    ts = sorted(ticks)
    while remaining_out > 0 and sqrtP > MIN_SQRT:
        ratios = [get_sqrt_ratio_at_tick(t) for t in ts]
        idx = bisect_right(ratios, sqrtP) - 1
        if idx >= 0:
            tick_next, sqrt_target = ts[idx], ratios[idx]
        else:
            tick_next, sqrt_target = None, MIN_SQRT

        # --- computeSwapStep (exactOut, zeroForOne) ---
        out_target = get_amount1_delta(sqrt_target, sqrtP, L, False)
        if remaining_out >= out_target:
            sqrt_next = sqrt_target
        else:
            # getNextSqrtPriceFromOutput, zeroForOne: price DOWN
            quo = ceil_div(remaining_out << 96, L) if L else 0
            if L == 0 or sqrtP - quo < MIN_SQRT:
                sqrt_next = MIN_SQRT
            else:
                sqrt_next = sqrtP - quo
        hit = sqrt_next == sqrt_target
        amount_out = out_target if hit else get_amount1_delta(
            sqrt_next, sqrtP, L, False)
        if amount_out > remaining_out:
            amount_out = remaining_out          # cap to the desired out
        amount_in = get_amount0_delta(sqrt_next, sqrtP, L, True)
        fee_amt = muldiv_ru(amount_in, fee, FEE_ONE - fee)
        remaining_out -= amount_out
        gross_in += amount_in + fee_amt
        steps.append(dict(tick=tick_next, sqrt_next=sqrt_next,
                          amount_in=amount_in, amount_out=amount_out,
                          fee=fee_amt, L=L))
        if hit and tick_next is not None:
            L += -ticks[tick_next][1]
        sqrtP = sqrt_next
        if not hit:
            break
    return sqrtP, L, gross_in, remaining_out, steps


def main():
    d = LEAD
    print(f"LEAD {d['tx'][:20]}… pool={d['pool']} fee={d['fee']}")
    print(f"  before : sqrtP={d['sqrtP']} L={d['L']}")
    print(f"  swap   : in={d['gross']} token0  out={d['out_chain']} token1")
    print(f"  target : sqrtP={d['target']}\n")

    # live pool state (drift check)
    slot0 = rpc("eth_call", [{"to": d["pool"], "data": "0x3850c7bd"},
                             "latest"])[2:]
    sqrt_now = int(slot0[0:64], 16)
    tick_now = int.from_bytes(bytes.fromhex(slot0[64:128]), "big", signed=True)
    liq_now = int(rpc("eth_call", [{"to": d["pool"], "data": "0x1a686502"},
                                    "latest"]), 16)
    print(f"pool now: sqrtP={sqrt_now} (tick {tick_now}) L={liq_now}")
    print(f"           (state read at LATEST; the swap is hours old --")
    print(f"            LP changes in between would distort the walk)\n")

    t_before = tick_of(d["sqrtP"])
    t_after = tick_of(d["target"])
    print(f"tick before ~{t_before}, tick after ~{t_after} "
          f"({t_before - t_after} ticks)\n")

    t_lo, t_hi = t_after - 64, t_before + 64
    ticks = fetch_ticks(d["pool"], t_lo, t_hi)
    print(f"initialized ticks in [{t_lo}, {t_hi}]: {len(ticks)}")
    for t in sorted(ticks):
        g, net = ticks[t]
        print(f"  tick {t:>8}: liquidityGross={g:>25} liquidityNet={net:>25}")

    if not ticks:
        print("\nno initialized ticks in range -> symmetric-crossing "
              "hypothesis DEAD (at least at current state)")
        return

    print("\n--- walk, exactInput (gross in) ---")
    s, L_end, out, rem, steps = walk_exact_in(
        d["sqrtP"], d["L"], ticks, d["gross"], d["fee"])
    for i, st in enumerate(steps):
        print(f"  seg {i}: from L={st['L']:.3e} -> tick {st['tick']} "
              f"in={st['amount_in']} out={st['amount_out']} fee={st['fee']}")
    print(f"  final sqrtP  ours={s}\n"
          f"              chain={d['target']}  "
          f"({'==' if s == d['target'] else '!='})")
    print(f"  total out    ours={out}\n"
          f"              chain={d['out_chain']}  "
          f"({'==' if out == d['out_chain'] else '!='})")
    print(f"  remaining input {rem}")

    print("\n--- walk, exactOutput (out fixed) ---")
    s2, L_end2, gin, rem2, steps2 = walk_exact_out(
        d["sqrtP"], d["L"], ticks, d["out_chain"], d["fee"])
    for i, st in enumerate(steps2):
        print(f"  seg {i}: from L={st['L']:.3e} -> tick {st['tick']} "
              f"in={st['amount_in']} out={st['amount_out']} fee={st['fee']}")
    print(f"  final sqrtP  ours={s2}\n"
          f"              chain={d['target']}  "
          f"({'==' if s2 == d['target'] else '!='})")
    print(f"  total gross  ours={gin}\n"
          f"              chain={d['gross']}  "
          f"({'==' if gin == d['gross'] else '!='})")
    print(f"  remaining out {rem2}")

    ok_in = (s == d["target"] and out == d["out_chain"])
    ok_out = (s2 == d["target"] and gin == d["gross"])
    print("\nVERDICT:",
          "exactInput walk reproduces the event bit-for-bit" if ok_in else
          "exactOutput walk reproduces the event bit-for-bit" if ok_out else
          "neither walk reproduces it -- hypothesis not confirmed "
          "(possibly drifted tick state)")
    if ok_in or ok_out:
        print("=> the LEAD is a multi-segment tick-crossing swap; "
              "single-L replay was the wrong model, the chain is fine.")


if __name__ == "__main__":
    main()
