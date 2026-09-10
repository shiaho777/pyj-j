#!/usr/bin/env python3
"""
profit.py -- the margin model: converting auction score into money.

CoW's solver competition ranks settlements by USER SURPLUS. Our
simulator gives users 100% of extracted value -> maximal score, ZERO
margin. The money question this answers:

    If we bid just above production's winning surplus (prod + epsilon),
    how much margin do we keep, how often do we win, and does the
    margin clear the settlement's gas?

Method (all exact rational arithmetic):
  - per-order surplus is computed in its buy token, then converted to
    WETH through the auction's OWN clearing prices (the book is its
    price oracle -- no external feed needed)
  - gas from the settlement receipt (gasUsed * effectiveGasPrice), in
    the same WETH unit
  - margin  = our_extractable - production_bid
  - enter   = margin > gas          (rational participation)
  - net     = margin - gas

Honest limits, stated up front:
  1. We see production's CHOSEN orders, not the full auction; our
     extractable is measured on the same subset (fair comparison, but
     the real book is bigger).
  2. The bid game is simplified: we assume we can bid prod+epsilon.
     Real solving is a blind first-price-ish game against the same
     number we measure -- this model prices the EDGE, not the game.
  3. Sample is one window; solver/scanlog/cowpaper.jsonl accumulates.
"""
import sys
from fractions import Fraction

sys.path.insert(0, ".")
from solver.cowpaper import measure, call_at               # noqa: E402
from solver.scan import rpc_retry, CHAINS                  # noqa: E402

WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
UNI_USDC_WETH = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
Q96 = 2 ** 96


def surplus_in(orders, fills, buys, dec, num):
    """Total user surplus of a solution, converted to `num` (WETH or
    USDC) via the settlement's clearing prices. (Fraction|None, ok)."""
    tokens = [t.lower() for t in dec["tokens"]]
    prices = dec["prices"]
    if num not in tokens:
        return None, False
    wi = tokens.index(num)
    total = Fraction(0)
    for o in orders:
        f = fills.get(o.oid, 0)
        b = buys.get(o.oid, 0)
        if f == 0:
            continue
        s = Fraction(b * o.sell_amt - o.buy_min * f, o.sell_amt)  # buy units
        bi = tokens.index(o.buy_tok.lower())
        total += s * Fraction(prices[bi], prices[wi])
    return total, True


def surplus_in_weth(orders, fills, buys, dec, usdc_per_weth=None):
    """WETH-denominated surplus; falls back to USDC numeraire (converted
    at the live rate) when the book has no WETH -- doubles the sample."""
    s, ok = surplus_in(orders, fills, buys, dec, WETH)
    if ok:
        return s, "weth"
    s, ok = surplus_in(orders, fills, buys, dec, USDC)
    if ok and usdc_per_weth:
        return s / usdc_per_weth, "usdc"
    return None, None


def gas_weth(txh):
    rc = rpc_retry(CHAINS["eth"]["logs"], "eth_getTransactionReceipt", [txh])
    return int(rc["gasUsed"], 16) * int(rc["effectiveGasPrice"], 16)


def weth_usd():
    """Live WETH price from the canonical Uniswap USDC/WETH 0.05% pool."""
    slot0 = call_at("latest", UNI_USDC_WETH, "0x3850c7bd")[2:]
    sqrtP = int(slot0[0:64], 16)
    # price_raw = token1/token0 = WETH-raw per USDC-raw (token0=USDC)
    price_raw = Fraction(sqrtP, Q96) ** 2
    # USDC per WETH = 10^12 / price_raw  (18-dec WETH, 6-dec USDC)
    return float(Fraction(10 ** 12) / price_raw)


def main(n_blocks=100):
    import os, pickle
    cache = "/tmp/cowrows.pkl"
    if "--reuse" in sys.argv and os.path.exists(cache):
        with open(cache, "rb") as f:
            rows, head = pickle.load(f)
        print(f"(reusing {len(rows)} cached rows from {cache})")
    else:
        rows, head = measure(n_blocks)
        try:
            with open(cache, "wb") as f:
                pickle.dump((rows, head), f)
        except Exception:                           # noqa: BLE001
            pass
    print(f"\n{len(rows)} settlements measured (head {head})")
    usd = None
    try:
        usd = weth_usd()
        print(f"WETH price: ${usd:,.0f} (live, canonical pool)")
    except Exception as e:                          # noqa: BLE001
        print(f"WETH price unavailable: {e}")

    usdc_rate = Fraction(str(usd)) if usd else None
    n_conv = n_unconv = n_usdc = 0
    wins = entered = 0
    net_total = Fraction(0)
    gas_total = 0
    margins = []
    for r in rows:
        ours, num = surplus_in_weth(r["orders"], r["our_f"], r["our_b"],
                                    r["dec"], usdc_rate)
        if num is None:
            n_unconv += 1
            continue
        n_usdc += (num == "usdc")
        prod, _ = surplus_in_weth(r["orders"], r["prod_f"], r["prod_b"],
                                  r["dec"], usdc_rate)
        n_conv += 1
        margin = ours - prod                        # WETH (Fraction)
        try:
            gas = gas_weth(r["tx"])
        except Exception:                           # noqa: BLE001
            gas = 0
        gas_total += gas
        margins.append((r["tx"], margin, gas))
        if margin > 0:
            wins += 1
            if margin > gas:
                entered += 1
                net_total += margin - gas

    print("=" * 70)
    print("  THE MARGIN MODEL -- bid (production + epsilon), keep the rest")
    print("=" * 70)
    print(f"  convertible settlements : {n_conv} "
          f"({n_usdc} via USDC numeraire; {n_unconv} skipped)")
    print(f"  auctions we would WIN   : {wins}/{n_conv}")
    print(f"  auctions worth ENTERING : {entered}/{n_conv}  (margin > gas)")
    W = Fraction(10 ** 18)                          # wei -> WETH
    net_w = net_total / W
    if usd:
        print(f"  net margin (entered)    : {float(net_w):.8f} WETH "
              f"= ${float(net_w) * usd:,.4f}")
        per_day = float(net_w) * 72                 # 100 blocks ~ 20 min
        print(f"  extrapolated per day    : {per_day:.6f} WETH "
              f"= ${per_day * usd:,.2f}  (72 windows/day, same rate)")
        print(f"  gas burned measuring    : {gas_total / 10**18:.6f} ETH "
              f"= ${gas_total / 10**18 * usd:,.2f}")
    else:
        print(f"  net margin (entered)    : {float(net_w):.8f} WETH")
    if margins:
        margins.sort(key=lambda m: -m[1])
        print("\n  top margins (WETH):")
        for tx, m, g in margins[:8]:
            print(f"    {tx[:18]}…  margin {float(m / W):+.8f}  "
                  f"gas {g / 10**18:.6f}  net {float((m - g) / W):+.8f}")
    print("=" * 70)
    print("  read this as the PRICE OF THE EDGE, not a P&L: the bid game")
    print("  (blind, competitive, epsilon-sensitive) and the fuller real")
    print("  order book are not modeled. see module docstring.")
    print("=" * 70)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(args[0]) if args else 100)
