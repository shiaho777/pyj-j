#!/usr/bin/env python3
"""
scan.py -- the differential scanner (C-line weapon).

Our exact integer math vs chain execution, at scale, on the interesting
population: transactions that swap the SAME V3 pool at least twice
(arbitrage / solver / rebalancing traffic -- exactly the MEV-shaped
flow a bug would hide in).

Why same-tx consecutive pairs: the previous swap event's final state IS
the next swap's true before-state -- no eth_call drift, and the receipt
proves no other pool event (mint/burn) happened in between. Each pair is
replayed bit-for-bit; mismatches get an IMPLIED FEE INTERVAL extracted
by binary search, so classification is evidence, not assumption:

  MATCH     our math == chain, bit-exact (the evidence base)
  CROSS     liquidity changed during the swap (tick boundary crossed);
            single-range replay not applicable -- not a lead
  FEE-DIFF  some fee explains the event but not the pool's fee():
            dynamic-fee evidence (implied interval printed)
  LEAD      no fee explains the event: our math and the chain disagree
            structurally. INVESTIGATE. (A lead is a question, not a
            finding -- our own math is always suspect #1.)

Also fixes L4's token1-in sign bug: V3 Swap event amounts are POOL
balance deltas (positive = token flowed in), so token1-in reads
a_in = amount1, out = -amount0.
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from solver.l4 import V3_TOPIC, parse_v3_swap            # noqa: E402
from solver.mainnet import rpc                            # noqa: E402
from solver.v3math import (                               # noqa: E402
    get_amount0_delta, get_amount1_delta,
    get_next_sqrt_price_from_amount0_rounding_up,
    get_next_sqrt_price_from_amount1_rounding_down)

FEE_ONE = 1_000_000
Q96 = 2 ** 96


def ceil_div(a, b):
    return -((-a) // b)


def muldiv_ru(a, b, d):
    """FullMath.mulDivRoundingUp."""
    return (a * b + d - 1) // d

# publicnode allows address-scoped reads but bans topic-only getLogs;
# drpc allows topic-only getLogs but rate-limits hard. So: drpc for the
# log scan (small chunks, paced), publicnode for receipts and eth_call.
LOGS_RPC = "https://eth.drpc.org"


def rpc_post(url, method, params):
    req = urllib.request.Request(url, data=json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method,
         "params": params}).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 (research)",
                 "Accept": "application/json"})
    try:
        raw = urllib.request.urlopen(req, timeout=40).read()
    except urllib.error.HTTPError as e:
        # drpc reports rate limits as raw HTTP errors, not JSON-RPC
        # errors; normalize so the retry logic can see them
        body = b""
        try:
            body = e.read()[:200]
        except Exception:                              # noqa: BLE001
            pass
        raise RuntimeError(f"HTTP {e.code} {body.decode('utf-8', 'replace')}")
    out = json.loads(raw)
    if "error" in out:
        raise RuntimeError(str(out["error"]))
    return out["result"]


# ------------------------------------------------------------------ rpc
def rpc_retry(method, params, tries=4):
    for i in range(tries):
        try:
            time.sleep(0.12)
            return rpc(method, params)
        except Exception as e:                             # noqa: BLE001
            if i == tries - 1:
                raise
            wait = 1.5 * (i + 1)
            print(f"  [rpc retry {i + 1}] {e} -- sleeping {wait:.1f}s")
            time.sleep(wait)


# ------------------------------------------------- monotone binary search
def _last(lo, hi, pred):
    """Last index in [lo, hi] with pred True (pred True on a prefix).
    Returns lo-1 if none."""
    best = lo - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if pred(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _first(lo, hi, pred):
    """First index in [lo, hi] with pred True (pred True on a suffix).
    Returns hi+1 if none."""
    best = hi + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if pred(mid):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best


# -------------------------------------------------- implied-fee algebra
def net_interval(sqrtP, L, gross, target, zero_for_one):
    """Integer net-input interval [n_lo, n_hi] whose predicted next
    price equals target EXACTLY. None if the rounding ladder steps over
    the chain price (no net reproduces it) -- a structural mismatch."""
    if zero_for_one:
        def s(n):
            return get_next_sqrt_price_from_amount0_rounding_up(sqrtP, L, n)
        # s non-increasing: >= target on a prefix, <= target on a suffix
        n_hi = _last(0, gross, lambda n: s(n) >= target)
        n_lo = _first(0, gross, lambda n: s(n) <= target)
    else:
        def s(n):
            return get_next_sqrt_price_from_amount1_rounding_down(sqrtP, L, n)
        # s non-decreasing
        n_lo = _first(0, gross, lambda n: s(n) >= target)
        n_hi = _last(0, gross, lambda n: s(n) <= target)
    if not (0 <= n_lo <= n_hi <= gross):
        return None
    # monotonicity makes s == target throughout; verify endpoints anyway
    if s(n_lo) != target or s(n_hi) != target:
        return None
    return n_lo, n_hi


def fee_interval(gross, n_lo, n_hi):
    """Fee values [f_lo, f_hi] (hundredths of a bip) whose net lands in
    [n_lo, n_hi]. net(f) = gross - ceil(gross*f/1e6), non-increasing.
    None if no fee in [0, 1e6] works (e.g. implied net > gross)."""
    def net(f):
        return gross - (gross * f + FEE_ONE - 1) // FEE_ONE

    f_hi = _last(0, FEE_ONE, lambda f: net(f) >= n_lo)
    f_lo = _first(0, FEE_ONE, lambda f: net(f) <= n_hi)
    if not (0 <= f_lo <= f_hi <= FEE_ONE):
        return None
    return f_lo, f_hi


# ---------------------------------------------------------- pair replay
def classify_pair(ev1, ev2, fee):
    """Replay ev2 from ev1's final state with pool fee. Classified dict."""
    if ev2["L"] != ev1["L"]:
        return dict(cls="CROSS")
    sqrtP, L = ev1["sqrtP"], ev1["L"]
    if ev2["amount0"] > 0:
        zf1 = True
        gross, out_chain = ev2["amount0"], -ev2["amount1"]
    else:
        zf1 = False
        gross, out_chain = ev2["amount1"], -ev2["amount0"]
    if gross <= 0 or L <= 0:
        return dict(cls="DUST")

    target = ev2["sqrtP"]
    base = dict(gross=gross, out_chain=out_chain, sqrtP=sqrtP, L=L,
                target=target, zf1=zf1, fee=fee)

    net = gross - (gross * fee + FEE_ONE - 1) // FEE_ONE
    if zf1:
        s_next = get_next_sqrt_price_from_amount0_rounding_up(sqrtP, L, net)
        out_pred = get_amount1_delta(sqrtP, s_next, L, False)
    else:
        s_next = get_next_sqrt_price_from_amount1_rounding_down(sqrtP, L, net)
        out_pred = get_amount0_delta(sqrtP, s_next, L, False)
    if s_next == target and out_pred == out_chain:
        return dict(cls="MATCH")

    # pass 2: exactOutput -- the price is derived FROM the output amount
    # (SqrtPriceMath.getNextSqrtPriceFromOutput, the add=false branches),
    # and the fee is taken on the DERIVED input as ceil(in*f/(1e6-f)).
    # Discovered by this scanner: 6/6 "unexplainable" leads were exactly
    # this case, bit-for-bit.
    if zf1:
        quo = ceil_div(out_chain << 96, L)
        if 0 < quo < sqrtP:
            s_out = sqrtP - quo
            amt_in = muldiv_ru(L * Q96, sqrtP - s_out, sqrtP * s_out)
        else:
            s_out = None
    else:
        den = L * Q96 - out_chain * sqrtP
        if den > 0:
            s_out = muldiv_ru(L * Q96, sqrtP, den)
            amt_in = ceil_div(L * (s_out - sqrtP), Q96)
        else:
            s_out = None
    if s_out is not None:
        fee_amt = muldiv_ru(amt_in, fee, FEE_ONE - fee)
        if s_out == target and amt_in + fee_amt == gross:
            return dict(cls="MATCH-OUT")

    ni = net_interval(sqrtP, L, gross, target, zf1)
    if ni is None:
        return dict(cls="LEAD",
                    why="no integer net reproduces the chain price", **base)
    n_lo, n_hi = ni
    base["net_interval"] = (n_lo, n_hi)
    if n_lo > gross:
        return dict(cls="LEAD",
                    why="implied net exceeds gross input (negative fee?)",
                    **base)
    # price reachable: the output is then fully determined by the price
    out_at_target = (get_amount1_delta if zf1 else get_amount0_delta)(
        sqrtP, target, L, False)
    if out_at_target != out_chain:
        return dict(cls="LEAD",
                    why="price reachable but output inconsistent", **base)
    fi = fee_interval(gross, n_lo, n_hi)
    if fi is None:
        return dict(cls="LEAD", why="no fee in [0,1e6] fits the net", **base)
    f_lo, f_hi = fi
    if f_lo <= fee <= f_hi:
        # would imply the primary replay matched -- internal inconsistency
        return dict(cls="INTERNAL-BUG",
                    why="fee() inside implied interval but replay differed",
                    f=(f_lo, f_hi), **base)
    return dict(cls="FEE-DIFF", f=(f_lo, f_hi), **base)


# -------------------------------------------------------------- scanning
def scan(n_blocks=600, chunk=50):
    head = int(rpc_retry("eth_blockNumber", []), 16)
    logs = []
    b = head - n_blocks
    while b < head:
        step = chunk
        rate_hits = 0
        while True:
            to = min(b + step, head) - 1
            try:
                time.sleep(3.5)
                part = rpc_post(LOGS_RPC, "eth_getLogs", [
                    {"fromBlock": hex(b), "toBlock": hex(to),
                     "topics": [V3_TOPIC]}])
                logs.extend(part)
                b = to + 1
                break
            except RuntimeError as ex:
                s = str(ex).lower()
                if ("rate limit" in s or "http 429" in s) and rate_hits < 6:
                    rate_hits += 1
                    time.sleep(10)
                    continue
                # "can't route" = range too large for drpc's providers
                if step > 10:
                    step //= 2
                    time.sleep(3)
                    continue
                raise
        print(f"  .. {len(logs)} logs through block {to}", flush=True)

    groups = {}
    for l in logs:
        ev = parse_v3_swap(l)
        if not ev:
            continue
        ev["tx"] = l["transactionHash"]
        ev["logIndex"] = int(l["logIndex"], 16)
        groups.setdefault((l["transactionHash"], ev["pool"]), []).append(ev)
    for v in groups.values():
        v.sort(key=lambda e: e["logIndex"])
    cand = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"blocks [{head - n_blocks}, {head}): {len(logs)} V3 swap events, "
          f"{len(groups)} (tx,pool) groups, {len(cand)} multi-swap groups")

    receipts = {}

    def receipt(tx):
        if tx not in receipts:
            rc = rpc_retry("eth_getTransactionReceipt", [tx])
            if int(rc.get("status", "0x0"), 16) != 1:
                receipts[tx] = None
            else:
                receipts[tx] = rc.get("logs", [])
        return receipts[tx]

    fees = {}

    def pool_fee(pool):
        if pool not in fees:
            try:
                fees[pool] = int(rpc_retry("eth_call", [
                    {"to": pool, "data": "0xddca3f43"}, "latest"]), 16)
            except RuntimeError:
                fees[pool] = None
        return fees[pool]

    results = []
    n_pairs_dropped = 0
    for (tx, pool), evs in sorted(cand.items()):
        rc_logs = receipt(tx)
        if rc_logs is None:
            continue
        fee = pool_fee(pool)
        for ev1, ev2 in zip(evs, evs[1:]):
            li1, li2 = ev1["logIndex"], ev2["logIndex"]
            clean = True
            for l in rc_logs:
                if (l["address"].lower() == pool
                        and li1 < int(l["logIndex"], 16) < li2
                        and l.get("topics")
                        and l["topics"][0] != V3_TOPIC):
                    clean = False
                    break
            if not clean:
                n_pairs_dropped += 1
                continue
            if fee is None:
                results.append(dict(tx=tx, pool=pool, cls="EXOTIC"))
                continue
            r = classify_pair(ev1, ev2, fee)
            r.update(tx=tx, pool=pool)
            results.append(r)
    return results, n_pairs_dropped


def report(results, n_dropped):
    order = ["MATCH", "MATCH-OUT", "CROSS", "FEE-DIFF", "LEAD",
             "INTERNAL-BUG", "EXOTIC", "DUST"]
    counts = {k: 0 for k in order}
    for r in results:
        counts[r["cls"]] += 1
    n = len(results)
    print("=" * 78)
    print("  DIFFERENTIAL SCAN -- our integer math vs chain execution")
    print("=" * 78)
    print(f"  replayable pairs: {n}"
          + (f"  (dropped {n_dropped}: pool event between swaps)" if n_dropped else ""))
    for k in order:
        if counts[k]:
            print(f"  {k:14} {counts[k]:>6}   "
                  f"({counts[k] * 100 // n if n else 0}%)")

    by_pool = {}
    for r in results:
        by_pool.setdefault(r["pool"], []).append(r)
    print(f"\n  per pool ({len(by_pool)}):")
    for pool, rs in sorted(by_pool.items(),
                           key=lambda kv: -len(kv[1])):
        cnt = {}
        for r in rs:
            cnt[r["cls"]] = cnt.get(r["cls"], 0) + 1
        parts = "  ".join(f"{k}:{v}" for k, v in sorted(cnt.items()))
        print(f"  {pool[:16]}…  {len(rs):>3} pairs  {parts}")

    leads = [r for r in results if r["cls"] in ("LEAD", "INTERNAL-BUG")]
    fee_diffs = [r for r in results if r["cls"] == "FEE-DIFF"]
    if fee_diffs:
        print(f"\n  FEE-DIFF implied intervals (hundredths of a bip):")
        for r in fee_diffs[:12]:
            print(f"    pool={r['pool'][:14]}… implied {r['f'][0]}..{r['f'][1]}"
                  f"  fee()={r['fee']}  tx={r['tx'][:16]}…")
    if leads:
        print(f"\n  LEADS ({len(leads)}) -- INVESTIGATE:")
        for r in leads:
            print(f"    tx={r['tx']}\n    pool={r['pool']}")
            print(f"      why: {r['why']}")
            print(f"      before sqrtP={r['sqrtP']} L={r['L']}")
            print(f"      swap gross={r['gross']} out_chain={r['out_chain']}"
                  f" target={r['target']} zf1={r['zf1']}")
            if "net_interval" in r:
                print(f"      net interval {r['net_interval']}")
    else:
        print("\n  no leads in this window")
    print("=" * 78)


def main():
    n_blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    results, n_dropped = scan(n_blocks=n_blocks)
    report(results, n_dropped)


if __name__ == "__main__":
    main()
