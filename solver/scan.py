#!/usr/bin/env python3
"""
scan.py -- the differential scanner (C-line weapon), archive edition.

Our exact integer math vs chain execution, at scale, on the interesting
population: ANY two consecutive swaps on a pool (same tx or CROSS-BLOCK
-- arbitrage, rebalancing, MEV-shaped flow; exactly where a bug hides).

The native archive (solver/store.py, sqlite) is the paywall-breaker:
free providers gate history behind archive fees, so we capture
continuously into our own store and replay from it. The OR-topic
getLogs (Swap | Mint | Burn) puts state-changing events in the same
stream, which means mint/burn-between detection needs NO receipts --
the receipt-aging failure mode is gone entirely.

Replay semantics (both, bit-exact):
  exactInput   price from the input (net of fee), output derived
  exactOutput  price from the OUTPUT amount (getNextSqrtPriceFromOutput,
               the add=false branches), fee = ceil(in*f/(1e6-f)) on the
               derived input -- discovered by this very scanner

Classification (evidence, not assumption -- implied-fee intervals are
extracted by binary search when the direct replay misses):

  MATCH      exactInput swap, price+output bit-exact
  MATCH-OUT  exactOutput swap, price+gross bit-exact
  CROSS      liquidity changed during the swap (tick boundary crossed)
  FEE-DIFF   some fee explains the event but not the pool's recorded
             fee -- for dynamic-fee pools this is a FEE ORACLE: the
             interval width is the measurement precision
  LEAD       no fee explains the event: our math and the chain disagree
             structurally. INVESTIGATE. (Suspect #1 is always our own
             math -- that is how the exactOutput gap was found.)

On LEAD classification the pool's tick structure is captured AT BLOCK
TIME (bitmap + ticks + slot0) -- crossed positions get burned, and
historical ticks sit behind the archive paywall. Arriving fresh IS
the native archive.
"""
import datetime
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from solver.l4 import V3_TOPIC, parse_v3_swap            # noqa: E402
from solver.store import (EventStore, MINT_TOPIC,         # noqa: E402
                          BURN_TOPIC)
from solver.tickwalk import capture_tick_context          # noqa: E402
from solver.v3math import (                               # noqa: E402
    get_amount0_delta, get_amount1_delta,
    get_next_sqrt_price_from_amount0_rounding_up,
    get_next_sqrt_price_from_amount1_rounding_down)

FEE_ONE = 1_000_000
Q96 = 2 ** 96
HERE = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(HERE, "scanstore.db")
LOGDIR = os.path.join(HERE, "scanlog")

# Uniswap V3 / Pancake V3 factories deploy at the same address on every
# chain; anything else is worth a look.
FACTORIES = {
    "0x1f98431c8ad98523631ae4a59f267346ea31f984": "uniswap-v3",
    "0x0bfbcf9fa4f9c56b0f40a671ad40e0805a091865": "pancake-v3",
    "0x33128a8fc17869897dce68ed026d694621f6fdfd": "uniswap-v3-base",
    "0x5e7bb104d84c7cb9b682aac2f3d509f5f406809a": "slipstream",
    "0xf8f2eb4940cfe7d13603dddd87f123820fc061ef": "cl-factory",
    "0x0fd83557b2be93617c9c1c1b6fd549401c74558c": "alienbase-v3",
}

# drpc: topic-only getLogs but only the last ~128 blocks, throttled.
# Official L2 endpoints: NO topic wall, deep history, 2000-block range
# cap and a result-size cap per call (adaptive halving handles both).
CHAINS = {
    "eth":  dict(logs="https://eth.drpc.org",
                 read="https://ethereum-rpc.publicnode.com",
                 blocks=120, chunk=50, pace=3.5, max_backfill=120),
    "bsc":  dict(logs="https://bsc.drpc.org",
                 read="https://bsc-rpc.publicnode.com",
                 blocks=100, chunk=15, pace=6.0, max_backfill=100),
    "base": dict(logs="https://mainnet.base.org",
                 read="https://mainnet.base.org",
                 blocks=100, chunk=50, pace=0.8, max_backfill=12000),
    "arb":  dict(logs="https://arb1.arbitrum.io/rpc",
                 read="https://arb1.arbitrum.io/rpc",
                 blocks=400, chunk=100, pace=0.8, max_backfill=30000),
}

TOPICS3 = [V3_TOPIC, MINT_TOPIC, BURN_TOPIC]


# ------------------------------------------------------------------ rpc
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
        # providers report rate limits as raw HTTP errors, not JSON-RPC
        # errors; normalize so the retry logic can see them
        body = b""
        try:
            body = e.read()[:200]
        except Exception:                              # noqa: BLE001
            pass
        raise RuntimeError(f"HTTP {e.code} {body.decode('utf-8', 'replace')}")
    except Exception as e:                             # noqa: BLE001
        # TLS drops / resets / timeouts from aggressive providers
        raise RuntimeError(f"NET {type(e).__name__}: {e}")
    out = json.loads(raw)
    if "error" in out:
        raise RuntimeError(str(out["error"]))
    return out["result"]


def rpc_retry(url, method, params, tries=4):
    for i in range(tries):
        try:
            time.sleep(0.12)
            return rpc_post(url, method, params)
        except Exception as e:                         # noqa: BLE001
            if i == tries - 1:
                raise
            wait = 1.5 * (i + 1)
            print(f"  [rpc retry {i + 1}] {str(e)[:120]} -- {wait:.1f}s")
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
        n_hi = _last(0, gross, lambda n: s(n) >= target)
        n_lo = _first(0, gross, lambda n: s(n) <= target)
    else:
        def s(n):
            return get_next_sqrt_price_from_amount1_rounding_down(sqrtP, L, n)
        n_lo = _first(0, gross, lambda n: s(n) >= target)
        n_hi = _last(0, gross, lambda n: s(n) <= target)
    if not (0 <= n_lo <= n_hi <= gross):
        return None
    if s(n_lo) != target or s(n_hi) != target:
        return None
    return n_lo, n_hi


def fee_interval(gross, n_lo, n_hi):
    """Fee values [f_lo, f_hi] (hundredths of a bip) whose net lands in
    [n_lo, n_hi]. net(f) = gross - ceil(gross*f/1e6), non-increasing."""
    def net(f):
        return gross - (gross * f + FEE_ONE - 1) // FEE_ONE

    f_hi = _last(0, FEE_ONE, lambda f: net(f) >= n_lo)
    f_lo = _first(0, FEE_ONE, lambda f: net(f) <= n_hi)
    if not (0 <= f_lo <= f_hi <= FEE_ONE):
        return None
    return f_lo, f_hi


def ceil_div(a, b):
    return -((-a) // b)


def muldiv_ru(a, b, d):
    """FullMath.mulDivRoundingUp."""
    return (a * b + d - 1) // d


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

    # pass 1: exactInput -- price from the input, net of fee
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

    # pass 2b: non-circular exactOutput check -- both legs derived from
    # the event's FINAL PRICE, settling exactly
    if zf1:
        in_pred = get_amount0_delta(sqrtP, target, L, True)
        out_pred = get_amount1_delta(sqrtP, target, L, False)
    else:
        in_pred = get_amount1_delta(sqrtP, target, L, True)
        out_pred = get_amount0_delta(sqrtP, target, L, False)
    fee_pred = muldiv_ru(in_pred, fee, FEE_ONE - fee)
    if out_pred == out_chain and in_pred + fee_pred == gross:
        return dict(cls="MATCH-OUT")

    # pass 2c: price-capped swaps (sqrtPriceLimitX96 set by the router)
    # -- the walk STOPS AT the limit, so the final price is assigned, not
    # derived: it is not in the input formula's image by construction.
    # Amounts settle as exact deltas and the fee is implied from them.
    # Discovered via a DCA drip bot on Base: 58 LEADs resolved at once.
    if out_pred == out_chain and 0 < in_pred < gross:
        fee_amt = gross - in_pred
        for f in (fee_amt * FEE_ONE // gross,
                  fee_amt * FEE_ONE // gross + 1):
            if 0 < f < FEE_ONE and \
                    in_pred + muldiv_ru(in_pred, f, FEE_ONE - f) == gross:
                return dict(cls="MATCH-LIMIT", f=(f, f), **base)

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
        return dict(cls="INTERNAL-BUG",
                    why="fee inside implied interval but replay differed",
                    f=(f_lo, f_hi), **base)
    return dict(cls="FEE-DIFF", f=(f_lo, f_hi), **base)


# -------------------------------------------------------------- scanning
def scan(chain, n_blocks=None):
    cfg = CHAINS[chain]
    if n_blocks is None:
        n_blocks = cfg["blocks"]
    store = EventStore(STORE_PATH)
    call = lambda m, p: rpc_retry(cfg["read"], m, p)   # noqa: E731

    head = int(rpc_retry(cfg["read"], "eth_blockNumber", []), 16)
    last = store.head(chain)
    # backfill from the archive cursor; on first run, a fresh window.
    # Beyond max_backfill we accept a gap (the first event after it
    # only loses pair-start eligibility, never correctness).
    start = (last + 1) if last is not None else head - n_blocks
    if head - start > cfg["max_backfill"]:
        start = head - cfg["max_backfill"]
        print(f"[{chain}] gap accepted: backfill capped at "
              f"{cfg['max_backfill']} blocks")

    known = store.known_pools(chain)
    b = start
    while b <= head:
        step = cfg["chunk"]
        rate_hits = 0
        while True:
            to = min(b + step, head)
            try:
                time.sleep(cfg["pace"])
                part = rpc_post(cfg["logs"], "eth_getLogs", [
                    {"fromBlock": hex(b), "toBlock": hex(to),
                     "topics": [TOPICS3]}])
                rows = []
                for l in part:
                    pool = l["address"].lower()
                    t0 = l["topics"][0] if l.get("topics") else ""
                    block = int(l["blockNumber"], 16)
                    if t0 == V3_TOPIC:
                        ev = parse_v3_swap(l)
                        if not ev:
                            continue
                        rows.append((chain, block, int(l["logIndex"], 16),
                                     l["transactionHash"], pool, "swap",
                                     str(ev["amount0"]), str(ev["amount1"]),
                                     str(ev["sqrtP"]), str(ev["L"]),
                                     str(ev["tick"])))
                    elif t0 in (MINT_TOPIC, BURN_TOPIC):
                        rows.append((chain, block, int(l["logIndex"], 16),
                                     l["transactionHash"], pool,
                                     "mint" if t0 == MINT_TOPIC else "burn",
                                     None, None, None, None, None))
                    else:
                        continue
                store.insert_events(chain, rows)
                store.set_head(chain, to)
                b = to + 1
                break
            except RuntimeError as ex:
                s = str(ex).lower()
                transient = ("rate limit" in s or "http 4" in s
                             or "http 5" in s or "internal error" in s
                             or s.startswith("net ") or "timed out" in s)
                if transient:
                    rate_hits += 1
                    wait = min(90, 15 + 10 * rate_hits)
                    print(f"  .. throttled (hit {rate_hits}): "
                          f"{s[:60]} -- sleeping {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                # range/size too large for the provider: halve
                if step > 10:
                    step //= 2
                    time.sleep(3)
                    continue
                raise
        print(f"  .. stored through block {to}", flush=True)

    st = store.stats(chain)
    print(f"[{chain}] archive: {st['events']:,} events, {st['pools']} pools"
          f" (through block {head})")

    # pairs form over the FULL archive (ev1 can be arbitrarily old);
    # the run reports only pairs whose second element is new
    all_pairs = store.pairs(chain)
    pairs = [(a, b) for a, b in all_pairs if b["block"] >= start]
    results = []
    for ev1, ev2 in pairs:
        pool = ev2["pool"]
        if pool not in known:
            # lazy meta capture: most pools in the stream never produce
            # a replayable pair, so we only pay 4 rpc calls for the ones
            # that do (first run); later runs capture only brand-new pools
            store.note_pool(chain, pool, ev2["block"], call)
            known.add(pool)
        fee = store.pool_fee(chain, pool)
        factory = store.pool_factory(chain, pool)
        ftag = FACTORIES.get(factory, factory or "?")
        if fee is None:
            results.append(dict(tx=ev2["tx"], pool=pool, cls="EXOTIC",
                                factory=ftag))
            continue
        r = classify_pair(ev1, ev2, fee)
        r.update(tx=ev2["tx"], pool=pool, factory=ftag)
        if r["cls"] in ("LEAD", "INTERNAL-BUG"):
            r["tick_context"] = capture_tick_context(
                pool, ev1["sqrtP"], ev2["sqrtP"], call)
            nt = len(r["tick_context"].get("ticks", {}))
            print(f"  [lead] tick context captured for {pool[:14]}…: "
                  f"{nt} initialized ticks in range", flush=True)
        results.append(r)
    return results, head, st


def report(chain, results, store_stats):
    order = ["MATCH", "MATCH-OUT", "MATCH-LIMIT", "CROSS", "FEE-DIFF",
             "LEAD", "INTERNAL-BUG", "EXOTIC", "DUST"]
    counts = {k: 0 for k in order}
    for r in results:
        counts[r["cls"]] += 1
    n = len(results)
    print("=" * 78)
    print(f"  DIFFERENTIAL SCAN [{chain}] -- our integer math vs chain")
    print(f"  archive: {store_stats['events']:,} events, "
          f"{store_stats['pools']} pools")
    print("=" * 78)
    print(f"  replayable pairs this run: {n}")
    for k in order:
        if counts[k]:
            print(f"  {k:14} {counts[k]:>6}   "
                  f"({counts[k] * 100 // n if n else 0}%)")

    by_factory = {}
    for r in results:
        by_factory.setdefault(r.get("factory", "?"), []).append(r)
    print(f"\n  by factory:")
    for fac, rs in sorted(by_factory.items(), key=lambda kv: -len(kv[1])):
        cnt = {}
        for r in rs:
            cnt[r["cls"]] = cnt.get(r["cls"], 0) + 1
        parts = "  ".join(f"{k}:{v}" for k, v in sorted(cnt.items()))
        print(f"  {fac[:20]:20} {len(rs):>5} pairs  {parts}")

    by_pool = {}
    for r in results:
        by_pool.setdefault(r["pool"], []).append(r)
    top = sorted(by_pool.items(), key=lambda kv: -len(kv[1]))[:15]
    print(f"\n  per pool (top {len(top)} of {len(by_pool)}):")
    for pool, rs in top:
        cnt = {}
        for r in rs:
            cnt[r["cls"]] = cnt.get(r["cls"], 0) + 1
        parts = "  ".join(f"{k}:{v}" for k, v in sorted(cnt.items()))
        print(f"  {pool[:16]}… [{rs[0].get('factory', '?')}]  "
              f"{len(rs):>4} pairs  {parts}")

    fee_diffs = [r for r in results if r["cls"] == "FEE-DIFF"]
    if fee_diffs:
        print(f"\n  FEE-DIFF implied intervals (hundredths of a bip):")
        for r in fee_diffs[:12]:
            print(f"    pool={r['pool'][:14]}… implied {r['f'][0]}..{r['f'][1]}"
                  f"  recorded fee={r['fee']}  tx={r['tx'][:16]}…")
    leads = [r for r in results if r["cls"] in ("LEAD", "INTERNAL-BUG")]
    if leads:
        print(f"\n  LEADS ({len(leads)}) -- INVESTIGATE:")
        for r in leads:
            print(f"    tx={r['tx']}\n    pool={r['pool']} "
                  f"[{r.get('factory', '?')}]")
            print(f"      why: {r['why']}")
            print(f"      before sqrtP={r['sqrtP']} L={r['L']}")
            print(f"      swap gross={r['gross']} out_chain={r['out_chain']}"
                  f" target={r['target']} zf1={r['zf1']}")
            if "net_interval" in r:
                print(f"      net interval {r['net_interval']}")
    else:
        print("\n  no leads in this run")
    print("=" * 78)
    return counts


def log_result(chain, head, store_stats, results, counts):
    """Append the run to solver/scanlog/<chain>.jsonl -- evidence
    accumulates across runs instead of evaporating."""
    os.makedirs(LOGDIR, exist_ok=True)
    rec = dict(ts=datetime.datetime.utcnow().isoformat() + "Z",
               chain=chain, head=head, archive=store_stats,
               pairs=len(results), counts=counts,
               leads=[{k: r[k] for k in ("tx", "pool", "factory", "cls",
                                         "why", "sqrtP", "L", "gross",
                                         "out_chain", "target", "zf1",
                                         "fee", "f", "tick_context")
                      if k in r}
                      for r in results if r["cls"] in ("LEAD", "INTERNAL-BUG",
                                                       "FEE-DIFF",
                                                       "MATCH-LIMIT")])
    with open(os.path.join(LOGDIR, f"{chain}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    args = [a for a in sys.argv[1:]]
    if not args:
        args = ["eth"]
    for i, arg in enumerate(args):
        if arg in CHAINS:
            chain = arg
            n_blocks = None
            if i + 1 < len(args) and args[i + 1].isdigit():
                n_blocks = int(args[i + 1])
            results, head, st = scan(chain, n_blocks)
            counts = report(chain, results, st)
            log_result(chain, head, st, results, counts)
            if i + 1 < len(args):
                time.sleep(5)               # inter-chain cooldown


if __name__ == "__main__":
    main()
