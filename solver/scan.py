#!/usr/bin/env python3
"""
scan.py -- the differential scanner (C-line weapon), multi-chain.

Our exact integer math vs chain execution, at scale, on the interesting
population: transactions that swap the SAME V3 pool at least twice
(arbitrage / solver / rebalancing traffic -- exactly the MEV-shaped flow
a bug would hide in). Leads do not grow on canonical Uniswap V3 (five
years of bounty pressure); they grow on forks -- so the scanner is
factory-aware and runs on the fork-heavy chains too.

Why same-tx consecutive pairs: the previous swap event's final state IS
the next swap's true before-state -- no eth_call drift, and the receipt
proves no other pool event (mint/burn) happened in between. Each pair is
replayed bit-for-bit under BOTH swap semantics:

  exactInput   price from the input (net of fee), output derived
  exactOutput  price from the OUTPUT amount (getNextSqrtPriceFromOutput,
               the add=false branches), fee = ceil(in*f/(1e6-f)) on the
               derived input -- discovered by this very scanner

Classification (evidence, not assumption -- implied-fee intervals are
extracted by binary search when the direct replay misses):

  MATCH      exactInput swap, price+output bit-exact
  MATCH-OUT  exactOutput swap, price+gross bit-exact
  CROSS      liquidity changed during the swap (tick boundary crossed)
  FEE-DIFF   some fee explains the event but not the pool's fee()
  LEAD       no fee explains the event: our math and the chain disagree
             structurally. INVESTIGATE. (Suspect #1 is always our own
             math -- that is how the exactOutput gap was found.)

RPC reality (honest infrastructure notes):
  - publicnode bans topic-only eth_getLogs but serves reads
  - drpc allows topic-only getLogs but only for the last ~128 blocks and
    rate-limits hard
  => drpc for logs (small chunks, paced), publicnode for everything else.
  Results append to solver/scanlog/<chain>.jsonl so runs accumulate.
"""
import datetime
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from solver.l4 import V3_TOPIC, parse_v3_swap            # noqa: E402
from solver.tickwalk import capture_tick_context          # noqa: E402
from solver.v3math import (                               # noqa: E402
    get_amount0_delta, get_amount1_delta,
    get_next_sqrt_price_from_amount0_rounding_up,
    get_next_sqrt_price_from_amount1_rounding_down)

FEE_ONE = 1_000_000
Q96 = 2 ** 96
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scanlog")

# Uniswap V3 and Pancake V3 factories are deployed at the same address on
# every chain (deterministic deployment); anything else is worth a look.
FACTORIES = {
    "0x1f98431c8ad98523631ae4a59f267346ea31f984": "uniswap-v3",
    "0x0bfbcf9fa4f9c56b0f40a671ad40e0805a091865": "pancake-v3",
}

CHAINS = {
    # NOTE: windows are capped near the providers' receipt horizon
    # (~128 blocks on fast chains) -- on BSC/Base a bigger window ages
    # out of publicnode's receipt serving before the scan reaches it.
    "eth":  dict(logs="https://eth.drpc.org",
                 read="https://ethereum-rpc.publicnode.com",
                 blocks=120, chunk=50, pace=3.5),
    "bsc":  dict(logs="https://bsc.drpc.org",
                 read="https://bsc-rpc.publicnode.com",
                 blocks=100, chunk=15, pace=6.0),
    "base": dict(logs="https://base.drpc.org",
                 read="https://base-rpc.publicnode.com",
                 blocks=100, chunk=15, pace=5.0),
    "arb":  dict(logs="https://arbitrum.drpc.org",
                 read="https://arbitrum-rpc.publicnode.com",
                 blocks=100, chunk=15, pace=5.0),
}


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

    # pass 2b: non-circular exactOutput check -- the circular form above
    # derives the price from the event's OUTPUT, which is the post-rounding
    # delta, not the desired amount the chain priced from. Here both legs
    # are derived from the event's FINAL PRICE and must settle exactly.
    if zf1:
        in_pred = get_amount0_delta(sqrtP, target, L, True)
        out_pred = get_amount1_delta(sqrtP, target, L, False)
    else:
        in_pred = get_amount1_delta(sqrtP, target, L, True)
        out_pred = get_amount0_delta(sqrtP, target, L, False)
    fee_pred = muldiv_ru(in_pred, fee, FEE_ONE - fee)
    if out_pred == out_chain and in_pred + fee_pred == gross:
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
def scan(chain, n_blocks=None, chunk=None):
    cfg = CHAINS[chain]
    if n_blocks is None:
        n_blocks = cfg["blocks"]
    if chunk is None:
        chunk = cfg["chunk"]
    head = int(rpc_retry(cfg["read"], "eth_blockNumber", []), 16)
    logs = []
    groups = {}
    receipts = {}

    def receipt(tx):
        if tx not in receipts:
            try:
                rc = rpc_retry(cfg["read"], "eth_getTransactionReceipt", [tx])
                if int(rc.get("status", "0x0"), 16) != 1:
                    receipts[tx] = None
                else:
                    receipts[tx] = rc.get("logs", [])
            except RuntimeError:
                # receipt aged out of the provider's horizon -> the tx
                # can never be verified; drop it (counted as stale)
                receipts[tx] = "stale"
        return receipts[tx]

    b = head - n_blocks
    while b < head:
        step = chunk
        rate_hits = 0
        while True:
            to = min(b + step, head) - 1
            try:
                time.sleep(cfg["pace"])
                part = rpc_post(cfg["logs"], "eth_getLogs", [
                    {"fromBlock": hex(b), "toBlock": hex(to),
                     "topics": [V3_TOPIC]}])
                logs.extend(part)
                b = to + 1
                break
            except RuntimeError as ex:
                s = str(ex).lower()
                transient = ("rate limit" in s or "http 429" in s
                             or s.startswith("net ") or "timed out" in s)
                if transient:
                    # never give up to soft bans: long exponential backoff
                    rate_hits += 1
                    wait = min(90, 15 + 10 * rate_hits)
                    print(f"  .. throttled (hit {rate_hits}): "
                          f"{s[:60]} -- sleeping {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                # "can't route" = range/size too large for the provider
                if step > 10:
                    step //= 2
                    time.sleep(3)
                    continue
                raise
        # fetch receipts for this chunk's multi-swap txs NOW, while they
        # are still inside the provider's receipt horizon
        chunk_groups = {}
        for l in part:
            ev = parse_v3_swap(l)
            if not ev:
                continue
            ev["tx"] = l["transactionHash"]
            ev["logIndex"] = int(l["logIndex"], 16)
            key = (l["transactionHash"], ev["pool"])
            groups.setdefault(key, []).append(ev)
            chunk_groups[key] = groups[key]
        for key, evs in chunk_groups.items():
            if len(evs) >= 2:
                receipt(key[0])
        print(f"  .. {len(logs)} logs through block {to}", flush=True)

    for v in groups.values():
        v.sort(key=lambda e: e["logIndex"])
    cand = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"[{chain}] blocks [{head - n_blocks}, {head}): {len(logs)} V3 "
          f"swap events, {len(groups)} (tx,pool) groups, "
          f"{len(cand)} multi-swap groups")

    fees = {}
    factories = {}

    def pool_fee(pool):
        if pool not in fees:
            try:
                fees[pool] = int(rpc_retry(cfg["read"], "eth_call", [
                    {"to": pool, "data": "0xddca3f43"}, "latest"]), 16)
            except RuntimeError:
                fees[pool] = None
        return fees[pool]

    def pool_factory(pool):
        if pool not in factories:
            try:
                r = rpc_retry(cfg["read"], "eth_call", [
                    {"to": pool, "data": "0xc45a0155"}, "latest"])
                factories[pool] = "0x" + r[2:][24:]
            except RuntimeError:
                factories[pool] = None
        return factories[pool]

    results = []
    n_pairs_dropped = 0
    for (tx, pool), evs in sorted(cand.items()):
        rc_logs = receipt(tx)
        if rc_logs is None or rc_logs == "stale":
            n_pairs_dropped += len(evs) - 1
            continue
        fee = pool_fee(pool)
        factory = pool_factory(pool)
        ftag = FACTORIES.get(factory, factory or "?")
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
                results.append(dict(tx=tx, pool=pool, cls="EXOTIC",
                                    factory=ftag))
                continue
            r = classify_pair(ev1, ev2, fee)
            r.update(tx=tx, pool=pool, factory=ftag)
            if r["cls"] in ("LEAD", "INTERNAL-BUG"):
                # capture the tick structure NOW, at block time: crossed
                # positions get burned (the first LEAD's position was),
                # and historical ticks sit behind the archive paywall.
                # Arriving fresh IS the native archive.
                r["tick_context"] = capture_tick_context(
                    pool, ev1["sqrtP"], ev2["sqrtP"],
                    lambda m, p: rpc_retry(cfg["read"], m, p))
                nt = len(r["tick_context"].get("ticks", {}))
                print(f"  [lead] tick context captured for {pool[:14]}…: "
                      f"{nt} initialized ticks in range", flush=True)
            results.append(r)
    return results, n_pairs_dropped, head, n_blocks, len(logs)


def report(chain, results, n_dropped):
    order = ["MATCH", "MATCH-OUT", "CROSS", "FEE-DIFF", "LEAD",
             "INTERNAL-BUG", "EXOTIC", "DUST"]
    counts = {k: 0 for k in order}
    for r in results:
        counts[r["cls"]] += 1
    n = len(results)
    print("=" * 78)
    print(f"  DIFFERENTIAL SCAN [{chain}] -- our integer math vs chain")
    print("=" * 78)
    print(f"  replayable pairs: {n}"
          + (f"  (dropped {n_dropped}: pool event between swaps)"
             if n_dropped else ""))
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
        print(f"  {fac:14} {len(rs):>3} pairs  {parts}")

    by_pool = {}
    for r in results:
        by_pool.setdefault(r["pool"], []).append(r)
    print(f"\n  per pool ({len(by_pool)}):")
    for pool, rs in sorted(by_pool.items(), key=lambda kv: -len(kv[1])):
        cnt = {}
        for r in rs:
            cnt[r["cls"]] = cnt.get(r["cls"], 0) + 1
        parts = "  ".join(f"{k}:{v}" for k, v in sorted(cnt.items()))
        print(f"  {pool[:16]}… [{rs[0].get('factory', '?')}]  "
              f"{len(rs):>3} pairs  {parts}")

    fee_diffs = [r for r in results if r["cls"] == "FEE-DIFF"]
    if fee_diffs:
        print(f"\n  FEE-DIFF implied intervals (hundredths of a bip):")
        for r in fee_diffs[:12]:
            print(f"    pool={r['pool'][:14]}… implied {r['f'][0]}..{r['f'][1]}"
                  f"  fee()={r['fee']}  tx={r['tx'][:16]}…")
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
        print("\n  no leads in this window")
    print("=" * 78)
    return counts


def log_result(chain, head, n_blocks, n_events, results, counts):
    """Append the run to solver/scanlog/<chain>.jsonl -- evidence
    accumulates across runs instead of evaporating."""
    os.makedirs(LOGDIR, exist_ok=True)
    rec = dict(ts=datetime.datetime.utcnow().isoformat() + "Z",
               chain=chain, head=head, blocks=n_blocks, events=n_events,
               pairs=len(results), counts=counts,
               leads=[{k: r[k] for k in ("tx", "pool", "factory", "cls",
                                         "why", "sqrtP", "L", "gross",
                                         "out_chain", "target", "zf1",
                                         "fee", "tick_context")
                      if k in r}
                      for r in results if r["cls"] in ("LEAD", "INTERNAL-BUG",
                                                       "FEE-DIFF")])
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
            results, n_dropped, head, nb, n_events = scan(chain, n_blocks)
            counts = report(chain, results, n_dropped)
            log_result(chain, head, nb, n_events, results, counts)
            if i + 1 < len(args):
                time.sleep(20)          # inter-chain cooldown


if __name__ == "__main__":
    main()
