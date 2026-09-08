#!/usr/bin/env python3
"""
l4.py -- fidelity ladder L4: bit-exact replay of the AMM interaction layer.

Technique (validated on a real swap, bit-for-bit):
  1. Find CoW settlements whose receipts contain Uniswap V3 Swap events
     (topic0 = 0xc42079f9...). Modern CoW solvers route via V3/Vault
     aggregators, not V2 pairs, so V3 is the target.
  2. SAME-BLOCK EVENT WALK: eth_call at block N-1 gives end-of-previous-
     block state, which DRIFTS if earlier txs in block N touched the pool
     (we measured this on the first try: predicted price moved less than
     chain). Fix: walk every tx in block N BEFORE ours, collect V3 Swap
     events on the same pool; the previous event's final (sqrtP, L) IS
     our true before-state.
  3. Run OUR v3math (getNextSqrtPriceFromAmount0RoundingUp +
     amount1Delta, EVM rounding semantics) on the before-state and the
     fee-adjusted input.
  4. Compare bit-for-bit with the on-chain event: final sqrtPriceX96
     and the output amount.
"""
import sys

sys.path.insert(0, ".")
from solver.mainnet import rpc, SETTLEMENT                     # noqa: E402
from solver.v3math import (swap_within_tick_exact_in,         # noqa: E402
                           get_next_sqrt_price_from_amount1_rounding_down,
                           get_amount0_delta)

V3_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"


def s256(x):
    return x - 2 ** 256 if x >= 2 ** 255 else x


def parse_v3_swap(log):
    d = log["data"][2:]
    if len(d) < 320:
        return None
    w = [int(d[i * 64:(i + 1) * 64], 16) for i in range(5)]
    return dict(pool=log["address"].lower(),
                amount0=s256(w[0]), amount1=s256(w[1]),
                sqrtP=w[2], L=w[3], tick=s256(w[4]))


def replay_one(our_txh):
    """Event-walk + replay. Returns result dicts or None."""
    tx = rpc("eth_getTransactionByHash", [our_txh])
    block = int(tx["blockNumber"], 16)
    rc = rpc("eth_getTransactionReceipt", [our_txh])
    if int(rc["status"], 16) != 1:
        return None

    ours = [parse_v3_swap(l) for l in rc.get("logs", [])
            if l.get("topics") and l["topics"][0] == V3_TOPIC]
    ours = [o for o in ours if o]
    if not ours:
        return None

    blk = rpc("eth_getBlockByNumber", [hex(block), True])
    txs = blk["transactions"]
    our_idx = next(i for i, t in enumerate(txs) if t["hash"] == our_txh)

    def pool_state_block1(pool):
        slot0 = rpc("eth_call", [{"to": pool, "data": "0x3850c7bd"},
                                 hex(block - 1)])[2:]
        sqrtP = int(slot0[0:64], 16)
        L = int(rpc("eth_call", [{"to": pool, "data": "0x1a686502"},
                                 hex(block - 1)]), 16)
        return sqrtP, L

    def event_walk_prev(pool):
        """Expensive: receipts of earlier txs in the block. Only used when
        the block-1 prediction mismatches (same-block drift)."""
        for t in reversed(txs[:our_idx]):
            rc2 = rpc("eth_getTransactionReceipt", [t["hash"]])
            for l in reversed(rc2.get("logs", [])):
                if (l.get("topics") and l["topics"][0] == V3_TOPIC
                        and l["address"].lower() == pool):
                    ev = parse_v3_swap(l)
                    if ev:
                        return ev["sqrtP"], ev["L"]
        return None

    def predict(sqrtP, L, fee, o):
        if o["amount0"] > 0:
            a_in_gross, a_out_chain = o["amount0"], -o["amount1"]
            s_next, used, a_out = swap_within_tick_exact_in(
                sqrtP, L, a_in_gross * (1_000_000 - fee) // 1_000_000,
                0, True)
        else:
            a_in_gross, a_out_chain = -o["amount1"], o["amount0"]
            s_next = get_next_sqrt_price_from_amount1_rounding_down(
                sqrtP, L, a_in_gross * (1_000_000 - fee) // 1_000_000)
            a_out = get_amount0_delta(sqrtP, s_next, L, False)
        return s_next, a_out, a_out_chain

    results = []
    for o in ours:
        pool = o["pool"]
        try:
            sqrtP, L = pool_state_block1(pool)
            fee = int(rpc("eth_call", [{"to": pool, "data": "0xddca3f43"},
                                       hex(block - 1)]), 16)
        except RuntimeError:
            results.append(dict(pool=pool, src="no-state", fee=None,
                                sqrtP_match=None, out_match=None,
                                sqrtP_pred=None, sqrtP_chain=None,
                                out_pred=None, out_chain=None))
            continue
        src = "block-1"
        s_next, a_out, a_out_chain = predict(sqrtP, L, fee, o)
        if not (s_next == o["sqrtP"] and a_out == a_out_chain):
            walked = event_walk_prev(pool)
            if walked:
                sqrtP, L = walked
                src = "event-walk"
                s_next, a_out, a_out_chain = predict(sqrtP, L, fee, o)

        results.append(dict(
            pool=pool, src=src, fee=fee,
            sqrtP_match=(s_next == o["sqrtP"]),
            out_match=(a_out == a_out_chain),
            sqrtP_pred=s_next, sqrtP_chain=o["sqrtP"],
            out_pred=a_out, out_chain=a_out_chain))
    return results


def main():
    head = int(rpc("eth_blockNumber", []), 16)
    txs = []
    for b in range(head - 60, head):
        blk = rpc("eth_getBlockByNumber", [hex(b), True]) or {}
        for tx in blk.get("transactions", []):
            if (tx.get("to") or "").lower() == SETTLEMENT:
                txs.append(tx["hash"])
        if len(txs) >= 25:
            break

    print(f"scanning {len(txs)} settlements for V3 swap events...\n")
    n = exact = 0
    for txh in txs:
        rs = replay_one(txh)
        if not rs:
            continue
        for r in rs:
            if r["src"] == "no-state":
                print(f"[SKIP] {txh[:18]}… pool={r['pool'][:10]}… "
                      f"(slot0 reverted; V4/exotic pool)")
                continue
            n += 1
            ok = r["sqrtP_match"] and r["out_match"]
            exact += 1 if ok else 0
            print(f"[{'EXACT' if ok else 'DIFF '}] {txh[:18]}… "
                  f"pool={r['pool'][:10]}… fee={r['fee']} state={r['src']}")
            print(f"   sqrtP: ours={r['sqrtP_pred']:,} "
                  f"chain={r['sqrtP_chain']:,} "
                  f"({'==' if r['sqrtP_match'] else '!='})")
            print(f"   out  : ours={r['out_pred']:,} "
                  f"chain={r['out_chain']:,} "
                  f"({'==' if r['out_match'] else '!='})")
    print("=" * 60)
    print(f"V3 swaps replayed: {n}, bit-exact: {exact} "
          f"({exact * 100 // n if n else 0}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
