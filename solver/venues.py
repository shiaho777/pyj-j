#!/usr/bin/env python3
"""
venues.py -- gap attribution: WHERE does production route that we can't see?

For every measured settlement (from the cached cowpaper rows), decode
the settle() interactions and classify each target/selector:

  v4         Uniswap V4 PoolManager (singleton, hooks)
  balancer   Balancer V2 Vault
  curve      Curve exchange selectors
  v2 / v3    direct canonical AMM calls
  adapter    private solver wrapper (unverifiable selectors)
  token      plain ERC20 calls (approve/transfer -- housekeeping)

Cross-tabulated against whether OUR solver routed the book (our_vol>0).
The venue mix of the ZERO-ROUTED settlements is the build-priority list
for coverage modules: measure first, build second.
"""
import pickle
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from solver.mainnet import _u256, _read_word                # noqa: E402
from solver.scan import rpc_retry, CHAINS                   # noqa: E402

CACHE = "/tmp/cowrows.pkl"

V4_POOL_MANAGER = "0x000000000004444c5dc75cb35ae444444b4a0a6e"
BALANCER_VAULT = "0xba12222222228d8ba445958a75a0704d566bf2c8"
PERMIT2 = "0x000000000022d473030f116ddee9f6b43ac78ba3"   # token plumbing
CURVE_SELECTORS = {
    "3df02124",   # exchange_underlying
    "a6417ed8",   # exchange
    "9269881d",   # exchange (int128 variant)
    "9ce31038",   # exchange_received
}
V2_SWAP = "022c0d9f"
V3_SWAP = "128acb08"
TOKEN_SELECTORS = {
    "095ea7b3",   # approve
    "a9059cbb",   # transfer
    "23b872dd",   # transferFrom
    "d0e30db0",   # deposit (WETH)
    "2e1a7d4d",   # withdraw (WETH)
}


def decode_interactions(data):
    """Interaction[3] from settle() calldata: each element is
    (address target, uint256 value, bytes callData) with the bytes at a
    tuple-relative offset (the battle-16 decoder, byte-offset bug fixed).
    """
    offs = [_u256(_read_word(data, 32 * k)) for k in range(4)]
    p = offs[3]
    out = []
    for k in range(3):
        ao = _u256(_read_word(data, p + 32 * k))
        base = p + ao
        n = _u256(_read_word(data, base))
        for j in range(n):
            eo = _u256(_read_word(data, base + 32 + 32 * j))
            eb = base + 32 + eo
            target = "0x" + data[eb + 12:eb + 32].hex()
            boff = _u256(_read_word(data, eb + 64))
            bbase = eb + boff
            cd_len = _u256(_read_word(data, bbase))
            cd = data[bbase + 32:bbase + 32 + cd_len]
            out.append((target.lower(), cd[:4].hex()))
    return out


def classify(target, sel):
    if target == PERMIT2:
        return "token"          # Permit2 transfers are housekeeping
    if target == V4_POOL_MANAGER:
        return "v4"
    if target == BALANCER_VAULT:
        return "balancer"
    if sel in CURVE_SELECTORS:
        return "curve"
    if sel == V2_SWAP:
        return "v2"
    if sel == V3_SWAP:
        return "v3"
    if sel in TOKEN_SELECTORS:
        return "token"
    return "adapter"


def main():
    with open(CACHE, "rb") as f:
        rows, head = pickle.load(f)
    print(f"{len(rows)} cached settlements (head {head})\n")

    routed_venues = Counter()
    zero_venues = Counter()
    n_routed = n_zero = n_fail = 0
    per_tx = []
    for r in rows:
        try:
            tx = rpc_retry(CHAINS["eth"]["logs"],
                           "eth_getTransactionByHash", [r["tx"]])
            data = bytes.fromhex(tx["input"][2:])
            if data[:4].hex() != "13d79a0b":
                n_fail += 1
                continue
            inters = decode_interactions(data[4:])
        except Exception as e:                          # noqa: BLE001
            n_fail += 1
            continue
        venues = Counter(classify(t, s) for t, s in inters)
        was_routed = r["our_vol"] > 0
        if was_routed:
            n_routed += 1
            routed_venues.update(venues)
        else:
            n_zero += 1
            zero_venues.update(venues)
        per_tx.append((r["tx"][:14], was_routed, dict(venues)))

    print(f"routed by us: {n_routed}   zero-routed: {n_zero}   "
          f"undecodable: {n_fail}\n")
    print("venue mix -- settlements WE ROUTED (interactions counted):")
    for v, n in routed_venues.most_common():
        print(f"  {v:10} {n}")
    print("\nvenue mix -- settlements WE ZERO-ROUTED (the gap):")
    for v, n in zero_venues.most_common():
        print(f"  {v:10} {n}")

    zt = sum(zero_venues.values())
    if zt:
        print("\nzero-routed gap attribution (share of their interactions):")
        for v, n in zero_venues.most_common():
            print(f"  {v:10} {n*100//zt:>3}%")
    print("\nper settlement:")
    for tx, routed, venues in per_tx:
        tag = "ROUTED" if routed else "ZERO  "
        print(f"  {tx}… {tag} {venues}")
    print("\nbuild priority = the top non-v2/v3 venue in the ZERO mix.")


if __name__ == "__main__":
    main()
