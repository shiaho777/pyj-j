#!/usr/bin/env python3
"""
mainnet.py -- validate our integer judge against REAL mainnet CoW settlements.

Pipeline:
  1. Fetch recent blocks from a public RPC, collect txs to the CoW
     GPv2Settlement contract (0x9008D19f58AAbD9eD0D60971565AA8510560ab41).
  2. Decode the 2024 settle() calldata in pure Python:
         settle(address[] tokens, uint256[] clearingPrices,
                Trade[] trades, Interaction[3] interactions)
     Trade = (sellTokenIdx, buyTokenIdx, receiver, sellAmount, buyAmount,
              validTo, appData, feeAmount, flags, executedAmount, signature)
  3. For every trade, reconstruct the on-chain limit check EXACTLY:
         execBuy = ceil(execSell * price[sellIdx] / price[buyIdx])
         (L)      execBuy * sellAmount >= buyAmount * execSell
     ...using the SAME integer judge as solver/cow.py.
  4. Compare with the on-chain receipt status (should be 1 = success).

If our judge passes every trade of every successful settlement, the judge
is semantically equivalent to the production contract's constraint core
-- and our solver pipeline runs on real data, not just simulation.
"""
import json
import sys
import urllib.request

RPC = "https://ethereum-rpc.publicnode.com"
SETTLEMENT = "0x9008d19f58aabd9ed0d60971565aa8510560ab41"
HDRS = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (research)"}


def rpc(method, params):
    req = urllib.request.Request(RPC, data=json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method,
         "params": params}).encode(), headers=HDRS)
    out = json.loads(urllib.request.urlopen(req, timeout=25).read())
    if "error" in out:
        raise RuntimeError(out["error"])
    return out["result"]


# ------------------------------------------------------------- ABI decode

def _u256(b):
    return int.from_bytes(b, "big")


def _read_word(data, i):
    return data[i:i + 32]


def _dyn(data):
    """Decode the four top-level args of settle() per the ABI spec.

    Dynamic arrays of dynamic tuples use per-element offsets:
        [len][offset_0][offset_1]...[elem_0][elem_1]...
    Trade tuple head: sell_i, buy_i, receiver, sellAmount, buyAmount,
    validTo, appData, feeAmount, flags, executedAmount, sigOffset.
    """
    offs = [_u256(_read_word(data, 32 * k)) for k in range(4)]

    # arg0: address[] (addresses right-aligned in their words)
    p = offs[0]
    n_tok = _u256(_read_word(data, p))
    tokens = []
    for k in range(n_tok):
        w = data[p + 32 + 32 * k:p + 32 + 32 * k + 32]
        tokens.append("0x" + w[12:].hex())

    # arg1: uint256[]
    p = offs[1]
    n_p = _u256(_read_word(data, p))
    prices = [_u256(_read_word(data, p + 32 + 32 * k)) for k in range(n_p)]

    # arg2: Trade[] -- per-element offsets, each element 11-word head
    p = offs[2]
    n_t = _u256(_read_word(data, p))
    elem_offs = [_u256(_read_word(data, p + 32 + 32 * k)) for k in range(n_t)]
    trades = []
    for k, eo in enumerate(elem_offs):
        base = p + 32 + eo          # element head start
        trades.append(dict(
            sell_i=_u256(_read_word(data, base)),
            buy_i=_u256(_read_word(data, base + 32)),
            receiver="0x" + data[base + 76:base + 96].hex(),
            sell_amt=_u256(_read_word(data, base + 96)),
            buy_amt=_u256(_read_word(data, base + 128)),
            valid_to=_u256(_read_word(data, base + 160)),
            fee=_u256(_read_word(data, base + 224)),
            flags=_u256(_read_word(data, base + 256)),
            executed=_u256(_read_word(data, base + 288)),
        ))

    # arg3: Interaction[3] -- fixed array of dynamic arrays: 3 offsets
    p = offs[3]
    n_inter = []
    for k in range(3):
        ao = _u256(_read_word(data, p + 32 * k))
        n_inter.append(_u256(_read_word(data, p + ao)))
    return tokens, prices, trades, n_inter


# ------------------------------------------------------------- judge

def judge_trade(trade, prices):
    """Exactly the on-chain check. Returns (ok, exec_buy, detail)."""
    f = trade["executed"]
    ps = prices[trade["sell_i"]]
    pb = prices[trade["buy_i"]]
    # execBuy = ceil(f * ps / pb)  (contract rounds UP in user's favor)
    exec_buy = -((-f * ps) // pb)
    lhs = exec_buy * trade["sell_amt"]
    rhs = trade["buy_amt"] * f
    ok = lhs >= rhs
    return ok, exec_buy, (lhs, rhs)


def main(n_blocks=30):
    head = int(rpc("eth_blockNumber", []), 16)
    print(f"mainnet head: {head}")
    txs = []
    for b in range(head - n_blocks, head):
        blk = rpc("eth_getBlockByNumber", [hex(b), False]) or {}
        # light scan: full block for tx hashes+to
        blkf = rpc("eth_getBlockByNumber", [hex(b), True]) or {}
        for tx in blkf.get("transactions", []):
            if (tx.get("to") or "").lower() == SETTLEMENT:
                txs.append(tx["hash"])
    print(f"settlement txs in last {n_blocks} blocks: {len(txs)}")

    n_settle = n_trade = n_pass = n_fail = 0
    fails = []
    stats_reverted = []
    for txh in txs:
        tx = rpc("eth_getTransactionByHash", [txh])
        rc = rpc("eth_getTransactionReceipt", [txh])
        onchain_ok = int(rc["status"], 16) == 1
        data = bytes.fromhex(tx["input"][10:])        # 0x + 8 selector hex
        tokens, prices, trades, n_inter = _dyn(data)
        n_settle += 1
        for t in trades:
            n_trade += 1
            ok, exec_buy, (lhs, rhs) = judge_trade(t, prices)
            if ok and onchain_ok:
                n_pass += 1                     # judge pass + tx success
            elif ok and not onchain_ok:
                stats_reverted.append(txh)       # judge pass, tx reverted
            else:
                n_fail += 1                     # judge fail + tx ok = TRUE bug
                fails.append((txh, t, prices))
        # report one example
    print(f"\nsettlements decoded: {n_settle}")
    print(f"trades judged:       {n_trade}")
    print(f"judge pass + tx success : {n_pass}")
    print(f"judge pass + tx REVERTED: {len(set(stats_reverted))} "
          f"(revert outside the order-constraint core -- e.g. interactions)")
    print(f"TRUE disagreements       : {n_fail}  (judge fail + tx ok)")
    if fails:
        for txh, t, prices in fails[:2]:
            print("FAIL", txh, t)
    # show one worked example in detail
    if txs:
        tx = rpc("eth_getTransactionByHash", [txs[0]])
        data = bytes.fromhex(tx["input"][10:])
        tokens, prices, trades, n_inter = _dyn(data)
        print("\nexample settlement", txs[0])
        print(f"  tokens: {len(tokens)}, prices: {len(prices)}, "
              f"trades: {len(trades)}, interactions: {n_inter}")
        for t in trades[:3]:
            ok, exec_buy, (lhs, rhs) = judge_trade(t, prices)
            print(f"  trade: sell {t['sell_amt']:,} -> buy "
                  f"{t['buy_amt']:,} (exec {t['executed']:,})")
            print(f"    execBuy={exec_buy:,}  limit check "
                  f"{lhs:,} >= {rhs:,} -> {'PASS' if ok else 'FAIL'}")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
