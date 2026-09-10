#!/usr/bin/env python3
"""
store.py -- the native event archive (the paywall-breaker).

Free providers gate history behind archive fees. We break that by
capturing continuously into our own sqlite archive and replaying from
it -- and the official L2 endpoints serve deep history in chunks, so
gaps between runs are backfillable.

Surface multiplication: the scanner used to replay only same-tx
consecutive swap pairs (~1.5% of events, receipt-verified). With the
store, ANY two consecutive pool events replay across blocks: the
OR-topic getLogs (Swap | Mint | Burn) brings state-changing events
into the same stream, so mint/burn-between detection needs NO
receipts at all -- the receipt-aging failure mode disappears.

Replay rule: event[i] swap + event[i+1] swap, nothing between in the
pool's stream. Contiguity is by construction: the ingest cursor only
advances over completed chunks.
"""
import os
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    chain TEXT NOT NULL,
    block INTEGER NOT NULL,
    log_index INTEGER NOT NULL,
    tx TEXT NOT NULL,
    pool TEXT NOT NULL,
    kind TEXT NOT NULL,          -- swap | mint | burn
    a0 TEXT, a1 TEXT,            -- int256 as decimal text (signed)
    sqrt_p TEXT, liq TEXT, tick TEXT,
    PRIMARY KEY (chain, tx, log_index)
);
CREATE INDEX IF NOT EXISTS ev_pool_order
    ON events (chain, pool, block, log_index);
CREATE TABLE IF NOT EXISTS heads (
    chain TEXT PRIMARY KEY,
    last_block INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pool_meta (
    chain TEXT NOT NULL,
    pool TEXT NOT NULL,
    factory TEXT, token0 TEXT, token1 TEXT,
    fee INTEGER,
    first_block INTEGER,
    PRIMARY KEY (chain, pool)
);
"""

MINT_TOPIC = ("0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853"
              "ae16239d0bde")
BURN_TOPIC = ("0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028c"
              "d568da98982c")


def s256(x):
    return x - 2 ** 256 if x >= 2 ** 255 else x


class EventStore:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ------------------------------------------------------------- cursor
    def head(self, chain):
        row = self.db.execute(
            "SELECT last_block FROM heads WHERE chain=?", (chain,)).fetchone()
        return row[0] if row else None

    def set_head(self, chain, block):
        self.db.execute(
            "INSERT INTO heads (chain, last_block) VALUES (?, ?) "
            "ON CONFLICT(chain) DO UPDATE SET last_block=?",
            (chain, block, block))
        self.db.commit()

    # ------------------------------------------------------------ ingest
    def insert_events(self, chain, rows):
        self.db.executemany(
            "INSERT OR IGNORE INTO events (chain, block, log_index, tx, "
            "pool, kind, a0, a1, sqrt_p, liq, tick) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)

    def known_pools(self, chain):
        return {r[0] for r in self.db.execute(
            "SELECT pool FROM pool_meta WHERE chain=?", (chain,))}

    def note_pool(self, chain, pool, block, call):
        """First-sight meta capture, at block time (the native archive
        applies to pool metadata too: fee/factory read while fresh)."""
        try:
            factory = "0x" + call("eth_call", [
                {"to": pool, "data": "0xc45a0155"}, "latest"])[2:][24:]
        except RuntimeError:
            factory = None
        try:
            token0 = "0x" + call("eth_call", [
                {"to": pool, "data": "0x0dfe1681"}, "latest"])[2:][24:]
        except RuntimeError:
            token0 = None
        try:
            token1 = "0x" + call("eth_call", [
                {"to": pool, "data": "0xd21220a7"}, "latest"])[2:][24:]
        except RuntimeError:
            token1 = None
        try:
            fee = int(call("eth_call", [
                {"to": pool, "data": "0xddca3f43"}, "latest"]), 16)
        except RuntimeError:
            fee = None
        self.db.execute(
            "INSERT OR IGNORE INTO pool_meta (chain, pool, factory, "
            "token0, token1, fee, first_block) VALUES (?,?,?,?,?,?,?)",
            (chain, pool, factory, token0, token1, fee, block))
        self.db.commit()

    def pool_fee(self, chain, pool):
        row = self.db.execute(
            "SELECT fee FROM pool_meta WHERE chain=? AND pool=?",
            (chain, pool)).fetchone()
        return row[0] if row else None

    def pool_factory(self, chain, pool):
        row = self.db.execute(
            "SELECT factory FROM pool_meta WHERE chain=? AND pool=?",
            (chain, pool)).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------- pairs
    def pairs(self, chain, since_block=0):
        """Consecutive swap pairs (ev1, ev2) from the archive: the
        previous swap's final state IS the next swap's before-state.
        A mint/burn between them breaks the chain (skipped by the
        adjacency rule itself)."""
        cur = self.db.execute(
            "SELECT block, log_index, tx, pool, kind, a0, a1, sqrt_p, "
            "liq, tick FROM events WHERE chain=? AND block>=? "
            "ORDER BY pool, block, log_index", (chain, since_block))
        prev = None          # last event per pool, as we walk its stream
        last_pool = None
        out = []
        for row in cur:
            (block, li, tx, pool, kind, a0, a1, sp, liq, tick) = row
            if pool != last_pool:
                prev, last_pool = None, pool
            if kind == "swap":
                if prev is not None and prev["kind"] == "swap":
                    out.append((prev, dict(
                        pool=pool, tx=tx, block=block, logIndex=li,
                        amount0=int(a0), amount1=int(a1),
                        sqrtP=int(sp), L=int(liq), tick=int(tick))))
                prev = dict(pool=pool, tx=tx, block=block, logIndex=li,
                            kind="swap", amount0=int(a0), amount1=int(a1),
                            sqrtP=int(sp), L=int(liq), tick=int(tick))
            else:
                prev = dict(kind=kind)       # state break
        return out

    def stats(self, chain):
        n = self.db.execute(
            "SELECT COUNT(*) FROM events WHERE chain=?", (chain,)).fetchone()
        p = self.db.execute(
            "SELECT COUNT(DISTINCT pool) FROM events WHERE chain=?",
            (chain,)).fetchone()
        return dict(events=n[0], pools=p[0])
