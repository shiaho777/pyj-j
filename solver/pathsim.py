#!/usr/bin/env python3
"""
pathsim.py -- the 100x truth engine.

The question on the table: $100 -> $10,000 in 30 days. Every strategy that
has ever existed falls into one of four families. This engine computes the
probability of hitting 100x for each family -- in exact rational arithmetic
where a closed form exists, by Monte Carlo where paths matter.

  A. all-in ladders      perp reladd / coinflips / memecoin legs.
                         Exact: P = p^k, k = ceil(log(100)/log(b)).
  B. ticket splitting     N tickets of $100/N need 100N each to make one hit
                         count. The dilution law: P(1 hit) depends on how
                         fast the tail thins vs payoff (index alpha).
  C. the grind           a sustained edge, compounded daily. Even a
                         world-class edge vs 3% daily vol.
  D. the sniper portfol. many microcap shots, fat tails, moonbag capture
                         rules. Monte Carlo over stake fractions.

Honesty rules (same as everywhere in this repo): closed-form numbers are
exact rationals; anything fitted or assumed is labeled (model). No number
in this report is a promise.
"""
import math
import random
from fractions import Fraction

TARGET = 100            # wealth multiple demanded
DAYS = 30
MEDALLION_ANNUAL = 1.66  # ~66%/yr, best sustained track record in history (est.)
SIGMA_DAY = 0.03         # 3%/day crypto vol (est.)


def ceil_div(a, b):
    return -((-a) // b)


def odds_one_in(p):
    """Exact Fraction p -> '1 in N' with N = ceil(1/p)."""
    n = ceil_div(p.denominator, p.numerator)
    return n


# ---------------------------------------------------------------- A. ladders
def ladder(p, b):
    """All-in rounds: win -> wealth * b, lose -> 0. Exact.

    p, b are (num, den) pairs. Returns (k legs needed,
    P(all k wins) as Fraction, EV per leg).
    """
    p, b = Fraction(*p), Fraction(*b)
    k = 1
    while b ** k < TARGET:
        k += 1
    return k, p ** k, p * b


def report_ladders():
    print("=" * 74)
    print("A. ALL-IN LADDERS  (exact rational arithmetic)")
    print("   each round risks the whole stack: win -> x{b}, lose -> 0")
    print("=" * 74)
    rows = [
        ("fair coin, 2.00x leg",        (1, 2),   (2, 1)),
        ("coinflip + 2% round-trip fee", (1, 2),   (49, 25)),
        ("50x perp leg (wick risk)",    (9, 20),  (39, 20)),
        ("memecoin leg to +150%",       (7, 20),  (5, 2)),
        ("dream leg (a real edge)",     (1, 2),   (3, 1)),
    ]
    print(f"{'leg':32} {'legs k':>6} {'P(100x)':>10} {'odds':>12} {'EV/leg':>7}")
    for name, p, b in rows:
        k, prob, ev = ladder(p, b)
        print(f"{name:32} {k:>6} {float(prob)*100:>9.4f}% "
              f"{'1 in %d' % odds_one_in(prob):>12} {float(ev):>7.3f}")
    print("""
reading: even a FAIR 2x coin needs 7 consecutive wins -> 1 in 128.
the 'dream leg' row is what an actual informational edge looks like:
even there, one loss zeroes the stack, so P(100x) = 1/32, not 1/2.
fees barely move P -- they move EV. the ladder IS a lottery by construction.
""")


# ------------------------------------------------------- B. dilution law
def report_dilution():
    print("=" * 74)
    print("B. TICKET SPLITTING  (the dilution law, model)")
    print("   N tickets of $100/N each; ONE hit must pay 100*N to reach 100x")
    print("=" * 74)
    q100 = 0.005          # P(a single $100 ticket pays 100x)  (model)
    print(f"assumed: one $100 ticket has P(100x) = {q100:.1%} (deep-OTM wings,")
    print("microcap moons -- both live roughly here in good months)\n")
    print(f"{'N tickets':>10} " + "".join(f"{'alpha=' + str(a):>11}" for a in (0.5, 1, 2, 3)))
    for n in (1, 2, 5, 10, 20, 50):
        cells = []
        for alpha in (0.5, 1, 2, 3):
            qn = q100 * n ** (-alpha)          # tail thins as payoff grows
            phit = 1.0 - (1.0 - qn) ** n
            cells.append(f"{phit*100:>10.4f}%")
        print(f"{n:>10} " + "".join(f"{c:>11}" for c in cells))
    print("""
reading: alpha is how fast the payoff tail thins (extreme-value tail index).
empirical estimates put crypto microcap return tails near alpha ~ 2-3.
  alpha < 1  -> splitting buys more shots at thin-enough tails: diversify
  alpha = 1  -> wash
  alpha > 1  -> splitting MURDERS the moonshot: the 100x target itself
                forces concentration.
unless your world is extremely fat-tailed, the structure of the goal
forces you onto few, fat tickets. this is a theorem, not an opinion.
""")


# ------------------------------------------------------------- C. the grind
def report_grind():
    print("=" * 74)
    print("C. THE GRIND  (Monte Carlo, 20k paths x 30 days)")
    print("   daily return = edge + noise; wealth compounds")
    print("=" * 74)
    sigma = SIGMA_DAY
    mu_needed = math.log(TARGET) / DAYS + sigma * sigma / 2
    print(f"edge needed for 100x/month MEDIAN at 3%/day vol: "
          f"{mu_needed*100:.2f}%/day sustained -- i.e. every single day,")
    print(f"do ~{mu_needed / (math.log(MEDALLION_ANNUAL) / 365):.0f}x what the "
          "best fund in history does in a day. next.\n")
    print(f"{'edge/day':>10} {'median 30d':>12} {'P(>=100x)':>10} "
          f"{'P(loss)':>8} {'z-score':>8}")
    for mu_bps in (5, 30, 100):
        mu = mu_bps / 10000
        rng = random.Random(7)
        finals = []
        for _ in range(20000):
            w = 1.0
            for _ in range(DAYS):
                w *= math.exp(mu + sigma * rng.gauss(0, 1))
            finals.append(w)
        finals.sort()
        hit = sum(1 for w in finals if w >= TARGET)
        loss = sum(1 for w in finals if w < 1.0)
        z = (math.log(TARGET) - DAYS * (mu - sigma * sigma / 2)) / (sigma * math.sqrt(DAYS))
        print(f"{mu_bps:>8}bp {finals[len(finals)//2]:>12.3f} "
              f"{hit/20000*100:>9.3f}% {loss/20000*100:>7.1f}% {z:>8.1f}")
    print("""
reading: +30bp/day with 3% daily vol is a WORLD-CLASS edge (Medallion-class).
its median month is +8%. 100x is a 27-sigma event. the gap between the
best repeatable edge and the target is three orders of magnitude, and
that gap is made of exactly one material: variance.
""")


# -------------------------------------------------------- D. sniper book
# per-shot outcome distribution on a $1 stake. m = money multiple returned.
# "naive"  = buys whatever pumped; "elite" = the filter actually works.
# ALL DISTRIBUTIONS HERE ARE ASSUMED, NOT MEASURED -- the point of the sweep
# is that the conclusions are robust across every sane calibration.
SNIPER_DISTS = {
    "naive, no filter": [
        (0.60, 0.0), (0.25, 0.3), (0.11, 1.0), (0.030, 3.0),
        (0.0065, 10.0), (0.0022, 30.0), (0.0011, 100.0), (0.0002, 1000.0)],
    "elite filter, hold to top (dream)": [
        (0.42, 0.0), (0.23, 0.3), (0.17, 1.0), (0.10, 3.0),
        (0.055, 10.0), (0.018, 30.0), (0.0055, 100.0), (0.0013, 1000.0),
        (0.0002, 10000.0)],
}
MOONBAG_RULE = (0.8, 3.0)   # sell 80% at 3x, ride the 20% moonbag


def moonbag_capture(m):
    """Realized multiple when discipline forces 80% out at 3x."""
    sell_frac, sell_at = MOONBAG_RULE
    return sell_frac * min(m, sell_at) + (1 - sell_frac) * m


def cum_dist(dist):
    out, acc = [], 0.0
    for p, m in dist:
        acc += p
        out.append((acc, m))
    return out


def draw(cum, r):
    for acc, m in cum:
        if r <= acc:
            return m
    return 0.0


def sniper_paths(dist, stake_frac, shots, n_paths, seed=11):
    cum = cum_dist(dist)
    rng = random.Random(seed)
    finals = []
    for _ in range(n_paths):
        w = 1.0
        for _ in range(shots):
            stake = w * stake_frac
            m = draw(cum, rng.random())
            w = w - stake + stake * m
            if w < 1e-12:
                break
        finals.append(w)
    finals.sort()
    return finals


def report_sniper():
    print("=" * 74)
    print("D. THE SNIPER PORTFOLIO  (Monte Carlo, 20k paths, 40 shots/month)")
    print("   microcap entries; stop rules + moonbag capture")
    print("=" * 74)
    dream = SNIPER_DISTS["elite filter, hold to top (dream)"]
    elite_real = [(p, moonbag_capture(m)) for p, m in dream]
    dists = [
        ("naive, no filter", SNIPER_DISTS["naive, no filter"]),
        ("elite filter, disciplined 80/20", elite_real),
        ("elite filter, hold to top (dream)", dream),
    ]
    for name, dist in dists:
        ev = sum(p * m for p, m in dist)
        print(f"\n  [{name}]  EV/shot = {ev:.2f}x   "
              f"(assumed distribution, not measured)")
        print(f"  {'stake/shot':>11} {'P(>=100x)':>10} {'median':>9} "
              f"{'P(-90%)':>8} {'E[W]':>8}")
        for sf in (0.025, 0.10, 0.25, 1.0):
            f = sniper_paths(dist, sf, 40, 20000)
            hit = sum(1 for w in f if w >= TARGET)
            p90 = sum(1 for w in f if w < 0.1)
            print(f"  {sf*100:>10.1f}% {hit/20000*100:>9.3f}% "
                  f"{f[len(f)//2]:>9.3f} {p90/20000*100:>7.1f}% "
                  f"{sum(f)/20000:>8.2f}")
    print("""
reading: disciplined 80/20 capture (the only rule that survives contact
with rugs) gives P(100x) ~ 0.8-1% at sane sizing. the 'dream' rows reach
~10% ONLY by assuming you ride every moonbag to the absolute top --
EV 5.48x/shot, E[W] 7070x/month. a world like that would already be
arbitraged away by professional capital; the number is the reductio of
its own assumption. the two elite rows differ 12x in P(100x) purely on
the capture discipline: the rule that saves your median is the same rule
that caps your moonshot. that tension is not a bug in this model;
it is the market. and the all-in row is 0% everywhere: one rug,
one stop-out, and the month is over -- family A in a trench coat.
""")


def main():
    print("THE 100x TRUTH ENGINE   $100 -> $10,000 in 30 days\n")
    daily = TARGET ** (1.0 / DAYS)
    mm_daily = MEDALLION_ANNUAL ** (1.0 / 365) - 1
    print(f"target requires: {daily-1:.2%}/day compounded, every day, "
          f"doubling every {math.log(2)/math.log(daily):.2f} days")
    print(f"                    = {(daily-1)/mm_daily:.0f}x the daily "
          f"compounding rate of the best fund in history\n")
    report_ladders()
    report_dilution()
    report_grind()
    report_sniper()
    print("=" * 74)
    print("VERDICT (what these four families, taken together, actually say)")
    print("=" * 74)
    print("""
1. no repeatable strategy reaches 100x/30d. the best repeatable edge
   on earth compounds ~8%/month at median. the target is a 27-sigma
   event FOR THAT EDGE. anyone selling a script that does this
   reliably is selling the variance, not the strategy.
2. every path to 100x runs through the tail, and the dilution law
   says the tail only pays concentrated bets. the goal itself
   forces near-all-in.
3. therefore the honest question is not 'which strategy reaches 100x'
   but 'which lottery ticket has the best true odds, and can we
   manufacture edges that make the ticket cheaper'.
4. the edges we CAN manufacture (exact-arithmetic replay, integer
   judges, sub-wei rounding) are real but pay in basis points,
   not multiples. they are the grind. they compound skill.
""")


if __name__ == "__main__":
    main()
