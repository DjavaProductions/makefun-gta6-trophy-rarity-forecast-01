"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (5%, 25%, 50%, 75%,
95%), scored by pinball loss on the logit scale.

Method, per trophy
------------------
1. Trend: a recency-weighted least-squares line of logit(rate) on
   ln(1 + hours since launch 2026-11-19T00:00:00Z), over every read in the
   case's history. The slope is shrunk toward the cross-trophy median slope,
   so a trophy with a noisy or flat series borrows strength from the rest.
2. Refresh: Exophase refreshes its counts on its own schedule (hours apart),
   not continuously. Between refreshes the page is frozen; at a refresh the
   tracked-player count jumps and rates fall, because the new players have
   earned far less than the early cohort. The program detects refresh events
   in the history (a read where the players-tracked count or any rate
   changed). When the history shows no refresh yet and the page has been
   stale a long time, the forecast is a two-component mixture in logit space:
   no refresh before the target (the trend) versus a refresh (the trend minus
   a dilution drop). When refreshes are already visible in the history, the
   trend already runs through post-refresh levels, so no extra level shift
   is applied. Quantiles come from inverting that mixture numerically.
3. Spread: the residual scatter of the trend fit, the slope's standard error
   times the extrapolation distance, and the refresh mixture's width, with a
   small floor so the forecast never claims impossible precision.

The program is deterministic: no randomness anywhere, no network, no model
calls, standard library only. Two runs on the same case write identical bytes.

Usage: python3 main.py (reads $MAKEFUN_INPUT_DIR or /input, writes
$MAKEFUN_OUTPUT_DIR or /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch of GTA VI, the origin of the time axis (UTC).
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- priors, all fixed in advance, never tuned on the practice answers ---
# Expected gap between Exophase refreshes when the history shows none yet.
# Launch night's first refresh landed ~4.5 h after the first probe.
GAP_PRIOR_HOURS = 4.5
# Growth factor of tracked players at a launch-era refresh (638 -> 3868).
K_PRIOR = 5.0
# Recency half-life for the trend fit, in hours.
HALF_LIFE_HOURS = 24.0
# Prior strength (in effective reads) when shrinking a slope to the median.
SHRINK_K = 3.0
# Floor on the predictive spread, in logit units.
SPREAD_FLOOR = 0.15
# A trophy at or above this rate is treated as saturated (stays ~100%).
SATURATED = 99.9


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l: float) -> float:
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(t: datetime) -> float:
    return (t - LAUNCH).total_seconds() / 3600.0


def wls(xs, ys, ws):
    """Weighted least squares for y = a + b*x.

    Returns (a, b, resid_std, se_b, x_weighted_mean). All inputs are lists of
    equal length; weights are positive.
    """
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    if sxx <= 1e-12:
        b = 0.0
    else:
        b = sxy / sxx
    a = my - b * mx
    n = len(xs)
    if n > 2 and sxx > 1e-12:
        sse = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        # Effective sample size for weighted residuals.
        n_eff = sw * sw / sum(w * w for w in ws)
        # Weighted mean squared residual, corrected for the 2 fitted params.
        resid_var = (sse / sw) * n_eff / max(1.0, n_eff - 2.0)
        resid_std = math.sqrt(max(0.0, resid_var))
        se_b = resid_std / math.sqrt(sxx) if sxx > 0 else 0.0
    else:
        resid_std = 0.0
        se_b = 0.0
    return a, b, resid_std, se_b, mx


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mixture_quantiles(components, levels):
    """Quantiles of a mixture of normals, by numerical inversion.

    components: list of (weight, mean, std) in logit space, weights sum to 1.
    Returns the quantile at each level, as a list in the same order.
    """
    if len(components) == 1:
        w, m, s = components[0]
        return [m + Z[i] * s for i in range(len(levels))]
    lo = min(m - 8.0 * max(s, 1e-6) for _, m, s in components)
    hi = max(m + 8.0 * max(s, 1e-6) for _, m, s in components)
    # CDF on a fine grid, then invert by linear interpolation.
    n_grid = 2001
    step = (hi - lo) / (n_grid - 1)
    grid = [lo + i * step for i in range(n_grid)]
    cdf = []
    for g in grid:
        total = 0.0
        for w, m, s in components:
            total += w * norm_cdf((g - m) / max(s, 1e-6))
        cdf.append(total)
    out = []
    for tau in levels:
        # Find the first grid point whose CDF reaches tau.
        j = 0
        while j < n_grid - 1 and cdf[j] < tau:
            j += 1
        if j == 0:
            out.append(grid[0])
        elif cdf[j] >= tau and cdf[j - 1] < tau and cdf[j] > cdf[j - 1]:
            frac = (tau - cdf[j - 1]) / (cdf[j] - cdf[j - 1])
            out.append(grid[j - 1] + frac * step)
        else:
            out.append(grid[j])
    return out


def detect_refreshes(history):
    """Find global refresh events in the history.

    A refresh is a read where the players-tracked count changed or any
    trophy's rate changed versus the previous read. Returns a list of
    (index, players_before, players_after) and the list of read times.
    """
    events = []
    for i in range(1, len(history)):
        prev, cur = history[i - 1], history[i]
        players_changed = cur.get("players_tracked") != prev.get("players_tracked")
        rates_changed = False
        pr, cr = prev.get("rates") or [], cur.get("rates") or []
        for a, b in zip(pr, cr):
            # A null is a missing read, not a change: only a difference
            # between two present rates counts.
            if a is not None and b is not None and a != b:
                rates_changed = True
                break
        if players_changed or rates_changed:
            events.append((i, prev.get("players_tracked"), cur.get("players_tracked")))
    return events


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the contest's five")
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    if not trophies:
        raise ValueError("case has no trophies")

    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_hours = (target - at).total_seconds() / 3600.0
    horizon_days = case["horizon_minutes"] / 1440.0

    last = history[-1]
    last_rates = last["rates"]
    last_time = parse_time(last["at"])

    # --- global refresh picture ---
    refresh_events = detect_refreshes(history)
    n_refresh = len(refresh_events)
    # Median gap between refreshes, in hours; the prior when none are seen.
    if n_refresh >= 2:
        gaps = []
        prev_t = parse_time(history[refresh_events[0][0]]["at"])
        for idx, _, _ in refresh_events[1:]:
            t = parse_time(history[idx]["at"])
            gaps.append((t - prev_t).total_seconds() / 3600.0)
            prev_t = t
        gap = median(gaps) if gaps else GAP_PRIOR_HOURS
        gap = min(max(gap, 1.0), 48.0)
    else:
        gap = GAP_PRIOR_HOURS
    # Hours since the last observed change (or since the first read, when the
    # page never changed in the window we can see).
    if refresh_events:
        last_change = parse_time(history[refresh_events[-1][0]]["at"])
    else:
        last_change = parse_time(history[0]["at"])
    staleness = max(0.0, (at - last_change).total_seconds() / 3600.0)
    # Probability at least one refresh lands between the last change and the
    # target, treating refreshes as roughly Poisson with rate 1/gap.
    p_refresh = 1.0 - math.exp(-(horizon_hours + staleness) / gap)
    p_refresh = min(max(p_refresh, 0.0), 0.95)

    # Effective time of each read: the page a read shows is the page as of
    # the last refresh at or before it, so a read's rate belongs to its last
    # refresh's time, not to the read's own clock time. Fitting rates
    # against read times would steepen the slope (early reads lag the curve
    # most), so the trend is fit on effective times. Reads before the first
    # refresh seen take the first read's time.
    change_times = [parse_time(history[idx]["at"]) for idx, _, _ in refresh_events]
    first_time = parse_time(history[0]["at"])
    eff_times = []
    ci = 0
    for read in history:
        t = parse_time(read["at"])
        while ci < len(change_times) and change_times[ci] <= t:
            ci += 1
        eff_times.append(change_times[ci - 1] if ci > 0 else first_time)
    # The answer is the rate as of the last refresh before the target, which
    # lands about half a gap before it on average: read the trend there.
    x_target = math.log1p(max(0.0, hours_since_launch(target) - gap / 2.0))

    # Growth factor of tracked players at a refresh, from the jumps seen.
    # With no jump seen, fall back to the launch-era prior.
    jumps = []
    for _, before, after in refresh_events:
        if before and after and before > 0 and after > before:
            jumps.append(after / before - 1.0)
    k_refresh = median(jumps) if jumps else K_PRIOR
    k_refresh = min(max(k_refresh, 0.0), 20.0)
    drop_scale = math.log1p(k_refresh)  # logit drop if new players earned nothing

    # --- per-trophy trend fits (first pass: collect slopes) ---
    fits = []
    for ti, trophy in enumerate(trophies):
        xs, ys, ws = [], [], []
        for read, eff_t in zip(history, eff_times):
            rate = read["rates"][ti]
            if rate is None:
                continue
            xs.append(math.log1p(max(0.0, hours_since_launch(eff_t))))
            ys.append(logit(rate))
            t = parse_time(read["at"])
            age_h = max(0.0, (last_time - t).total_seconds() / 3600.0)
            ws.append(0.5 ** (age_h / HALF_LIFE_HOURS))
        if len(xs) >= 2:
            a, b, resid, se_b, mx = wls(xs, ys, ws)
        elif len(xs) == 1:
            a, b, resid, se_b, mx = ys[0], 0.0, 0.0, 0.0, xs[0]
        else:
            a, b, resid, se_b, mx = logit(last_rates[ti] if last_rates[ti] is not None else 1.0), 0.0, 0.0, 0.0, 0.0
        fits.append({"a": a, "b": b, "resid": resid, "se_b": se_b,
                     "mx": mx, "n": len(xs)})

    good_slopes = [f["b"] for f in fits if f["n"] >= 3]
    median_slope = median(good_slopes) if good_slopes else 0.0
    good_resid = [f["resid"] for f in fits if f["n"] >= 3 and f["resid"] > 0]
    median_resid = median(good_resid) if good_resid else 0.0

    rows = []
    for ti, trophy in enumerate(trophies):
        r_last = last_rates[ti]
        if r_last is None:
            # No read ever listed this trophy: borrow the median last rate.
            seen = [lr for lr in last_rates if lr is not None]
            r_last = median(seen) if seen else 1.0
        # A saturated trophy stays saturated: everyone already has it.
        if r_last >= SATURATED:
            rows.append({"id": trophy["id"],
                         "q": [round4(v) for v in (99.5, 99.8, 99.95, 99.99, 99.99)]})
            continue

        f = fits[ti]
        # Shrink the slope toward the cross-trophy median; the weight grows
        # with the number of reads behind this trophy's fit.
        w_self = f["n"] / (f["n"] + SHRINK_K) if f["n"] > 0 else 0.0
        b = w_self * f["b"] + (1.0 - w_self) * median_slope
        # Keep the fitted level anchored: re-solve the intercept at the
        # weighted mean x so shrinkage moves the slope, not the level.
        a = f["a"] + (f["b"] - b) * f["mx"] if f["n"] >= 2 else f["a"]
        m_logit = a + b * x_target

        # Predictive spread in logit units.
        resid = f["resid"] if f["resid"] > 0 else median_resid
        extrap = abs(x_target - f["mx"])
        var = resid * resid + (f["se_b"] * extrap) ** 2
        # The random-walk floor: the baseline's spread for this horizon.
        floor_var = (0.5 * math.sqrt(horizon_days)) ** 2
        spread = math.sqrt(var + 0.25 * floor_var)
        spread = max(spread, SPREAD_FLOOR)

        # --- refresh mixture ---
        # When the history already shows refreshes, the trend runs through
        # post-refresh levels: a future refresh is a timing lag, not a level
        # shift, so the drop is zero. When the history is flat, the trend is
        # the pre-refresh level and a refresh before the target dilutes the
        # rates: the drop is the log growth factor, softened for trophies
        # new players catch up on (higher current rate -> more catch-up).
        if n_refresh == 0:
            catch_up = math.sqrt(clamp(r_last) / 100.0)
            drop = drop_scale * (1.0 - catch_up)
            drop = min(max(drop, 0.0), 3.0)
        else:
            drop = 0.0
        if drop > 0.05 and p_refresh > 0.02:
            components = [(1.0 - p_refresh, m_logit, spread),
                          (p_refresh, m_logit - drop, spread)]
            q_logits = mixture_quantiles(components, LEVELS)
        else:
            q_logits = [m_logit + Z[i] * spread for i in range(len(LEVELS))]

        qs = []
        prev_q = None
        for ql in q_logits:
            q = clamp(inv_logit(ql))
            if prev_q is not None and q < prev_q:
                q = prev_q
            qs.append(round4(q))
            prev_q = q
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    if not inputs:
        raise SystemExit(f"no case files in {INPUT_DIR}")
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(json.dumps(forecast, separators=(",", ":")),
                            encoding="utf-8")


if __name__ == "__main__":
    main()
