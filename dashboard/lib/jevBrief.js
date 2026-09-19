/**
 * Pure functions for Jev regime pills — NO fetch, no side effects.
 *
 * Contains: PILLS, JEV_QUESTIONS, buildState, ruleVerdicts, conflictPairs,
 * mergeVerdicts, diffSinceYesterday, toData, pillFactors.
 *
 * Every function accepts possibly-null/undefined values and returns a safe
 * shape with no thrown exceptions.
 */

// ---------------------------------------------------------------------------
// Pills and verdict order (mild → severe)
// ---------------------------------------------------------------------------

export const PILLS = ['regime', 'recession', 'breadth', 'hedging', 'conflict'];

export const VERDICT_ORDER = {
    regime: ['risk-on', 'neutral', 'risk-off'],
    recession: ['low', 'rising', 'high'],
    breadth: ['broad', 'narrow', 'rolling-over'],
    hedging: ['cheap', 'fair', 'expensive'],
    conflict: ['aligned', 'mild-divergence', 'major-divergence'],
};

// ---------------------------------------------------------------------------
// Jev questions (one per pill)
// ---------------------------------------------------------------------------

export const JEV_QUESTIONS = {
    regime: {
        type: 'choice',
        instructions: 'Assess the overall market regime based on the state text. Consider trend strength, volatility, and credit conditions.',
        criteria: {
            'risk-on': 'Markets are in a clear risk-on regime — equities trending strongly above moving averages, volatility low, credit spreads tight, and sentiment optimistic but not extreme.',
            'neutral': 'Markets are in a neutral or mixed regime — no strong directional signal. Some indicators are positive, others negative or mixed.',
            'risk-off': 'Markets are in a risk-off regime — equities weak or rolling over, elevated volatility, widening credit spreads, and defensive positioning dominates.',
        },
    },
    recession: {
        type: 'choice',
        instructions: 'Assess the probability or severity of a recession based on the economic indicators provided.',
        criteria: {
            low: 'Recession risk is low. The yield curve is positively sloped, the Sahm rule is below 0.2, claims are low, and the NFCI is negative or near zero.',
            rising: 'Recession risk is rising. One or more leading indicators are flashing caution — yield curve flattening or inverted, Sahm rule approaching 0.2, claims ticking up, or financial conditions tightening.',
            high: 'Recession risk is high. Multiple indicators are flashing red — Sahm rule at 0.5 or above, persistent yield curve inversion with elevated claims, or significantly tight financial conditions.',
        },
    },
    breadth: {
        type: 'choice',
        instructions: 'Assess the breadth of equity market participation. Consider whether the rally or sell-off is broad or concentrated.',
        criteria: {
            broad: 'Market breadth is healthy. Both equal-weight (RSP) and small-cap (IWM) are participating positively relative to the S&P 500 over the last 20 trading days.',
            narrow: 'Market breadth is narrow or mixed. Participation is not uniformly positive — some segments are lagging or the data is insufficient to call broad participation.',
            'rolling-over': 'Market breadth is deteriorating. Equal-weight and small-cap are rolling over, and the RSP/SPY ratio is declining relative to its 50-day average, suggesting a broadening sell-off.',
        },
    },
    hedging: {
        type: 'choice',
        instructions: 'Assess the cost or expensiveness of portfolio hedging based on implied volatility levels and the volatility risk premium.',
        criteria: {
            cheap: 'Hedging is cheap. Implied volatility percentile is low (below 20th percentile over the last year) and the volatility risk premium is modest (below 6).',
            fair: 'Hedging is fairly priced. Implied volatility is in the middle of its one-year range and the volatility risk premium is moderate.',
            expensive: 'Hedging is expensive. Implied volatility is elevated (above 70th percentile) or the volatility risk premium is wide (above 10), meaning option buyers are paying a high premium.',
        },
    },
    conflict: {
        type: 'choice',
        instructions: 'Assess the degree of divergence or conflict between different market signals and asset classes.',
        criteria: {
            aligned: 'Major market signals are aligned. No significant divergences detected between sentiment and price, yield curves, credit and equities, or breadth and the index.',
            'mild-divergence': 'One notable divergence is present between market signals. This warrants attention but does not yet signal a regime change.',
            'major-divergence': 'Two or more significant divergences are present between market signals. This level of internal conflict often precedes or accompanies regime shifts.',
        },
    },
};

// ---------------------------------------------------------------------------
// buildState — compact plain text from the data contract object
// ---------------------------------------------------------------------------

// Round BEFORE choosing the sign so -0.012 prints "0.0%", never "-0.0%".
const fmtPct = (n) => {
    if (!Number.isFinite(n)) return 'n/a';
    const r = Math.round(n * 10) / 10 || 0;
    return `${r > 0 ? '+' : ''}${r.toFixed(1)}%`;
};
const fmtNum = (n) => (Number.isFinite(n) ? n.toFixed(2) : 'n/a');
const fmtInt = (n) => (Number.isFinite(n) ? String(Math.round(n)) : 'n/a');

/**
 * Safely coerce a value to a finite number or null.
 * Strips % and commas when the value is a string; returns null for
 * 'N/A', NaN, undefined, or any other non-finite input.
 */
const num = (v) => {
    if (Number.isFinite(v)) return v;
    if (typeof v === 'string') {
        const parsed = parseFloat(v.replace(/%/g, '').replace(/,/g, ''));
        return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
};

export function buildState(data) {
    if (!data) return '';
    const lines = [];

    // 1. SPY
    const s = data.spy || {};
    lines.push(
        `1. SPY ${fmtNum(s.price)}, ${fmtPct(s.ma200Pct)} vs 200-day avg, ${fmtPct(s.high52Pct)} from 52-week high, RSI ${fmtNum(s.rsi)}, today ${fmtPct(s.chgPct)}`
    );

    // 2. Fear & Greed
    const fg = data.fg || {};
    const fgRating = fg.rating || 'n/a';
    const fgPrevWeek = fg.prevWeek != null ? String(fg.prevWeek) : 'n/a';
    const fgPrevMonth = fg.prevMonth != null ? String(fg.prevMonth) : 'n/a';
    lines.push(
        `2. Fear & Greed ${fmtInt(fg.score)} (${fgRating}), previous week ${fgPrevWeek}, previous month ${fgPrevMonth}`
    );

    // 3. Volatility (SPY IV + VRP)
    const volSpy = (data.vol || {}).spy || {};
    lines.push(
        `3. SPY IV ${fmtNum(volSpy.iv)}, IV rank ${fmtNum(volSpy.ivRank1y)}, IV percentile ${fmtNum(volSpy.ivPctile1y)}, realized vol ${fmtNum(volSpy.rv21)}, VRP ${fmtNum(volSpy.vrp)}`
    );

    // 4. Economic / FRED
    const fred = data.fred || {};
    lines.push(
        `4. Yield curve ${fmtNum(fred.yieldCurve)}, Sahm rule ${fmtNum(fred.sahmRule)}, jobless claims ${fmtNum(fred.claims)}k, BBB credit spread ${fmtNum(fred.creditSpread)}, real yields ${fmtNum(fred.realYields)}, copper/gold ${fmtNum(fred.copperGold)}, sentiment ${fmtInt(fred.sentiment)}, NFCI ${fmtNum(fred.nfci)}`
    );

    // 5. T10Y3M
    lines.push(`5. 10Y-3M spread ${fmtNum(data.t10y3m)}`);

    // 6. Breadth
    const breadth = data.breadth || {};
    const rsp = breadth.rspSpy || {};
    const iwm = breadth.iwmSpy || {};
    const xlk = breadth.xlkXlu || {};
    const hyg = breadth.hygLqd || {};
    lines.push(
        `6. RSP/SPY ${fmtNum(rsp.ratio)}, 20d ${fmtPct(rsp.chg20Pct)}, 60d ${fmtPct(rsp.chg60Pct)}, vs 50d avg ${fmtPct(rsp.vs50dPct)}`
    );
    lines.push(
        `7. IWM/SPY ${fmtNum(iwm.ratio)}, 20d ${fmtPct(iwm.chg20Pct)}, 60d ${fmtPct(iwm.chg60Pct)}`
    );
    lines.push(
        `8. XLK/XLU ${fmtNum(xlk.ratio)}, 20d ${fmtPct(xlk.chg20Pct)}`
    );
    lines.push(
        `9. HYG/LQD ${fmtNum(hyg.ratio)}, 20d ${fmtPct(hyg.chg20Pct)}`
    );

    // 10. AAII
    lines.push(`10. AAII bull-bear spread ${fmtPct(data.aaiiDiff)}`);

    return lines.join('\n');
}

// ---------------------------------------------------------------------------
// conflictPairs — detect divergences
// ---------------------------------------------------------------------------

export function conflictPairs(data) {
    const pairs = [];
    if (!data) return pairs;

    const s = data.spy || {};
    const fg = data.fg || {};
    const fred = data.fred || {};
    const breadth = data.breadth || {};
    const rsp = breadth.rspSpy || {};
    const hyg = breadth.hygLqd || {};

    const ma200Pct = s.ma200Pct;
    const fgScore = fg.score;
    const high52Pct = s.high52Pct;
    const yieldCurve = fred.yieldCurve;
    const t10y3m = data.t10y3m;

    // sentiment vs price
    if (fgScore != null && ma200Pct != null) {
        if ((fgScore < 35 && ma200Pct > 3) || (fgScore > 65 && ma200Pct < -3)) {
            const detail =
                fgScore < 35
                    ? `Fear (${fgScore}) despite strong trend (+${ma200Pct.toFixed(1)}% above 200d MA)`
                    : `Greed (${fgScore}) despite weak trend (${ma200Pct.toFixed(1)}% below 200d MA)`;
            pairs.push({ pair: 'sentiment vs price', detail });
        }
    }

    // 2s10s vs 3m10y
    if (yieldCurve != null && t10y3m != null) {
        if ((yieldCurve > 0 && t10y3m < 0) || (yieldCurve < 0 && t10y3m > 0)) {
            pairs.push({
                pair: '2s10s vs 3m10y',
                detail: `2s10s=${yieldCurve.toFixed(2)}%, 3m10y=${t10y3m.toFixed(2)}% — opposite signs`,
            });
        }
    }

    // credit vs equities
    if (ma200Pct != null && hyg.chg20Pct != null) {
        if (ma200Pct > 0 && hyg.chg20Pct < -1.5) {
            pairs.push({
                pair: 'credit vs equities',
                detail: `SPY ${ma200Pct.toFixed(1)}% above 200d MA yet HYG/LQD -${Math.abs(hyg.chg20Pct).toFixed(1)}% over 20d`,
            });
        }
    }

    // breadth vs index
    if (high52Pct != null && rsp.chg20Pct != null) {
        if (high52Pct > -2 && rsp.chg20Pct < -1.5) {
            pairs.push({
                pair: 'breadth vs index',
                detail: `SPY within ${high52Pct.toFixed(1)}% of 52w high yet RSP/SPY ${rsp.chg20Pct.toFixed(1)}% over 20d`,
            });
        }
    }

    return pairs;
}

// ---------------------------------------------------------------------------
// ruleVerdicts
// ---------------------------------------------------------------------------

const safeNum = (v) => (v != null && Number.isFinite(v) ? v : null);
const zeroIfMissing = (v) => (v != null && Number.isFinite(v) ? (v > 0 ? 1 : -1) : 0);
const ternaryMissing = (v, ifPos, ifNeg) => (v != null && Number.isFinite(v) ? (v > 0 ? ifPos : ifNeg) : 0);

export function ruleVerdicts(data) {
    if (!data) {
        const r = {};
        for (const pill of PILLS) r[pill] = { verdict: 'n/a', reason: 'no data' };
        return r;
    }

    const s = data.spy || {};
    const fg = data.fg || {};
    const fred = data.fred || {};
    const brd = data.breadth || {};
    const rsp = brd.rspSpy || {};
    const iwm = brd.iwmSpy || {};
    const hyg = brd.hygLqd || {};
    const volSpy = (data.vol || {}).spy || {};

    const ma200Pct = safeNum(s.ma200Pct);
    const fgScore = safeNum(fg.score);
    const hygChg20 = safeNum(hyg.chg20Pct);
    const yieldCurve = safeNum(fred.yieldCurve);
    const sahmRule = safeNum(fred.sahmRule);
    const claims = safeNum(fred.claims);
    const nfci = safeNum(fred.nfci);
    const rspChg20 = safeNum(rsp.chg20Pct);
    const rspVs50d = safeNum(rsp.vs50dPct);
    const iwmChg20 = safeNum(iwm.chg20Pct);
    const ivPctile = safeNum(volSpy.ivPctile1y);
    const vrp = safeNum(volSpy.vrp);
    const high52Pct = safeNum(s.high52Pct);

    // ---- regime ----
    let regimeScore = 0;
    const regimeParts = [];

    // ma200Pct contribution
    if (ma200Pct != null) {
        regimeScore += ma200Pct > 0 ? 1 : -1;
        regimeParts.push(`ma200=${ma200Pct > 0 ? '+' : ''}${ma200Pct.toFixed(1)}% (${ma200Pct > 0 ? '+1' : '-1'})`);
    } else {
        regimeParts.push('ma200=mv (0)');
    }

    // fear/greed contribution
    if (fgScore != null) {
        if (fgScore >= 50) {
            regimeScore += 1;
            regimeParts.push(`fg=${fgScore} (>=50, +1)`);
        } else if (fgScore < 30) {
            regimeScore -= 1;
            regimeParts.push(`fg=${fgScore} (<30, -1)`);
        } else {
            regimeParts.push(`fg=${fgScore} (30-49, 0)`);
        }
    } else {
        regimeParts.push('fg=mv (0)');
    }

    // HYG/LQD contribution
    if (hygChg20 != null) {
        if (hygChg20 > 0) {
            regimeScore += 1;
            regimeParts.push(`hygLqd=${fmtPct(hygChg20)} (>0, +1)`);
        } else if (hygChg20 < -1) {
            regimeScore -= 1;
            regimeParts.push(`hygLqd=${fmtPct(hygChg20)} (<-1, -1)`);
        } else {
            regimeParts.push(`hygLqd=${fmtPct(hygChg20)} (-1 to 0, 0)`);
        }
    } else {
        regimeParts.push('hygLqd=mv (0)');
    }

    let regimeVerdict;
    if (regimeScore >= 2) regimeVerdict = 'risk-on';
    else if (regimeScore <= -1) regimeVerdict = 'risk-off';
    else regimeVerdict = 'neutral';

    const regime = {
        verdict: regimeVerdict,
        reason: `Score ${regimeScore}: ${regimeParts.join(', ')}`,
    };

    // ---- recession ----
    let recessionVerdict;
    const recessionReasons = [];

    if (sahmRule != null && sahmRule >= 0.5) {
        recessionVerdict = 'high';
        recessionReasons.push(`Sahm rule ${sahmRule.toFixed(2)} >= 0.5`);
    } else if (yieldCurve != null && yieldCurve < 0 && claims != null && claims >= 260) {
        recessionVerdict = 'high';
        recessionReasons.push(`yield curve ${yieldCurve.toFixed(2)}% < 0 and claims ${claims.toFixed(0)}k >= 260k`);
    } else if (sahmRule != null && sahmRule >= 0.2) {
        recessionVerdict = 'rising';
        recessionReasons.push(`Sahm rule ${sahmRule.toFixed(2)} >= 0.2`);
    } else if (yieldCurve != null && yieldCurve < 0) {
        recessionVerdict = 'rising';
        recessionReasons.push(`yield curve ${yieldCurve.toFixed(2)}% < 0`);
    } else if (claims != null && claims >= 260) {
        recessionVerdict = 'rising';
        recessionReasons.push(`claims ${claims.toFixed(0)}k >= 260k`);
    } else if (nfci != null && nfci > 0) {
        recessionVerdict = 'rising';
        recessionReasons.push(`NFCI ${nfci.toFixed(2)} > 0`);
    } else {
        recessionVerdict = 'low';
        recessionReasons.push('no recession signals triggered');
    }

    const recession = {
        verdict: recessionVerdict,
        reason: recessionReasons.join('; '),
    };

    // ---- breadth ----
    let breadthVerdict;
    const breadthReasons = [];

    if (rspChg20 == null && iwmChg20 == null) {
        breadthVerdict = 'narrow';
        breadthReasons.push('no breadth data');
    } else {
        if (rspChg20 != null && rspVs50d != null && rspChg20 < -1.5 && rspVs50d < 0) {
            breadthVerdict = 'rolling-over';
            breadthReasons.push(`RSP/SPY 20d ${rspChg20.toFixed(1)}% (< -1.5) and vs 50d avg ${rspVs50d.toFixed(1)}% (< 0)`);
        } else if (rspChg20 != null && rspChg20 > 0 && iwmChg20 != null && iwmChg20 > 0) {
            breadthVerdict = 'broad';
            breadthReasons.push(`RSP/SPY 20d ${rspChg20.toFixed(1)}% > 0 and IWM/SPY 20d ${iwmChg20.toFixed(1)}% > 0`);
        } else {
            breadthVerdict = 'narrow';
            const parts = [];
            if (rspChg20 != null) parts.push(`RSP/SPY 20d ${rspChg20.toFixed(1)}%`);
            if (iwmChg20 != null) parts.push(`IWM/SPY 20d ${iwmChg20.toFixed(1)}%`);
            breadthReasons.push(parts.join(', ') + ' — not meeting broad or rolling-over thresholds');
        }
    }

    const breadth = {
        verdict: breadthVerdict,
        reason: breadthReasons.join('; '),
    };

    // ---- hedging ----
    let hedgingVerdict;
    const hedgingReasons = [];

    if (ivPctile == null && vrp == null) {
        hedgingVerdict = 'fair';
        hedgingReasons.push('no vol data (fair by default)');
    } else {
        const cheap = ivPctile != null && ivPctile < 20 && vrp != null && vrp < 6;
        const expensive = (ivPctile != null && ivPctile > 70) || (vrp != null && vrp > 10);

        if (expensive) {
            hedgingVerdict = 'expensive';
            const parts = [];
            if (ivPctile != null && ivPctile > 70) parts.push(`IV percentile ${ivPctile.toFixed(0)} > 70`);
            if (vrp != null && vrp > 10) parts.push(`VRP ${vrp.toFixed(1)} > 10`);
            hedgingReasons.push(parts.join('; '));
        } else if (cheap) {
            hedgingVerdict = 'cheap';
            hedgingReasons.push(`IV percentile ${ivPctile.toFixed(0)} < 20 and VRP ${vrp.toFixed(1)} < 6`);
        } else {
            hedgingVerdict = 'fair';
            hedgingReasons.push(`IV percentile ${ivPctile != null ? ivPctile.toFixed(0) : 'n/a'}, VRP ${vrp != null ? vrp.toFixed(1) : 'n/a'} — middle range`);
        }
    }

    const hedging = {
        verdict: hedgingVerdict,
        reason: hedgingReasons.join('; '),
    };

    // ---- conflict ----
    const cps = conflictPairs(data);
    let conflictVerdict;
    if (cps.length === 0) conflictVerdict = 'aligned';
    else if (cps.length === 1) conflictVerdict = 'mild-divergence';
    else conflictVerdict = 'major-divergence';

    const conflict = {
        verdict: conflictVerdict,
        reason: cps.length === 0
            ? 'no divergences detected'
            : cps.map((cp) => `${cp.pair}: ${cp.detail}`).join('; '),
    };

    return { regime, recession, breadth, hedging, conflict };
}

// ---------------------------------------------------------------------------
// mergeVerdicts
// ---------------------------------------------------------------------------

export function mergeVerdicts(rule, jev, floor = 0.6) {
    const merged = {};
    if (!rule) return merged;

    for (const pill of PILLS) {
        const rv = rule[pill];
        if (!rv || !rv.verdict) {
            merged[pill] = { verdict: 'n/a', p: null, by: 'rule', reason: 'no rule verdict' };
            continue;
        }

        const jv = jev ? jev[pill] : null;
        const allowedVerdicts = VERDICT_ORDER[pill] || [];

        if (jv && allowedVerdicts.includes(jv.verdict) && jv.p != null && jv.p >= floor) {
            merged[pill] = {
                verdict: jv.verdict,
                p: jv.p,
                by: 'jev',
                reason: rv.reason || '',
            };
        } else {
            merged[pill] = {
                verdict: rv.verdict,
                p: null,
                by: 'rule',
                reason: rv.reason || '',
            };
        }
    }

    return merged;
}

// ---------------------------------------------------------------------------
// diffSinceYesterday
// ---------------------------------------------------------------------------

const severityOf = (pill, verdict) => {
    const order = VERDICT_ORDER[pill];
    if (!order) return -1;
    return order.indexOf(verdict);
};

export function diffSinceYesterday(today, yesterday) {
    if (!yesterday) {
        return { direction: 'none', changed: [], noBaseline: true };
    }

    const changed = [];
    for (const pill of PILLS) {
        const tV = today && today[pill] ? today[pill].verdict : undefined;
        const yV = yesterday[pill] ? yesterday[pill].verdict : undefined;
        if (tV != null && yV != null && tV !== yV) {
            changed.push({ pill, from: yV, to: tV });
        }
    }

    if (changed.length === 0) {
        return { direction: 'none', changed: [], noBaseline: false };
    }

    let allTowardSevere = true;
    let allTowardMild = true;

    for (const c of changed) {
        const sevFrom = severityOf(c.pill, c.from);
        const sevTo = severityOf(c.pill, c.to);
        if (sevTo > sevFrom) allTowardMild = false;
        if (sevTo < sevFrom) allTowardSevere = false;
    }

    let direction;
    if (allTowardSevere) direction = 'hardening';
    else if (allTowardMild) direction = 'softening';
    else direction = 'mixed';

    return { direction, changed, noBaseline: false };
}

// ---------------------------------------------------------------------------
// toData — normalise raw route payloads into the data contract
// ---------------------------------------------------------------------------

export function toData(raw) {
    if (!raw) return null;

    // /api/spy — wrap every numeric field in num() to handle 'N/A' strings
    const spyRaw = raw.spy || {};
    const spy = {
        price: num(spyRaw.current ?? spyRaw.price ?? null),
        chgPct: num(spyRaw.dailyChange?.pct ?? spyRaw.chgPct ?? null),
        ma200Pct: num(spyRaw.ma200?.pct ?? spyRaw.ma200Pct ?? null),
        high52Pct: num(spyRaw.week52High?.pct ?? spyRaw.high52Pct ?? null),
        rsi: num(spyRaw.rsi ?? null),
    };

    // /api/fear-greed
    const fgRaw = raw.fg || {};
    const fg = {
        score: num(fgRaw.score ?? null),
        rating: fgRaw.rating ?? null,
        prevWeek: num(fgRaw.previousWeek ?? fgRaw.prevWeek ?? null),
        prevMonth: num(fgRaw.previousMonth ?? fgRaw.prevMonth ?? null),
    };

    // /api/vol — tickers arrive in UPPERCASE, data contract uses lowercase keys
    const volRaw = raw.vol || {};
    const volTickers = volRaw.tickers || [];
    const vol = {};
    for (const t of volTickers) {
        vol[String(t.ticker).toLowerCase()] = {
            iv: num(t.iv ?? null),
            ivRank1y: num(t.ivRank1y ?? null),
            ivPctile1y: num(t.ivPctile1y ?? null),
            rv21: num(t.rv21 ?? null),
            vrp: num(t.vrp ?? null),
        };
    }

    // /api/fred
    const fredRaw = raw.fred || {};
    const fred = {
        yieldCurve: num(fredRaw.yieldCurve?.value ?? fredRaw.yieldCurve ?? null),
        sahmRule: num(fredRaw.indicators?.sahmRule?.value ?? fredRaw.sahmRule ?? null),
        claims: num(fredRaw.indicators?.claims?.value ?? fredRaw.claims ?? null),
        creditSpread: num(fredRaw.indicators?.creditSpread?.value ?? fredRaw.creditSpread ?? null),
        realYields: num(fredRaw.indicators?.realYields?.value ?? fredRaw.realYields ?? null),
        copperGold: num(fredRaw.indicators?.copperGold?.value ?? fredRaw.copperGold ?? null),
        sentiment: num(fredRaw.indicators?.sentiment?.value ?? fredRaw.sentiment ?? null),
        nfci: num(fredRaw.checklist?.nfci?.value ?? fredRaw.nfci ?? null),
    };

    // /api/breadth
    const breadthRaw = raw.breadth || {};
    const breadthPairs = breadthRaw.pairs || {};
    const breadth = {};
    for (const [key, pair] of Object.entries(breadthPairs)) {
        breadth[key] = {
            ratio: num(pair.ratio ?? null),
            chg20Pct: num(pair.chg20Pct ?? null),
            chg60Pct: num(pair.chg60Pct ?? null),
            vs50dPct: num(pair.vs50dPct ?? null),
        };
    }

    // /api/sheets — AAII bull-bear spread (may be "24.50%" or "N/A")
    const sheetsRaw = raw.sheets || {};
    const aaiiDiff = num(sheetsRaw.AAIIDiff ?? null);

    // T10Y3M from raw
    const t10y3m = num(raw.t10y3m ?? null);

    return { spy, fg, vol, fred, t10y3m, breadth, aaiiDiff };
}

// ---------------------------------------------------------------------------
// pillFactors — per-pill input details for the detail modal
// ---------------------------------------------------------------------------

export function pillFactors(data) {
    if (!data) {
        const empty = {};
        for (const pill of PILLS) {
            empty[pill] = { summary: 'no data', rows: [] };
        }
        return empty;
    }

    const s = data.spy || {};
    const fg = data.fg || {};
    const fred = data.fred || {};
    const brd = data.breadth || {};
    const rsp = brd.rspSpy || {};
    const iwm = brd.iwmSpy || {};
    const hyg = brd.hygLqd || {};
    const volSpy = (data.vol || {}).spy || {};

    const ma200Pct = safeNum(s.ma200Pct);
    const fgScore = safeNum(fg.score);
    const hygChg20 = safeNum(hyg.chg20Pct);
    const yieldCurve = safeNum(fred.yieldCurve);
    const sahmRule = safeNum(fred.sahmRule);
    const claims = safeNum(fred.claims);
    const nfci = safeNum(fred.nfci);
    const rspChg20 = safeNum(rsp.chg20Pct);
    const rspVs50d = safeNum(rsp.vs50dPct);
    const iwmChg20 = safeNum(iwm.chg20Pct);
    const high52Pct = safeNum(s.high52Pct);
    const ivPctile = safeNum(volSpy.ivPctile1y);
    const vrp = safeNum(volSpy.vrp);
    const t10y3m = safeNum(data.t10y3m);

    const factors = {};

    // ---- regime ----
    {
        let score = 0;
        const rows = [];

        // SPY vs 200-day avg
        let ma200Effect = '0';
        let ma200Hit = false;
        if (ma200Pct != null) {
            if (ma200Pct > 0) { ma200Effect = '+1'; ma200Hit = true; score += 1; }
            else { ma200Effect = '−1'; ma200Hit = true; score -= 1; }
        }
        rows.push({
            label: 'SPY vs 200-day avg',
            value: fmtPct(ma200Pct),
            test: '> 0 → +1, else −1',
            hit: ma200Hit,
            effect: ma200Effect,
        });

        // Fear & Greed
        let fgEffect = '0';
        let fgHit = false;
        if (fgScore != null) {
            if (fgScore >= 50) { fgEffect = '+1'; fgHit = true; score += 1; }
            else if (fgScore < 30) { fgEffect = '−1'; fgHit = true; score -= 1; }
            // else 30-49: no hit, no score change
        }
        rows.push({
            label: 'Fear & Greed',
            value: fmtInt(fgScore),
            test: '≥ 50 → +1 · < 30 → −1 · else 0',
            hit: fgHit,
            effect: fgEffect,
        });

        // HYG/LQD 20d
        let hygEffect = '0';
        let hygHit = false;
        if (hygChg20 != null) {
            if (hygChg20 > 0) { hygEffect = '+1'; hygHit = true; score += 1; }
            else if (hygChg20 < -1) { hygEffect = '−1'; hygHit = true; score -= 1; }
            // else between -1 and 0: no hit, score unchanged
        }
        rows.push({
            label: 'HYG/LQD 20d',
            value: fmtPct(hygChg20),
            test: '> 0 → +1 · < −1 → −1 · else 0',
            hit: hygHit,
            effect: hygEffect,
        });

        let regimeVerdict;
        if (score >= 2) regimeVerdict = 'risk-on';
        else if (score <= -1) regimeVerdict = 'risk-off';
        else regimeVerdict = 'neutral';

        factors.regime = {
            summary: `Score ${score} → ${regimeVerdict} (risk-on needs ≥ 2, risk-off ≤ −1)`,
            rows,
        };
    }

    // ---- recession ----
    // One row per INPUT (not per rule) so the popup never repeats an input; each row
    // carries every threshold that input takes part in. Verdict = strongest effect
    // among the rows that fired — identical to ruleVerdicts' first-match chain.
    {
        const rows = [];
        const claimsHigh = claims != null && claims >= 260;
        const ycInverted = yieldCurve != null && yieldCurve < 0;

        const sahmEffect = sahmRule == null ? '' : sahmRule >= 0.5 ? 'high' : sahmRule >= 0.2 ? 'rising' : '';
        rows.push({
            label: 'Sahm rule',
            value: fmtNum(sahmRule),
            test: '≥ 0.2 → rising · ≥ 0.5 → high',
            hit: sahmEffect !== '',
            effect: sahmEffect,
        });

        const ycEffect = !ycInverted ? '' : claimsHigh ? 'high' : 'rising';
        rows.push({
            label: 'Yield curve (2s10s)',
            value: fmtNum(yieldCurve),
            test: '< 0 → rising · < 0 with claims ≥ 260k → high',
            hit: ycEffect !== '',
            effect: ycEffect,
        });

        const claimsEffect = !claimsHigh ? '' : ycInverted ? 'high' : 'rising';
        rows.push({
            label: 'Jobless claims',
            value: claims != null ? `${Math.round(claims)}k` : 'n/a',
            test: '≥ 260k → rising · ≥ 260k with curve < 0 → high',
            hit: claimsEffect !== '',
            effect: claimsEffect,
        });

        const nfciHit = nfci != null && nfci > 0;
        rows.push({
            label: 'NFCI',
            value: fmtNum(nfci),
            test: '> 0 → rising',
            hit: nfciHit,
            effect: nfciHit ? 'rising' : '',
        });

        const fired = rows.filter((r) => r.hit);
        const recessionVerdict = fired.some((r) => r.effect === 'high') ? 'high' : fired.length ? 'rising' : 'low';
        const names = fired.map((r) => r.label).join(', ');

        factors.recession = {
            summary: recessionVerdict === 'low'
                ? 'Strongest warning wins; none tripped → low'
                : `${names} tripped → ${recessionVerdict}`,
            rows,
        };
    }

    // ---- breadth ----
    {
        const rows = [];

        // Determine verdict
        const rollingOver = rspChg20 != null && rspVs50d != null && rspChg20 < -1.5 && rspVs50d < 0;
        const broadRsp = rspChg20 != null && rspChg20 > 0 && iwmChg20 != null && iwmChg20 > 0;

        let breadthVerdict = 'narrow';
        if (rollingOver) breadthVerdict = 'rolling-over';
        else if (broadRsp) breadthVerdict = 'broad';

        // 1. RSP/SPY 20d
        let rsp20Effect = '';
        let rsp20Hit = false;
        if (rollingOver) { rsp20Effect = 'rolling-over'; rsp20Hit = true; }
        else if (broadRsp) { rsp20Effect = 'broad'; rsp20Hit = true; }
        rows.push({
            label: 'RSP/SPY 20d',
            value: fmtPct(rspChg20),
            test: '< −1.5% with RSP vs 50d < 0 → rolling-over · > 0 with IWM > 0 → broad',
            hit: rsp20Hit,
            effect: rsp20Effect,
        });

        // 2. RSP/SPY vs 50d avg
        const vs50dHit = rollingOver;
        rows.push({
            label: 'RSP/SPY vs 50d avg',
            value: fmtPct(rspVs50d),
            test: '< 0 (with 20d < −1.5%) → rolling-over',
            hit: vs50dHit,
            effect: vs50dHit ? 'rolling-over' : '',
        });

        // 3. IWM/SPY 20d
        const iwmHit = broadRsp;
        rows.push({
            label: 'IWM/SPY 20d',
            value: fmtPct(iwmChg20),
            test: '> 0 (with RSP 20d > 0) → broad',
            hit: iwmHit,
            effect: iwmHit ? 'broad' : '',
        });

        let summary;
        if (breadthVerdict === 'broad') {
            summary = 'Broad participation: RSP and IWM both positive over 20d';
        } else if (breadthVerdict === 'rolling-over') {
            summary = 'Rolling over: RSP/SPY declining below 50d avg';
        } else {
            summary = 'Narrow participation: not meeting broad or rolling-over thresholds';
        }

        factors.breadth = { summary, rows };
    }

    // ---- hedging ----
    {
        const rows = [];

        // Determine verdict
        const cheap = ivPctile != null && ivPctile < 20 && vrp != null && vrp < 6;
        const expensive = (ivPctile != null && ivPctile > 70) || (vrp != null && vrp > 10);
        let hedgingVerdict;
        if (expensive) hedgingVerdict = 'expensive';
        else if (cheap) hedgingVerdict = 'cheap';
        else hedgingVerdict = 'fair';

        // 1. IV percentile (1y)
        const ivExpensive = ivPctile != null && ivPctile > 70;
        const ivCheap = ivPctile != null && ivPctile < 20 && vrp != null && vrp < 6;
        const ivHit = ivExpensive || ivCheap;
        let ivEffect = '';
        if (ivExpensive) ivEffect = 'expensive';
        else if (ivCheap) ivEffect = 'cheap';
        rows.push({
            label: 'IV percentile (1y)',
            value: fmtInt(ivPctile),
            test: '> 70 → expensive · < 20 (with VRP < 6) → cheap',
            hit: ivHit,
            effect: ivEffect,
        });

        // 2. VRP
        const vrpExpensive = vrp != null && vrp > 10;
        const vrpCheap = vrp != null && vrp < 6 && ivPctile != null && ivPctile < 20;
        const vrpHit = vrpExpensive || vrpCheap;
        let vrpEffect = '';
        if (vrpExpensive) vrpEffect = 'expensive';
        else if (vrpCheap) vrpEffect = 'cheap';
        rows.push({
            label: 'VRP',
            value: fmtNum(vrp),
            test: '> 10 → expensive · < 6 (with IV pct < 20) → cheap',
            hit: vrpHit,
            effect: vrpEffect,
        });

        factors.hedging = {
            summary: `Hedging is ${hedgingVerdict}`,
            rows,
        };
    }

    // ---- conflict ----
    {
        const cps = conflictPairs(data);

        const pairDefs = [
            {
                label: 'sentiment vs price',
                value: `F&G ${fmtInt(fgScore)} · SPY ${fmtPct(ma200Pct)} vs 200d`,
                test: 'F&G < 35 with SPY > +3% · or F&G > 65 with SPY < −3%',
            },
            {
                label: '2s10s vs 3m10y',
                value: `2s10s ${fmtNum(yieldCurve)} · 3m10y ${fmtNum(t10y3m)}`,
                test: 'Opposite signs between 2s10s and 3m10y',
            },
            {
                label: 'credit vs equities',
                value: `SPY ${fmtPct(ma200Pct)} vs 200d · HYG/LQD ${fmtPct(hygChg20)}`,
                test: 'SPY > 0% vs 200d with HYG/LQD 20d < −1.5%',
            },
            {
                label: 'breadth vs index',
                value: `SPY ${fmtPct(high52Pct)} from 52w high · RSP/SPY ${fmtPct(rspChg20)}`,
                test: 'SPY within 2% of high with RSP/SPY 20d < −1.5%',
            },
        ];

        const rows = pairDefs.map((def) => {
            const firing = cps.find((cp) => cp.pair === def.label);
            return {
                label: def.label,
                value: def.value,
                test: def.test,
                hit: !!firing,
                effect: firing ? 'divergence' : '',
            };
        });

        const numFiring = cps.length;
        let conflictVerdict;
        if (numFiring === 0) conflictVerdict = 'aligned';
        else if (numFiring === 1) conflictVerdict = 'mild-divergence';
        else conflictVerdict = 'major-divergence';

        factors.conflict = {
            summary: `${numFiring} of 4 pairs diverge → ${conflictVerdict}`,
            rows,
        };
    }

    return factors;
}