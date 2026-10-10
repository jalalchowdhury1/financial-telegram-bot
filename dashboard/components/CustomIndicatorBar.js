'use client';
import Delta from './Delta';
import { useMark } from './MarkProvider';

// "Survey Sep 23 · aaii.com" under the AAII pill; a missed week (> 9 days, set by
// /api/sheets from lib/aaii.js) turns it into a warning so old data never looks fresh.
export function aaiiAsOf(aaii) {
    if (!aaii || !aaii.as_of) return null;
    const d = new Date(`${aaii.as_of}T12:00:00Z`);
    if (Number.isNaN(d.getTime())) return null;
    const label = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
    const stale = !!aaii.stale;
    return (
        <div className="pill-detail" data-testid="aaii-asof"
            style={{ fontSize: '0.6rem', marginTop: '2px', color: stale ? 'var(--orange)' : 'var(--text-muted)', fontWeight: stale ? 700 : 400 }}>
            {stale ? `⚠️ STALE · survey ${label}` : `Survey ${label}`} · {aaii.source}
        </div>
    );
}

/**
 * Phone-only one-liner under the AAII number (≤500px hides every .pill-detail, so the
 * survey date and the STALE warning above never showed on the iPhone). The number on
 * the pill is AAIIDiff = bear − bull, so a positive diff means bears lead:
 * "Bears +11.9 · Sep 30", or an orange "⚠ STALE · Sep 23" when a week was missed.
 * No survey date or no number -> nothing (never a guess).
 */
export function aaiiCaption(aaii, diffStr) {
    if (!aaii || !aaii.as_of) return null;
    const diff = parseFloat(diffStr);
    if (!Number.isFinite(diff)) return null;
    const d = new Date(`${aaii.as_of}T12:00:00Z`);
    if (Number.isNaN(d.getTime())) return null;
    const label = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
    const stale = !!aaii.stale;
    const lead = Math.abs(diff).toFixed(1);
    const side = diff > 0 ? `Bears +${lead}` : diff < 0 ? `Bulls +${lead}` : 'Even';
    return (
        <div className={`pill-caption${stale ? ' pill-caption-stale' : ''}`} data-testid="aaii-caption">
            {stale ? `⚠ STALE · ${label}` : `${side} · ${label}`}
        </div>
    );
}

/**
 * Orange "⚠ cached Oct 8" line under a pill whose value /api/sheets served from a
 * last-good copy or a lagged source (its field is in `_meta.staleFields`, see
 * lib/sheetsCascade.js). Shown on desktop AND phone — an old value must never look
 * live. Nothing stale → nothing rendered.
 */
export function staleNote(sheets, fields) {
    const stale = sheets?._meta?.staleFields || [];
    const hit = fields.filter((f) => stale.includes(f));
    if (!hit.length) return null;
    const info = sheets._meta.fields?.[hit[0]] || {};
    const d = info.savedAt ? new Date(info.savedAt) : null;
    const when = d && !Number.isNaN(d.getTime())
        ? d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/New_York' })
        : null;
    const src = info.source || '';
    const what = when ? `cached ${when}` : /FRED/.test(src) ? 'FRED, lags a day' : /stopped updating/.test(src) ? 'frozen sheet' : 'not live';
    return (
        <div data-testid={`stale-${hit[0]}`} title={info.source || ''}
            style={{ fontSize: '0.6rem', marginTop: '2px', color: 'var(--orange)', fontWeight: 700 }}>
            ⚠ STALE · {what}
        </div>
    );
}

export default function CustomIndicatorBar({ sheets, loading }) {
    // AAII is the only pill that earns a mark: it prints weekly on Thursdays.
    // VIX changes daily, and NotSoBoring / FrontRunner are not in the history sheet.
    const aaiiMark = useMark('aaiiDiff', parseFloat(sheets?.AAIIDiff));
    return (
        <div className="indicator-bar">
            <div className={`indicator-pill${!loading && sheets?.NotSoBoring && sheets.NotSoBoring !== 'ON' ? ' pill-alert' : ''}`}>
                <div className="label">
                    <span className="tooltip-trigger" data-tooltip="Crash Detector: Monitors Tech (QQQ) and Bonds (TMF) for drops >6-7%. Defensive shift adds Gold and USD to dilute risk.">
                        <span className="emoji">🛡️</span>NotSoBoring
                    </span>
                </div>
                <div className="value">{loading ? '...' : (sheets?.NotSoBoring || 'N/A')}</div>
                {!loading && staleNote(sheets, ['NotSoBoring'])}
            </div>
            {/* The only pill that links out: opens the Catalyst Radar, which shows how
                close each Frontrunner trigger is and what crossing it would do. */}
            <a
                href="https://nuts-radar.vercel.app"
                target="_blank"
                rel="noopener noreferrer"
                title="Open the Catalyst Radar — how close each trigger is, and what is scheduled"
                className={`indicator-pill pill-link${!loading && sheets?.FrontRunner && !sheets.FrontRunner.startsWith('BIL') ? ' pill-alert' : ''}`}
            >
                <div className="label">
                    <span className="tooltip-trigger" data-tooltip="A contrarian strategy that rotates into Volatility (VIX) hedges when markets overheat (RSI > 79) and buys oversold Tech/Leveraged ETFs during deep dips. Click to open the Catalyst Radar.">
                        <span className="emoji">🔑</span>FrontRunner
                    </span>
                    <span className="pill-out" aria-hidden="true">↗</span>
                </div>
                <div className="value">{loading ? '...' : (sheets?.FrontRunner || 'N/A')}</div>
                {!loading && staleNote(sheets, ['FrontRunner'])}
            </a>
            <div className={`indicator-pill${!loading && sheets?.AAIIDiff && parseFloat(sheets.AAIIDiff) > 20 ? ' pill-alert' : ''}`}>
                <div className="label"><span className="emoji">🔸</span>AAII Diff</div>
                {!loading && sheets?.AAIIDiff ? (() => {
                    const val = parseFloat(sheets.AAIIDiff);
                    const isBullish = val > 20;
                    const targetDate = new Date();
                    targetDate.setMonth(targetDate.getMonth() + 6);
                    const dateStr = targetDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
                    return (
                        <>
                            <Delta mark={aaiiMark} className="value" chartKey="aaiiDiff" raw={parseFloat(sheets?.AAIIDiff)}
                                format={(v) => `${v > 0 ? '+' : ''}${v.toFixed(2)}%`}>
                                <span style={{ color: isBullish ? 'var(--green)' : 'var(--text-primary)' }}>
                                    {sheets.AAIIDiff}
                                </span>
                            </Delta>
                            {aaiiCaption(sheets?.AAII, sheets.AAIIDiff)}
                            <div className="pill-detail" style={{ fontSize: '0.68rem', marginTop: '4px', color: isBullish ? 'var(--green)' : 'var(--text-muted)', fontWeight: 600 }}>
                                {isBullish ? '🟢' : '⚪'} {isBullish ? 'Bullish' : 'Neutral'} outlook → {dateStr}
                            </div>
                            <div className="pill-detail" style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                                Threshold: &gt;20% = bullish 6mo forward
                            </div>
                            {aaiiAsOf(sheets?.AAII)}
                        </>
                    );
                })() : <div className="value">{loading ? '...' : 'N/A'}</div>}
            </div>
            <div className={`indicator-pill${!loading && sheets?.VIX?.current && parseFloat(sheets.VIX.current) > parseFloat(sheets.VIX.threeMonth) ? ' pill-alert' : ''}`}>
                <div className="label">
                    <span className="tooltip-trigger" data-tooltip="Oversold Signal: Short-term panic (Current VIX) exceeds medium-term expectations (3M VIX). Often precedes a market recovery.">
                        <span className="emoji">🎢</span>VIX (Current | 3M)
                    </span>
                </div>
                {/* Desktop reads one line "14.84 | 17.77 | GREED01". On a phone the half-width
                    card wrapped mid-line, so ≤560px drops the last " | " and puts the tag
                    on its own line as a small chip (globals.css "phone pass"). */}
                <div className="value">
                    {loading ? '...' : (sheets?.VIX?.current
                        ? <>
                            {`${sheets.VIX.current} | ${sheets.VIX.threeMonth}`}
                            <span className="vix-tag"><span className="vix-tag-sep"> | </span>{sheets.VIX.fearGreed}</span>
                        </>
                        : 'N/A')}
                </div>
                {!loading && staleNote(sheets, ['vixCurrent', 'vixThreeMonth', 'vixFearGreed'])}
            </div>
        </div>
    );
}
