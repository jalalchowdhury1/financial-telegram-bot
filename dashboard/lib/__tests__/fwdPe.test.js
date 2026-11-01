import { parseFactSet, reportUrl, reportCandidates, decodeFwd, encodeFwd, joinFwd, resolveFwdPe, FWD_KV_KEY, PROBE_GAP_MS, PROBE_BUDGET_MS, BLOG_FEED, blogPosts, reportDayOf, htmlText } from '../fwdPe';

// Sentences copied from real reports (text as unpdf gives it, line breaks and all).
const OCT_2026 = 'PS guidance. • Valuation: The forward 12-month P/E ratio for the S&P 500 is 18.7. This P/E ratio is below the\n5-year average (19.8) and below the 10-year average (19.1). To receive this report via e-mail';
const JAN_2017 = 'The forward 12-month P/E ratio is now 17.1, which is above the 5-year average and the 10-year average. During the upcoming week ... '
    + 'The forward 12-month P/E ratio is 17.1. This P/E ratio is above the 5-year average of 15.1, and above the 10-year average of 14.4.';
const APR_2011 = '+ Valuation: The current 12-month forward P/E ratio for the S&P 500 is 13.1. This P/E ratio is above the average forward 12-month P/E ratio of the past year (12.7). '
    + 'The current forward 12-month P/E ratio of 13.1 is well below the prior ten-year average forward 12-month P/E ratio of 15.4.';
const JUN_2012 = '+ Valuation: The current 12-Month forward P/E ratio is 11.9. This P/E ratio is below the average forward 12-Month P/E ratio over the past 10 years (14.4). + Earnings Scorecard';
const JUL_2014 = '+ Valuation: The current 12-month forward P/E ratio is 15.7. This P/E ratio is based on Wednesday’s closing price (1974.62) and forward 12-month EPS estimate ($126.17). '
    + 'On the other hand, the current forward 12-month P/E ratio is still (slightly) below the 15-year average (15.8). '
    + 'Of the 22 companies that have reported earnings to date for Q2 2014, 73% have reported earnings above the mean estimate. '.repeat(4)
    + 'The P/E ratio of 15.7 for the index as a whole is above the prior 5-year average forward 12-month P/E ratio of 13.3, and above the prior 10-year average forward 12-month P/E ratio of 13.8. At the sector level';
const TRAILING_TRAP = 'The forward 12-month P/E ratio is 18.7. The trailing 12-month P/E ratio is 25.9, which is above the 5-year average of 24.4 and above the 10-year average of 23.7.';

describe('parseFactSet', () => {
    it('reads page 1 of a 2026 report', () => {
        expect(parseFactSet(OCT_2026)).toEqual({ pe: 18.7, a5: 19.8, a10: 19.1 });
    });
    it('takes the averages from the first statement that names them (2017 wording)', () => {
        expect(parseFactSet(JAN_2017)).toEqual({ pe: 17.1, a5: 15.1, a10: 14.4 });
    });
    it('2011 wording: forward 12-month = 12-month forward, a ten-year average and no 5-year one', () => {
        expect(parseFactSet(APR_2011)).toEqual({ pe: 13.1, a5: null, a10: 15.4 });
    });
    it('2012 wording: "the average … over the past 10 years (14.4)"', () => {
        expect(parseFactSet(JUN_2012)).toEqual({ pe: 11.9, a5: null, a10: 14.4 });
    });
    it('2013–2016: the averages from the index-as-a-whole sentence, never the 15-year average', () => {
        expect(parseFactSet(JUL_2014)).toEqual({ pe: 15.7, a5: 13.3, a10: 13.8 });
    });
    it('never takes the trailing P/E or its averages', () => {
        expect(parseFactSet(TRAILING_TRAP)).toEqual({ pe: 18.7, a5: null, a10: null });
    });
    it('null without a forward P/E', () => {
        expect(parseFactSet('The trailing 12-month P/E ratio is 25.9.')).toBeNull();
        expect(parseFactSet(null)).toBeNull();
    });
});

describe('report dates', () => {
    it('builds the PDF url, with the A variant', () => {
        expect(reportUrl('2026-10-30')).toMatch(/Earnings%20Insight\/EarningsInsight_103026\.pdf$/);
        expect(reportUrl('2022-08-05', 'A')).toMatch(/EarningsInsight_080522A\.pdf$/);
    });
    it('lists Wednesdays to Fridays after the newest report, newest first', () => {
        expect(reportCandidates('2026-11-06', '2026-10-30')).toEqual(['2026-11-06', '2026-11-05', '2026-11-04']);
        expect(reportCandidates('2026-11-02', '2026-10-30')).toEqual([]);
    });
});

describe('baked file', () => {
    const rows = [{ date: '2011-04-01', pe: 13.1, a5: null, a10: 15.4 }, { date: '2026-10-30', pe: 18.7, a5: 19.8, a10: 19.1 }];
    it('round-trips', () => {
        const body = encodeFwd(rows, { source: 'x' });
        expect(body).toMatchObject({ source: 'x', from: '2011-04-01', to: '2026-10-30' });
        expect(decodeFwd(JSON.parse(JSON.stringify(body)))).toEqual(rows);
    });
    it('joins newer rows only', () => {
        expect(joinFwd(rows.slice(0, 1), [{ date: '2011-04-01', pe: 99 }, rows[1]])).toEqual(rows);
        expect(decodeFwd({ from: 'x' })).toBeNull();
    });
});

describe('resolveFwdPe', () => {
    const NOW = Date.parse('2026-11-06T20:00:00Z'); // Fri 15:00 ET, the day's report is out
    const kvWith = (doc) => {
        const kv = { doc, get: jest.fn(async () => kv.doc), set: jest.fn(async (k, v) => { kv.doc = v; return true; }) };
        return kv;
    };
    const pdfText = async (bytes) => bytes.text;
    const run = (kv, fetchPdf, extra = {}) => resolveFwdPe({ fetchPdf, pdfText, kv, bakedTo: '2026-10-23', today: '2026-11-06', now: NOW, ...extra });

    it('finds the new report, parses it and keeps it in KV', async () => {
        const kv = kvWith({ rows: [{ date: '2026-10-30', pe: 18.7, a5: 19.8, a10: 19.1 }], seen: '2026-10-30', checkedAt: NOW - 4 * 3600e3 });
        const fetchPdf = jest.fn(async (url) => (url.includes('110626.pdf') ? { text: OCT_2026.replace('18.7', '18.4') } : null));
        const r = await run(kv, fetchPdf);
        expect(r.rows.map((x) => [x.date, x.pe])).toEqual([['2026-10-30', 18.7], ['2026-11-06', 18.4]]);
        expect(r.seen).toBe('2026-11-06');
        expect(kv.set).toHaveBeenCalledWith(FWD_KV_KEY, { rows: r.rows, seen: '2026-11-06', checkedAt: NOW });
    });
    it('does not look again within 3 h, or while the newest is under 5 days old', async () => {
        const fetchPdf = jest.fn();
        await run(kvWith({ rows: [], seen: null, checkedAt: NOW - PROBE_GAP_MS + 60e3 }), fetchPdf);
        await run(kvWith({ rows: [{ date: '2026-11-04', pe: 18 }], checkedAt: 0 }), fetchPdf);
        expect(fetchPdf).not.toHaveBeenCalled();
    });
    it('a report it cannot read moves `seen` but not the rows (so /api/freshness goes red)', async () => {
        const kv = kvWith({ rows: [{ date: '2026-10-30', pe: 18.7 }], checkedAt: 0 });
        const r = await run(kv, async (url) => (url.includes('110626.pdf') ? { text: 'new layout' } : null));
        expect(r.rows).toHaveLength(1);
        expect(r.seen).toBe('2026-11-06');
        expect(r.messages).toContain('2026-11-06: report has no forward P/E line');
    });
    it('tries the A file, and drops rows the bake already has', async () => {
        const kv = kvWith({ rows: [{ date: '2026-10-16', pe: 19 }], checkedAt: 0 });
        const r = await run(kv, async (url) => (url.includes('110526A.pdf') ? { text: OCT_2026 } : null));
        expect(r.rows.map((x) => x.date)).toEqual(['2026-11-05']);
    });
    it('stops fetching once its time budget is spent (newest day first)', async () => {
        let t = 1e12;
        const clock = jest.spyOn(Date, 'now').mockImplementation(() => t);
        const fetchPdf = jest.fn(async () => { t += PROBE_BUDGET_MS + 1; return null; }); // FactSet hanging
        const r = await run(kvWith(null), fetchPdf);
        clock.mockRestore();
        expect(fetchPdf.mock.calls.map(([u]) => u.split('_').pop())).toEqual(['110626.pdf']);
        expect(r.messages).toContain('time budget hit at 2026-11-06');
    });
    it('FactSet or KV down: answers with what it has, never throws', async () => {
        const kv = { get: async () => { throw new Error('down'); }, set: async () => { throw new Error('down'); } };
        const r = await run(kv, async () => { throw new Error('HTTP 429'); });
        expect(r.rows).toEqual([]);
        expect(r.messages[0]).toBe('KV read failed');
        expect(r.messages).toContain('2026-11-06: HTTP 429');
        const gated = jest.fn();
        await run(kvWith(null), gated, { faults: new Set(['fwdpe_http']) });
        expect(gated).not.toHaveBeenCalled();
    });
});

describe('FactSet blog (the second layer)', () => {
    const post = (pe) => `<html><script>var x = "P/E ratio is 99.9";</script><p>The forward 12-month P/E ratio is <b>${pe}</b>, which is below the 5-year average (19.8) and below the 10-year average&nbsp;(19.1).</p></html>`;
    const feed = (items) => `<rss><channel><link>https://insight.factset.com</link>${items.map(([slug, when]) => `<item><title>S&amp;P 500 ${slug}</title><link>https://insight.factset.com/${slug}</link><pubDate>${when}</pubDate></item>`).join('')}</channel></rss>`;

    it('reads the same sentence as the PDF, and dates a post by the report it retells', () => {
        expect(parseFactSet(htmlText(post('18.7')))).toEqual({ pe: 18.7, a5: 19.8, a10: 19.1 });
        expect(blogPosts(feed([['fri', 'Sat, 07 Nov 2026 02:00:00 GMT'], ['mon', 'Mon, 09 Nov 2026 16:00:00 GMT']])))
            .toEqual([{ date: '2026-11-09', link: 'https://insight.factset.com/mon' }, { date: '2026-11-06', link: 'https://insight.factset.com/fri' }]); // 02:00 GMT Sat = Fri evening in New York
        expect(['2026-11-09', '2026-11-08', '2026-11-05'].map(reportDayOf)).toEqual(['2026-11-06', '2026-11-06', '2026-11-05']);
    });

    const NOW = Date.parse('2026-11-06T20:00:00Z');
    const kvWith = (doc) => ({ get: async () => doc, set: jest.fn(async () => true) });
    const pdfText = async (bytes) => bytes.text;
    const run = (extra) => resolveFwdPe({ pdfText, bakedTo: '2026-10-23', today: '2026-11-06', now: NOW, ...extra });
    const PAGES = {
        [BLOG_FEED]: feed([['fri', 'Fri, 06 Nov 2026 18:08:00 GMT'], ['mining', 'Thu, 05 Nov 2026 16:00:00 GMT'], ['mon', 'Mon, 02 Nov 2026 16:00:00 GMT']]),
        'https://insight.factset.com/fri': post('18.4'), 'https://insight.factset.com/mining': '<p>Drill results</p>', 'https://insight.factset.com/mon': post('18.7'),
    };
    const fetchText = jest.fn(async (url) => PAGES[url]);
    beforeEach(() => fetchText.mockClear());

    it('no PDF this week: the blog post stands in; posts retelling a known report are never fetched', async () => {
        const kv = kvWith({ rows: [{ date: '2026-10-30', pe: 18.7, a5: 19.8, a10: 19.1 }], checkedAt: 0 });
        const r = await run({ kv, fetchPdf: async () => null, fetchText });
        expect(r.rows.map((x) => [x.date, x.pe])).toEqual([['2026-10-30', 18.7], ['2026-11-06', 18.4]]);
        expect(r.messages).toContain('from the blog: 2026-11-06');
        expect(r.seen).toBe('2026-11-06'); // found = seen: no red freshness row while the screen is current
        expect(fetchText.mock.calls.map(([u]) => u)).not.toContain('https://insight.factset.com/mon'); // Mon 2 Nov retells 30 Oct
        expect(kv.set).toHaveBeenCalledWith(FWD_KV_KEY, expect.objectContaining({ rows: r.rows }));
    });
    it('not read when a PDF did, nor behind its gate', async () => {
        await run({ kv: kvWith(null), fetchPdf: async (url) => (url.includes('110626.pdf') ? { text: OCT_2026 } : null), fetchText });
        await run({ kv: kvWith(null), fetchPdf: async () => null, fetchText, faults: new Set(['fwdpe_http', 'fwdpe_blog']) });
        expect(fetchText).not.toHaveBeenCalled();
    });
    it('KV down: the instance copy stands in, and every answer refreshes that copy', async () => {
        const copy = { rows: [{ date: '2026-11-06', pe: 18.4 }], seen: '2026-11-06', checkedAt: NOW - 3600e3 };
        const lastGood = { load: jest.fn(() => copy), save: jest.fn() };
        const r = await run({ kv: { get: async () => { throw new Error('down'); } }, lastGood, fetchPdf: jest.fn(), fetchText });
        expect(r.rows).toEqual(copy.rows);
        expect(r.messages).toEqual(['KV read failed', 'KV missed: served this instance\'s last copy']);
        expect(lastGood.save).toHaveBeenCalledWith(copy);
    });
});
