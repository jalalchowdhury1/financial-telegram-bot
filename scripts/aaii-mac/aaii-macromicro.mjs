#!/usr/bin/env node
/**
 * 🍎 AAII backup tier, run on the Mac mini by launchd (com.jalal.aaii-macromicro).
 *
 * Why: aaii.com 503s for days at a time and AAII's Substack lags a week; Vercel cannot read
 * MacroMicro (Cloudflare challenge), but a real Chrome on the Mac can. So this opens
 * MacroMicro in real Google Chrome (own throwaway profile, window parked off-screen, always
 * closed), reads "Latest Stats", and — ONLY when its survey week is newer than the committed
 * one — updates `dashboard/lib/data/aaiiNewest.json` on main through the GitHub contents API
 * (the Mac's existing `gh` login; no KV or Vercel secrets on this machine). The push makes
 * Vercel redeploy; the dashboard's AAII resolver (dashboard/lib/aaii.js) serves that file
 * whenever its live tiers have an older week. About one commit a week, at most.
 *
 * Logs one line per run to ~/Library/Logs/aaii-macromicro.log. `--dry-run` reads only.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright-core';
import { parseLatestStats, toPayload, shouldPush } from './parse.mjs';

const URL = 'https://en.macromicro.me/charts/20828/us-aaii-sentimentsurvey';
const REPO = 'jalalchowdhury1/financial-telegram-bot';
const FILE = 'dashboard/lib/data/aaiiNewest.json';
const GH = fs.existsSync('/opt/homebrew/bin/gh') ? '/opt/homebrew/bin/gh' : 'gh'; // native arm64 gh (Rosetta gh froze)
const HOME = os.homedir();
const PROFILE = path.join(HOME, '.cache/aaii-macromicro-profile');
const LOG = path.join(HOME, 'Library/Logs/aaii-macromicro.log');
const DRY = process.argv.includes('--dry-run');

const log = (msg) => {
    const line = `${new Date().toISOString()} ${msg}`;
    console.log(line);
    try { fs.appendFileSync(LOG, `${line}\n`); } catch { /* console still has it */ }
};

const gh = (args, input) => execFileSync(GH, args, { input, encoding: 'utf8', timeout: 60000, stdio: ['pipe', 'pipe', 'pipe'] });

/** The committed file: { sha, value } (value null when missing). */
function readCommitted() {
    const out = JSON.parse(gh(['api', `repos/${REPO}/contents/${FILE}?ref=main`]));
    const text = Buffer.from(out.content, 'base64').toString('utf8');
    return { sha: out.sha, value: JSON.parse(text) };
}

function commit(value, sha, message) {
    const body = JSON.stringify({
        message,
        content: Buffer.from(`${JSON.stringify(value, null, 2)}\n`).toString('base64'),
        sha,
        branch: 'main',
    });
    const out = JSON.parse(gh(['api', '-X', 'PUT', `repos/${REPO}/contents/${FILE}`, '--input', '-'], body));
    return out.commit?.sha?.slice(0, 7);
}

async function readMacroMicro() {
    const ctx = await chromium.launchPersistentContext(PROFILE, {
        channel: 'chrome',
        headless: false, // headless Chrome gets Cloudflare's challenge; a parked real window passes
        args: ['--window-position=-2400,-2400', '--window-size=1200,900', '--disable-blink-features=AutomationControlled'],
        ignoreDefaultArgs: ['--enable-automation'],
    });
    try {
        const page = ctx.pages()[0] || await ctx.newPage();
        await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 60000 });
        let text = '';
        for (let i = 0; i < 30; i++) {
            await page.waitForTimeout(2000);
            text = await page.innerText('body').catch(() => '');
            if (parseLatestStats(text)) break;
        }
        const stats = parseLatestStats(text);
        if (!stats) {
            const why = /Just a moment|security verification/i.test(text) ? 'Cloudflare challenge did not clear' : 'Latest Stats not found';
            throw new Error(`${why} (title: ${await page.title().catch(() => '?')})`);
        }
        return stats;
    } finally {
        await ctx.close().catch(() => {}); // never leave a Chrome window behind
    }
}

try {
    const stats = await readMacroMicro();
    const payload = toPayload(stats);
    const said = `macromicro ${stats.released} → week ${payload.as_of}: bull ${payload.bull} neutral ${payload.neutral} bear ${payload.bear} diff ${payload.diff}`;
    if (DRY) { log(`DRY ${said}`); process.exit(0); }
    const { sha, value: cur } = readCommitted();
    if (!shouldPush(cur, payload)) {
        log(`OK ${said} | repo already has ${cur?.data?.as_of} (${cur?.data?.source}) — no commit`);
        process.exit(0);
    }
    const c = commit({ data: payload, savedAt: new Date().toISOString() }, sha,
        `AAII backup: survey week ${payload.as_of} (diff ${payload.diff}) from MacroMicro\n\nWritten by scripts/aaii-mac on the Mac mini (launchd com.jalal.aaii-macromicro).`);
    log(`PUSHED ${said} | replaced ${cur?.data?.as_of || 'nothing'} | commit ${c}`);
} catch (e) {
    log(`FAIL ${String(e?.stderr || e?.message || e).replace(/\s+/g, ' ').slice(0, 200)}`);
    process.exit(1);
}
