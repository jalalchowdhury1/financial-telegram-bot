#!/usr/bin/env node
/**
 * 🍎 AAII backup tier, run on the Mac mini by launchd (com.jalal.aaii-macromicro).
 *
 * Why: aaii.com 503s for days at a time and AAII's Substack lags a week; Vercel cannot read
 * MacroMicro (Cloudflare challenge), but a real Chrome on the Mac can. So this opens
 * MacroMicro in real Google Chrome (own throwaway profile, window parked off-screen, always
 * closed), reads "Latest Stats", and writes the survey to Upstash KV `ftb:aaii:newest` —
 * ONLY when its survey week is newer than what KV holds. The dashboard's AAII resolver
 * (dashboard/lib/aaii.js) serves that copy whenever its live tiers have an older week.
 *
 * Reads ONLY KV_REST_API_URL / KV_REST_API_TOKEN from ~/.config/ftb-kv.env (chmod 600,
 * written by `vercel env pull`). Logs one line per run to ~/Library/Logs/aaii-macromicro.log.
 * `--dry-run` reads and prints, writes nothing.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { parseLatestStats, toPayload, shouldPush } from './parse.mjs';

const URL = 'https://en.macromicro.me/charts/20828/us-aaii-sentimentsurvey';
const KEY = 'ftb:aaii:newest';
const HOME = os.homedir();
const ENV_FILE = path.join(HOME, '.config/ftb-kv.env');
const PROFILE = path.join(HOME, '.cache/aaii-macromicro-profile');
const LOG = path.join(HOME, 'Library/Logs/aaii-macromicro.log');
const DRY = process.argv.includes('--dry-run');

const log = (msg) => {
    const line = `${new Date().toISOString()} ${msg}`;
    console.log(line);
    try { fs.appendFileSync(LOG, `${line}\n`); } catch { /* console still has it */ }
};

function kvCreds() {
    const out = {};
    for (const line of fs.readFileSync(ENV_FILE, 'utf8').split('\n')) {
        const m = /^(KV_REST_API_URL|KV_REST_API_TOKEN)=(.*)$/.exec(line.trim());
        if (m) out[m[1]] = m[2].replace(/^["']|["']$/g, '');
    }
    if (!out.KV_REST_API_URL || !out.KV_REST_API_TOKEN) throw new Error(`KV creds missing in ${ENV_FILE}`);
    return { base: out.KV_REST_API_URL.replace(/\/$/, ''), token: out.KV_REST_API_TOKEN };
}

async function kv(creds, pathPart, init = {}) {
    const res = await fetch(`${creds.base}${pathPart}`, {
        ...init,
        headers: { Authorization: `Bearer ${creds.token}`, ...(init.headers || {}) },
        signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) throw new Error(`KV ${pathPart.split('/')[1]} HTTP ${res.status}`);
    return res.json();
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
    const creds = kvCreds();
    const got = await kv(creds, `/get/${encodeURIComponent(KEY)}`);
    const cur = typeof got?.result === 'string' ? JSON.parse(got.result) : got?.result;
    if (!shouldPush(cur, payload)) {
        log(`OK ${said} | KV already has ${cur?.data?.as_of} (${cur?.data?.source}) — no write`);
        process.exit(0);
    }
    await kv(creds, `/set/${encodeURIComponent(KEY)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: payload, savedAt: new Date().toISOString() }),
    });
    log(`PUSHED ${said} | replaced ${cur?.data?.as_of || 'nothing'}`);
} catch (e) {
    log(`FAIL ${String(e?.message || e).slice(0, 200)}`);
    process.exit(1);
}
