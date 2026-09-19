/**
 * Jev API client — sends state text to TypeSafe's system-one endpoint and
 * returns parsed verdicts.
 *
 * Uses fetch + AbortController. Never throws: returns null on any failure
 * (no key, network error, timeout, unknown verdict shape) and logs one short
 * line via console.error.
 */

export const JEV_TIMEOUT_MS = 6000;
export const JEV_P_FLOOR = 0.6;

const JEV_API_URL = 'https://api.typesafe.ai/v1/systemone';

/**
 * Default HTTP poster. Uses fetch with an AbortController timeout.
 * Returns the parsed JSON body on success, throws on any failure.
 */
export async function defaultPoster(url, body, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const res = await fetch(url, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${process.env.TYPESAFE_API_KEY || ''}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
            signal: controller.signal,
        });
        if (!res.ok) {
            throw new Error(`Jev API HTTP ${res.status}`);
        }
        return res.json();
    } finally {
        clearTimeout(timer);
    }
}

/**
 * Call Jev with `state` text and `questions` (the JEV_QUESTIONS shape).
 *
 * @param {string} state  — compact plain text of market facts
 * @param {object} questions — { k: { type: 'choice', instructions, criteria: { verdict: description } } }
 * @param {Function} [post] — poster function, defaultPoster by default
 * @returns {object|null} — { k: { verdict, p } } for each answer whose
 *   `choice` is one of that question's criteria keys. Returns null when
 *   TYPESAFE_API_KEY is missing, or on any network/parse error.
 */
export async function judgeMany(state, questions, post = defaultPoster) {
    if (!process.env.TYPESAFE_API_KEY) {
        console.error('[jev] no TYPESAFE_API_KEY — skipping Jev call');
        return null;
    }

    try {
        const body = {
            state,
            model: 'jev-latest',
            questions,
        };
        const data = await post(JEV_API_URL, body, JEV_TIMEOUT_MS);

        if (!data || !data.answers) {
            console.error('[jev] Jev response missing answers');
            return null;
        }

        const result = {};
        for (const [k, answer] of Object.entries(data.answers)) {
            const question = questions[k];
            if (!question) continue;
            const criteriaKeys = Object.keys(question.criteria || {});
            const verdict = answer?.choice;
            if (!verdict || !criteriaKeys.includes(verdict)) {
                console.error(`[jev] unknown verdict "${verdict}" for question "${k}" — dropping`);
                continue;
            }
            result[k] = {
                verdict,
                p: typeof answer.confidence === 'number' ? answer.confidence : null,
            };
        }
        return Object.keys(result).length > 0 ? result : null;
    } catch (err) {
        console.error(`[jev] ${err.message}`);
        return null;
    }
}