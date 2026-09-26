import '@testing-library/jest-dom'

// Instant-open snapshots (lib/snapshot.js) live in localStorage: start every test clean
// so one test's saved copy never hydrates the next test's component.
beforeEach(() => {
    try { window.localStorage.clear(); } catch { /* node environment */ }
});
