# Polymarket Trending Bets Table — Design Spec

**Date**: 2026-04-27
**Feature**: Add a Polymarket data table to the main dashboard showing top 10 non-settled, most-trending bets from today (excluding sports)
**Status**: Approved for Implementation

---

## 1. Overview

The financial dashboard will gain a new section displaying Polymarket's most-trending prediction market bets, sourced from the `warproxxx/poly_data` library. The table will show bet name, current odds, and volume — updated fresh each time the dashboard page loads.

**Success Criteria:**
- ✓ Table displays top 10 non-settled Polymarket bets by volume
- ✓ Sports category is excluded
- ✓ Data fetches fresh on each dashboard page load
- ✓ Graceful error handling if API unavailable
- ✓ Follows existing project patterns (backend fetcher, Lambda endpoint, frontend component)

---

## 2. Architecture

### Data Flow
```
Polymarket API (via warproxxx/poly_data)
    ↓
bot/fetchers.py: fetch_polymarket_trending()
    ↓
lambda_handler.py: /api/polymarket endpoint
    ↓
dashboard/components/PolymarketTable.js (via useEffect)
    ↓
Browser: Rendered table (name, odds, volume)
```

### API Contract

**Endpoint**: `GET /api/polymarket`

**Response** (200 OK):
```json
{
  "bets": [
    {
      "name": "Will Donald Trump be re-elected in 2024?",
      "odds": 0.65,
      "volume": 2500000
    },
    ...
  ],
  "timestamp": "2026-04-27T20:15:00Z",
  "error": null
}
```

**Error Response** (5xx):
```json
{
  "bets": [],
  "error": "Polymarket API unavailable",
  "timestamp": "2026-04-27T20:15:00Z"
}
```

---

## 3. Backend Implementation

### Python Fetcher: `bot/fetchers.py`

Add new function `fetch_polymarket_trending()`:

**Responsibilities:**
- Import `warproxxx/poly_data` (or equivalent public API)
- Query Polymarket for non-settled markets
- Filter out sports category
- Sort by volume (descending)
- Return top 10: `[{name, odds, volume}, ...]`
- Handle API failures gracefully (return empty list with error logged)

**Error Handling:**
- Catch rate-limit errors (429) → log to CloudWatch, return empty list
- Catch network timeouts → return empty list with error message
- Never raise exceptions (graceful degradation)

**Caching (optional, Phase 2):**
- Consider caching results for 5 minutes to avoid hammering the Polymarket API on every dashboard load

---

### Lambda Handler: `lambda_handler.py`

Add new endpoint:

**Path**: `/api/polymarket`
**Method**: `GET`
**Auth**: Public (same as `/api/spy`, `/api/spy-daily-move`, `/api/market-extra`)

**Handler Logic:**
```
1. Call fetch_polymarket_trending()
2. Build response object with bets, timestamp, error
3. Return via _ok() helper (handles NaN cleaning)
```

**Update `_PUBLIC_GET_PATHS`:**
```python
_PUBLIC_GET_PATHS = {'/api/spy', '/api/spy-daily-move', '/api/market-extra', '/api/polymarket'}
```

---

## 4. Frontend Implementation

### New Component: `dashboard/components/PolymarketTable.js`

**Props**: None (data is fetched internally)

**State:**
- `bets`: array of bet objects
- `loading`: boolean
- `error`: string | null

**Behavior:**
- On component mount (`useEffect`): fetch from `/api/polymarket`
- Display table with columns: Name, Odds, Volume
- Show loading spinner while fetching
- Show error message if fetch fails
- Empty state: "No data available"

**Styling:**
- Match existing dashboard component aesthetics (ExtraMarketsGrid, EconomicIndicatorGrid)
- Table or card-based grid layout (to be determined during implementation)

---

### Main Page Integration: `dashboard/app/page.js`

**Change:**
- Import `PolymarketTable`
- Add `<PolymarketTable />` as a new section below existing components

**Section Title**: "Polymarket Trending Bets"

---

## 5. Dependencies & Installation

### Backend
Add to `aws/requirements-lambda.txt`:
```
poly_data  # or warproxxx/poly-data if available on PyPI
```

### Frontend
No new npm packages required (use existing React/Next.js)

---

## 6. Error Handling & Resilience

**Scenario 1: Polymarket API Down**
- Lambda returns 200 with empty bets array and error message
- Frontend displays: "Data unavailable — try refreshing"
- Does not break page load

**Scenario 2: Rate Limit Hit**
- Lambda logs error, returns empty bets
- Frontend gracefully shows empty state

**Scenario 3: Network Timeout**
- Fetcher catches timeout, logs to CloudWatch
- Returns empty array with error message
- Frontend displays error state

---

## 7. Testing Strategy

### Backend Testing
- Test `fetch_polymarket_trending()` with mock API response
- Verify filtering (sports excluded, non-settled only)
- Verify top 10 by volume sorting
- Test error handling (rate limit, timeout, invalid response)

### Frontend Testing
- Verify table renders with sample data
- Verify loading state displays
- Verify error state displays
- Verify empty state displays

### End-to-End
- Deploy to Lambda staging
- Verify `/api/polymarket` returns correct format
- Load dashboard, verify table appears and updates

---

## 8. Deployment Checklist

- [ ] Add `poly_data` to `aws/requirements-lambda.txt`
- [ ] Create new branch: `feature/polymarket-table`
- [ ] Implement `fetch_polymarket_trending()` in `bot/fetchers.py`
- [ ] Add `/api/polymarket` endpoint to `lambda_handler.py`
- [ ] Create `PolymarketTable.js` component
- [ ] Integrate into `dashboard/app/page.js`
- [ ] Test locally (mock API responses)
- [ ] Package and deploy Lambda (`manylinux2014_x86_64`)
- [ ] Verify endpoint works: `curl https://<lambda-url>/api/polymarket`
- [ ] Verify dashboard table appears
- [ ] PR review
- [ ] Merge to main

---

## 9. Phase 2 Considerations (Future)

- Add caching to Lambda endpoint (5-min TTL) to reduce API calls
- Add sorting/filtering controls to the table (by odds, volume, name)
- Add category multi-select filter (show only certain categories)
- Historical tracking: log top bets to database for trend analysis
- Real-time updates via WebSocket (if Polymarket offers it)

---

## Notes

- This design follows the existing project pattern: Python fetcher → Lambda endpoint → Next.js component
- Error handling is non-breaking (graceful degradation)
- No breaking changes to existing endpoints
- Data freshness: page load (not scheduled)
