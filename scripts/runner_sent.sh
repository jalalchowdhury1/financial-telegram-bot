#!/bin/bash
# Prints how many daily_report.yml runs in the last 24 h (or the last $1 seconds)
# actually SENT the brief.
# A run whose guard skipped (the Lambda already delivered) also ends "success", so a
# run's conclusion alone is not proof: count only runs whose report step succeeded.
# Used by daily_report.yml (guard) and health-check.yml (delivery evidence). Needs GH_TOKEN.
set -uo pipefail
max_age=${1:-86400}
n=0
for id in $(gh run list --workflow daily_report.yml --status success --limit 20 --json databaseId,createdAt \
    --jq ".[] | select((now - (.createdAt | fromdateiso8601)) < $max_age) | .databaseId" 2>/dev/null); do
  sent=$(gh run view "$id" --json jobs \
    --jq '[.jobs[].steps[] | select(.name == "Generate and send financial report" and .conclusion == "success")] | length' 2>/dev/null)
  [ "${sent:-0}" -gt 0 ] && n=$((n + 1))
done
echo "$n"
