#!/usr/bin/env bash
#
# purge_test_data.sh — delete Phase-5 simulated Webhook payloads that carry no
# timestamp. Real capture paths (live webhook, WAHA store, file export) always
# attach a timestamp; a NULL timestamp uniquely identifies the rows created by
# the Phase-5 manual sim harness (currently the 7 "/unhinged_on", "/unhinged_off"
# and "lmaooo @…" rows in the test group).
#
# Scope: project-wide `timestamp IS NULL` (not just the test group). Confirmed
# today this matches exactly the same 7 rows, but the wider predicate stays
# correct in any environment instead of hard-coding one chat id. The
# 30727051714790 fake-sender criterion from the original brief is intentionally
# DROPPED — Part-1 diagnosis confirmed no such sender rows exist (the number
# only appears inside real members' message bodies).
#
# This set is hygiene-only: the rows were already confirmed ABSENT from chunks
# (NULL-timestamp messages are skipped by sessionize), so no session/chunk
# cleanup is needed — this deletes from `messages` only.
#
# Usage:
#   scripts/purge_test_data.sh            # DRY-RUN: list matching rows, delete nothing
#   scripts/purge_test_data.sh --confirm  # actually delete, then print the count
#
# Environment:
#   PG_PSQL=<cmd>   override the psql invocation (default: docker compose exec)
#
# Reads POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB from the repo's .env.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: .env not found at ${ENV_FILE}" >&2
  exit 1
fi

# Load .env (docker-compose style KEY=value file).
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

CONFIRM=0
for arg in "$@"; do
  case "${arg}" in
    --confirm) CONFIRM=1 ;;
    *) echo "ERROR: unknown argument '${arg}' (expected: --confirm)" >&2; exit 1 ;;
  esac
done

PG_USER="${POSTGRES_USER:-unipods}"
PG_DB="${POSTGRES_DB:-unipods}"

# How we reach psql. Default: the stack's postgres container. Override with
# PG_PSQL when the script's requester runs outside the compose project.
if [[ -n "${PG_PSQL:-}" ]]; then
  PSQL_CMD=(${PG_PSQL})
else
  PSQL_CMD=(docker compose exec -T postgres psql -U "${PG_USER}" -d "${PG_DB}")
fi

WHERE_CLAUSE="timestamp IS NULL"

echo "=============================================================="
echo " Purge Phase-5 sim rows (NULL-timestamp payloads)"
echo "   scope     : messages.timestamp IS NULL (project-wide)"
echo "   postgres  : user=${PG_USER} db=${PG_DB}"
if [[ "${CONFIRM}" -eq 1 ]]; then
  echo "   mode      : *** DELETE ***"
else
  echo "   mode      : DRY-RUN (pass --confirm to delete)"
fi
echo "=============================================================="

COUNT_SQL="SELECT count(*) FROM messages WHERE ${WHERE_CLAUSE};"
LIST_SQL="SELECT id, chat_id, sender_id, from_me, body FROM messages
WHERE ${WHERE_CLAUSE}
ORDER BY created_at, id;"

if [[ "${CONFIRM}" -eq 1 ]]; then
  echo
  echo "Matching rows (will be deleted):"
else
  echo
  echo "Matching rows:"
fi

"${PSQL_CMD[@]}" -P pager=off -c "${LIST_SQL}"

if [[ "${CONFIRM}" -eq 1 ]]; then
  echo
  echo "Deleting ${WHERE_CLAUSE} ..."
  "${PSQL_CMD[@]}" -c "DELETE FROM messages WHERE ${WHERE_CLAUSE};"
  echo
  echo "Confirm remaining NULL-timestamp rows:"
  "${PSQL_CMD[@]}" -P pager=off -tA -c "${COUNT_SQL}"
  echo " (0 = all sim rows purged)"
else
  echo
  echo "DRY-RUN: nothing deleted."
  echo "Rows to delete:"
  "${PSQL_CMD[@]}" -P pager=off -tA -c "${COUNT_SQL}"
  echo "Re-run with --confirm to actually delete."
fi