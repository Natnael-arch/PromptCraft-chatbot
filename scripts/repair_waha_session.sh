#!/usr/bin/env bash
#
# repair_waha_session.sh — tear down and re-pair a WAHA session WITH history
# sync (noweb store) config enabled.
#
# WAHA only applies config.noweb.store (enabled/fullSync) at session CREATE
# time, BEFORE the QR is scanned. Patching a live session does not backfill
# history (and can lose it), so the only correct flow is:
#   logout -> confirm STOPPED -> delete -> recreate(with store config) -> start
#   -> HUMAN scans the QR (that last step cannot be scripted).
#
# Usage:
#   scripts/repair_waha_session.sh [session_name]     # default: "default"
#
# Environment:
#   WAHA_STORE_FULL_SYNC=true|false   fullSync for the new session (default: false)
#   WAHA_HTTP_BASE=<url>              override the WAHA base URL (else derived
#                                     from .env: localhost:$WAHA_PORT when
#                                     WAHA_BASE_URL is the internal compose URL)
#
# Reads WAHA_API_KEY / WAHA_PORT / WAHA_BASE_URL from the repo's .env, so it
# works unchanged in any environment this repo is deployed to.
#
# *** DESTRUCTIVE: logs out and DELETES the given session. The WhatsApp
# account must re-scan the QR afterwards. This is NOT a config hot-reload. ***
#
set -euo pipefail

SESSION_NAME="${1:-default}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: .env not found at ${ENV_FILE}" >&2
  exit 1
fi

# Load .env (docker-compose style KEY=value file; last assignment wins, same
# semantics as `docker compose`).
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

FULL_SYNC="${WAHA_STORE_FULL_SYNC:-false}"
case "${FULL_SYNC}" in
  true|false) ;;
  *) echo "ERROR: WAHA_STORE_FULL_SYNC must be 'true' or 'false' (got: ${FULL_SYNC})" >&2; exit 1 ;;
esac

API_KEY="${WAHA_API_KEY:-}"
if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: WAHA_API_KEY is empty/missing in ${ENV_FILE}" >&2
  exit 1
fi

# Resolve the WAHA base URL reachable from WHERE THIS SCRIPT RUNS.
# WAHA_BASE_URL in .env is the *internal* compose-network URL (http://waha:3000)
# when the stack runs under docker-compose; from the host we need localhost.
if [[ -n "${WAHA_HTTP_BASE:-}" ]]; then
  BASE_URL="${WAHA_HTTP_BASE%/}"
else
  _host="$(printf '%s' "${WAHA_BASE_URL:-}" | sed -E 's#^https?://([^:/]+).*#\1#')"
  if [[ -n "${_host}" && "${_host}" != "waha" ]]; then
    BASE_URL="${WAHA_BASE_URL%/}"
  else
    BASE_URL="http://localhost:${WAHA_PORT:-3000}"
  fi
fi

BODY_FILE="$(mktemp)"
trap 'rm -f "${BODY_FILE}"' EXIT

LAST_STATUS=""
LAST_BODY=""

# api_call METHOD PATH [JSON_BODY] [ALLOWED_STATUS...]
# Prints "METHOD PATH -> HTTP <code>". Allowed statuses default to 2xx.
# Exits non-zero (with the response body) when the status is not allowed.
api_call() {
  local method="$1" path="$2" data="${3:-}"
  shift 3 || true
  local allowed=("$@")
  if [[ ${#allowed[@]} -eq 0 ]]; then
    allowed=()
  fi

  local -a args=(-sS -o "${BODY_FILE}" -w '%{http_code}'
    -X "${method}" "${BASE_URL}${path}"
    -H "X-Api-Key: ${API_KEY}")
  if [[ -n "${data}" ]]; then
    args+=(-H 'Content-Type: application/json' --data "${data}")
  fi

  local status=""
  if ! status="$(curl "${args[@]}")"; then
    echo "  ${method} ${path} -> CURL FAILED (no response from ${BASE_URL})" >&2
    exit 1
  fi

  LAST_STATUS="${status}"
  LAST_BODY="$(cat "${BODY_FILE}" 2>/dev/null || true)"
  printf '  %-6s %-50s -> HTTP %s\n' "${method}" "${path}" "${status}"

  # 2xx always OK.
  if [[ "${status}" =~ ^2[0-9][0-9]$ ]]; then
    return 0
  fi
  local ok=""
  for code in ${allowed[@]+"${allowed[@]}"}; do
    [[ "${status}" == "${code}" ]] && ok=1 && break
  done
  if [[ -n "${ok}" ]]; then
    return 0
  fi
  echo "  FAILED (HTTP ${status}) — response: ${LAST_BODY}" >&2
  exit 1
}

get_session_status() {
  # Prints the session status, or "NOT_FOUND" (HTTP 404).
  local status
  status="$(curl -sS -o "${BODY_FILE}" -w '%{http_code}' \
    -X GET "${BASE_URL}/api/sessions/${SESSION_NAME}" \
    -H "X-Api-Key: ${API_KEY}")" || {
      echo "  GET /api/sessions/${SESSION_NAME} -> CURL FAILED" >&2
      exit 1
    }
  if [[ "${status}" == "404" ]]; then
    printf '  %-6s %-50s -> HTTP 404\n' "GET" "/api/sessions/${SESSION_NAME}" >&2
    echo "NOT_FOUND"
    return 0
  fi
  if [[ ! "${status}" =~ ^2[0-9][0-9]$ ]]; then
    printf '  %-6s %-50s -> HTTP %s\n' "GET" "/api/sessions/${SESSION_NAME}" "${status}" >&2
    echo "  FAILED (HTTP ${status}) — response: $(cat "${BODY_FILE}")" >&2
    exit 1
  fi
  printf '  %-6s %-50s -> HTTP %s\n' "GET" "/api/sessions/${SESSION_NAME}" "${status}" >&2
  # Extract "status":"..." without requiring jq.
  sed -n 's/.*"status":"\([^"]*\)".*/\1/p' "${BODY_FILE}" | head -n1
}

echo "=============================================================="
echo " WAHA session re-pair (history-sync store config enabled)"
echo "   session : ${SESSION_NAME}"
echo "   base url: ${BASE_URL}"
echo "   fullSync: ${FULL_SYNC}  (WAHA_STORE_FULL_SYNC)"
echo " *** DESTRUCTIVE: logs out + deletes '${SESSION_NAME}'.     ***"
echo " *** A human must re-scan the QR afterwards.                ***"
echo "=============================================================="

CURRENT_STATUS="$(get_session_status)"

if [[ "${CURRENT_STATUS}" == "NOT_FOUND" ]]; then
  echo "Step 1/6: logout  — session does not exist, nothing to log out (skipping)"
  echo "Step 2/6: confirm STOPPED — session does not exist (skipping)"
  echo "Step 3/6: delete  — session does not exist (skipping)"
else
  echo "Step 1/6: logout (current status: ${CURRENT_STATUS})"
  api_call POST "/api/sessions/${SESSION_NAME}/logout" "" 404

  echo "Step 2/6: confirm the session is unlinked (STOPPED or SCAN_QR_CODE)"
  FINAL_STATUS=""
  for _ in $(seq 1 30); do
    FINAL_STATUS="$(get_session_status)"
    if [[ "${FINAL_STATUS}" == "STOPPED" \
          || "${FINAL_STATUS}" == "SCAN_QR_CODE" \
          || "${FINAL_STATUS}" == "NOT_FOUND" ]]; then
      break
    fi
    sleep 1
  done
  if [[ "${FINAL_STATUS}" != "STOPPED" \
        && "${FINAL_STATUS}" != "SCAN_QR_CODE" \
        && "${FINAL_STATUS}" != "NOT_FOUND" ]]; then
    echo "  ERROR: session still paired after logout (status '${FINAL_STATUS}')" >&2
    exit 1
  fi
  echo "  confirmed status: ${FINAL_STATUS}"

  echo "Step 3/6: delete session"
  api_call DELETE "/api/sessions/${SESSION_NAME}" "" 404
fi

echo "Step 4/6: create session with store config (enabled=true, fullSync=${FULL_SYNC})"
CREATE_BODY="$(printf '{"name":"%s","config":{"noweb":{"store":{"enabled":true,"fullSync":%s}}}}' \
  "${SESSION_NAME}" "${FULL_SYNC}")"
echo "  body: ${CREATE_BODY}"
api_call POST "/api/sessions/" "${CREATE_BODY}"

echo "Step 5/6: start session"
api_call POST "/api/sessions/${SESSION_NAME}/start"

echo "Step 6/6: manual QR scan required"
cat <<EOF

==============================================================
 Session '${SESSION_NAME}' recreated with history sync and started.
 WAHA should now report status SCAN_QR_CODE.

 >>> NEXT STEP IS MANUAL — a human with the phone must scan: <<<

  1. Open the WAHA dashboard:  ${BASE_URL}
     (click Authorize and enter your WAHA_API_KEY if prompted)
  2. Open the session QR there, or fetch the image directly:
       curl -o qr.png "${BASE_URL}/api/${SESSION_NAME}/auth/qr?format=image" \\
         -H "X-Api-Key: \$WAHA_API_KEY"
  3. On the phone: WhatsApp -> Settings -> Linked Devices
     -> Link a Device -> scan the QR.
     The QR refreshes periodically; re-fetch if it goes stale.

 This script does NOT scan the QR — pairing always needs a human.
 After scanning, confirm with:
   curl -s "${BASE_URL}/api/sessions/${SESSION_NAME}" -H "X-Api-Key: \$WAHA_API_KEY"
 History backfill (store.fullSync) then runs in the background and can take
 anywhere from under a minute to several minutes to complete.
==============================================================
EOF
