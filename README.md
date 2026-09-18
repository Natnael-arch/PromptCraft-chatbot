# unipods-bot — Phase 1: WhatsApp group-memory ingestion skeleton

Every message sent in a WhatsApp group is captured into Postgres by this stack, proving
the full pipe — WhatsApp → WAHA → webhook → FastAPI → Postgres — works end to end before
any AI/retrieval logic is added.

| Service            | What it does here                                                         | Image                                   |
| ------------------ | ------------------------------------------------------------------------- | --------------------------------------- |
| `waha`             | WhatsApp connection (NOWEB engine — no browser), delivers webhooks        | `devlikeapro/waha:noweb-2026.8.2`       |
| `postgres`         | Target database for captured messages (+ `pgvector`, used in Phase 2)     | `pgvector/pgvector:pg16`                |
| `redis`            | Provisioned now for the Phase 3 job queue (unused in Phase 1)             | `redis:7-alpine`                        |
| `backend`          | FastAPI app: receives webhooks, persists `messages`, exposes debug routes | built from `./backend`                  |

### Layout

```
unipods-bot/
  docker-compose.yml
  .env.example
  README.md
  backend/
    Dockerfile
    requirements.txt
    alembic.ini
    alembic/                 # migrations (schema is versioned, not create_all)
      env.py
      versions/0001_initial_schema.py
    app/
      main.py                # FastAPI app + startup
      config.py              # pydantic-settings reads .env
      db.py                  # engine / session / Base
      models.py              # ORM: messages, messages_unparsed, chunks
      schemas.py             # Pydantic models
      webhook.py             # POST /webhook/waha
      routes_debug.py        # GET /health, GET /messages
```

On a captured message the backend stores one row in `messages`. Webhooks that cannot be
mapped (e.g. if a future WAHA release changes the payload shape) are never dropped —
they land in the `messages_unparsed` fallback table with a `reason`. An empty `chunks`
table (`embedding vector(1024)`) is provisioned now so Phase 2 needs no migration.

---

## 1. Prerequisites

- **Docker** (with Docker Compose). On most installs compose is included:
  ```bash
  docker --version
  docker compose version
  ```
- Linux/macOS shell for the curl examples below. Windows users: run the commands from
  WSL2 or PowerShell (substitute `$WAHA_API_KEY` with your actual key).
- A **dedicated WhatsApp account / phone number** to pair the bot with (do not pair a
  number you use day-to-day — linked devices are visible to the account owner and WAHA
  has full read/write access to it).

---

## 2. Configure `.env`

```bash
cp .env.example .env
```

What to fill in:

- `WAHA_API_KEY` — any long random string. This is the API key WAHA requires on every
  request (`X-Api-Key` header) and the same key the backend uses for its health probe.
  Generate one with `openssl rand -hex 32`.
- `POSTGRES_PASSWORD` — a Postgres password (used by the database container and by the
  backend to build its connection URL).
- `WAHA_SESSION` — the session name; the default `default` is fine unless you want a
  different one — then use that name in every curl below.
- `WAHA_PORT` / `BACKEND_PORT` — host ports for WAHA's Swagger/API (3000) and the
  backend API (8000). Leave as-is unless a port is taken.
- `WAHA_BASE_URL` / `WHATSAPP_HOOK_URL` — **internal** compose-network URLs. Only change
  these if you relocate services off the default network. See Troubleshooting.

The rest (`POSTGRES_USER`, `POSTGRES_DB`, internal URLs, `REDIS_URL`) have working
defaults and can stay untouched.

---

## 3. Start the stack

```bash
docker compose up -d
```

This builds and starts all four services. Postgres runs healthchecks before the backend
even boots; the backend then applies Alembic migrations (`alembic upgrade head`) and
starts uvicorn. Verify:

```bash
docker compose ps
```

All four services should be `Up` / `healthy`. Then confirm the backend sees its two
dependencies:

```bash
curl -s http://localhost:8000/health
# {"status":"ok","db":true,"waha":true}
```

---

## 4. Start a WAHA session and pair it via QR

First load `.env` into the shell so `$WAHA_API_KEY` expands in the curl commands
(this also allows `curl -o qr.png ...`):

```bash
set -a && source .env && set +a
```

### 4a. Create and start the session

```bash
curl -X POST http://localhost:3000/api/sessions \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: $WAHA_API_KEY" \
  -d '{"name":"default"}'
# {"name":"default","status":"STARTING","engine":{"engine":"NOWEB"},...}
```

The session starts immediately and reports `SCAN_QR_CODE`.

### 4b. Fetch and scan the QR code

```bash
curl -o qr.png "http://localhost:3000/api/default/auth/qr?format=image" \
  -H "X-Api-Key: $WAHA_API_KEY"
open qr.png   # mac: `open`, linux: `xdg-open`, windows: `start`
```

Scan it from the **dedicated phone number**:
**WhatsApp → Settings → Linked Devices → Link a Device → scan the QR.**

The QR refreshes periodically; if the session is past `SCAN_QR_CODE`, just re-run the
`curl -o qr.png` command above and scan the new one.

**Alternative — pairing code (NOWEB supports it):** instead of scanning, ask WAHA for a
code and enter it from `Linked Devices → Link with phone number`:

```bash
curl -X POST "http://localhost:3000/api/default/auth/request-code" \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: $WAHA_API_KEY" \
  -d '{"phoneNumber":"<country_code><full_number>"}'   # e.g. {"phoneNumber":"16501234567"}
# {"code":"ABCD-ABCD"}
```

### 4c. Confirm the session is working

```bash
curl -s "http://localhost:3000/api/sessions/default" -H "X-Api-Key: $WAHA_API_KEY"
```

Expect `"status": "WORKING"` and the `me` object populated with your bot's number.

> **Alternative UI:** open http://localhost:3000/ (WAHA's Swagger). Click **Authorize**
> and paste `WAHA_API_KEY`. Then run `POST /api/sessions` with body `{"name":"default"}`
> and `GET /api/default/auth/qr` — Swagger displays/lets you download the QR directly.

### Session persistence

Pairing data lives in the `waha_sessions` named volume mounted at `/app/.sessions`.
WAHA also auto-restarts previously-running sessions on boot, so after any restart the
session comes back without re-scanning:

```bash
docker compose restart waha
curl -s "http://localhost:3000/api/sessions/default" -H "X-Api-Key: $WAHA_API_KEY"   # still WORKING
```

---

## 5. Confirm the webhook is wired

1. Add the paired number to a WhatsApp group, and send a message in that group from
   **another** account (or reply from the bot's own account).
2. Within a couple of seconds the backend logs `Captured message ...` — watch it with:
   ```bash
   docker compose logs -f backend
   ```
3. Verify it is in the database:
   ```bash
   curl -s "http://localhost:8000/messages?limit=10"
   ```
   ```json
   [
     {
       "id": "38f2…",
       "waha_message_id": "false_…@c.us_…",
       "session_name": "default",
       "chat_id": "1203630…@g.us",
       "is_group": true,
       "sender_id": "…@c.us",
       "sender_name": "…",
       "from_me": false,
       "msg_type": "text",
       "body": "hello world from the group",
       "timestamp": "2026-09-18T14:12:31+00:00",
       "raw_payload": "…"
     }
   ]
   ```
   Filter to one group: `curl -s "http://localhost:8000/messages?chat_id=1203630…@g.us"`.

Everything the webhook received is also stored verbatim in `raw_payload`, and any payload
the mapper couldn't parse went to `messages_unparsed` instead of being lost.

### Idempotency check (webhook retries)

WAHA retries failed deliveries, so the same payload must never insert twice. Send the
same webhook body twice and confirm only one row exists:

```bash
curl -s -X POST http://localhost:8000/webhook/waha \
  -H "Content-Type: application/json" \
  -d '{"event":"message","session":"default","payload":{"id":"dupe_test_00001","from":"999@c.us","fromMe":false,"body":"hi","timestamp":1694900000}}'
# first call:  {"status":"ok","stored":true,"deduplicated":false}
# second call: {"status":"ok","stored":false,"deduplicated":true}
```

---

## 5.5. Phase 2/3: embedding, retrieval, and answer providers

Phase 2 added a retrieval pipeline over the captured history: exported/backfilled
messages are grouped into **sessions**, each session is split into **chunks**, and
each chunk is embedded into a `vector(1024)` in the `chunks` table (imports are
idempotent — re-importing a file deletes-and-replaces that chat's chunks, never
duplicates). Phase 3 made **Gemini the primary provider** for embeddings and
answer generation, while keeping the mock/extractive providers for offline dev.

### Embedding provider (`EMBEDDING_PROVIDER`)

| Provider             | How it works                                                        | Needs API key |
| -------------------- | ------------------------------------------------------------------- | ------------- |
| `gemini`             | Gemini via the google-genai SDK; requests MRL `output_dimensionality=1024` (an API request param, the API truncates server-side) and L2-renormalizes the truncated vector. Default model `gemini-embedding-001`. | `GEMINI_API_KEY` |
| `openai_compatible`  | Any OpenAI-style `/embeddings` server (Ollama/vLLM/...). Use `EMBEDDING_API_URL`/`EMBEDDING_API_KEY`/`EMBEDDING_MODEL`. | API key for your server |
| `mock`               | Deterministic pseudo-random unit vectors, **DEV ONLY** (offline, no key). Good for exercising the pipeline end-to-end. | — |

`EMBEDDING_DIMENSIONS` must match the provider's output dims (the schema column
is `VECTOR(1024)`). For Gemini that means `1024`; MRL truncation happens natively
via the request parameter, not via client-side slicing.

### Answer provider (`ANSWER_PROVIDER`)

| Provider       | Behavior                                                                 | Needs API key |
| -------------- | ------------------------------------------------------------------------ | ------------- |
| `gemini`       | Retrieves chunks/sessions the same way, then passes the extracted context + citations to Gemini (`ANSWER_MODEL`, default `gemini-2.5-flash`) to synthesize a natural-language `answer_text`. `citations`/`sources` are unchanged (routes_ask contract is stable). Falls back to extractive on any failure. | `GEMINI_API_KEY` |
| `extractive`   | Returns the Phase-2 bullet list verbatim — offline default, no LLM call.  | —             |

### Ask flow (POST /ask)

1. `route_question` classifies the question: **time_range** (regex, temporal
   phrases/dates) or **semantic**.
2. **semantic** routes embed the question with the active embedder and run
   `hybrid_search` (pgvector cosine + full-text, merged with reciprocal-rank
   fusion) over that chat's chunks.
3. The answer provider synthesizes `answer_text` over the retrieved context;
   `citations` (message-level) and `sources` (chunk/session-level) are always
   attached.

---

## 6. Troubleshooting

**No messages arriving / `GET /messages` stays empty — check, in order:**

1. **Session status.** Is the session actually `WORKING`?
   ```bash
   curl -s "http://localhost:3000/api/sessions/default" -H "X-Api-Key: $WAHA_API_KEY"
   ```
   - `SCAN_QR_CODE` → pairing incomplete; re-fetch/scan the QR (section 4b).
   - `FAILED` → restart the session (`POST /api/sessions/default/restart`), then
     logout & re-pair if it persists.
2. **Webhook URL reachable from inside the WAHA container.** WAHA resolves the hook
   through the **compose network**, so the URL must be `http://backend:8000/...`, never
   `http://localhost:8000/...` — `localhost` inside the `waha` container is WAHA itself.
   Verify from inside WAHA's container:
   ```bash
   docker compose exec waha node -e "fetch('http://backend:8000/health').then(r=>r.text()).then(console.log)"
   # {"status":"ok","db":true,"waha":true}
   ```
   If that fails, check `docker compose ps` (backend container running) and that
   `WHATSAPP_HOOK_URL` in `.env` points at `backend:8000` — then `docker compose up -d`
   to recreate `waha` with the new hook URL.
3. **Backend logs.** `docker compose logs -f backend`. You should see
   `Captured message id=…` on every webhook and `Ignoring unhandled WAHA event=…` for
   event types Phase 1 doesn't process (silent no-ops are expected and fine).
4. **Was the webhook really delivered?** Subscribe to more events (optional) by setting
   `WHATSAPP_HOOK_EVENTS=message,session.status` in `.env` + `docker compose up -d`, then
   watch `docker compose logs -f backend` while sending a message — `session.status`
   events arriving proves WAHA→backend delivery works.
5. **API key mismatch.** If the backend log shows `401` from the WAHA health probe
   (`Health: WAHA returned HTTP 401`), the `WAHA_API_KEY` passed to `waha` differs from
   the one in `WAHA_BASE_URL` calls. Make sure both come from the same `.env` value and
   run `docker compose up -d` (recreates `waha` and `backend`).
6. **Polling vs webhook.** WAHA also exposes `GET /api/messages?chatId=…&session=default`
   (needs `X-Api-Key`) as a cross-check that the message event fired at all.

**Common WAHA gotchas**

- **Engine:** this compose file pins the NOWEB engine (`noweb-2026.8.2`, no browser).
  Don't swap to the `latest`/`webjs` tags — tag is pinned deliberately; bump it in
  `docker-compose.yml` after reading the [changelog](https://waha.devlike.pro/docs/overview/changelog/).
- **Media:** WAHA holds received media at `media.url` and requires `X-Api-Key` to
  download. Phase 1 doesn't download media — `media_path` stays `NULL` until Phase 3.
- **`messages_unparsed`:** if captures stop but webhooks keep arriving, rows here tell
  you *why* (the `reason` column). Inspect with:
  ```bash
  docker compose exec postgres psql -U unipods -d unipods \
    -c 'SELECT reason, count(*) FROM messages_unparsed GROUP BY reason;'
  ```
- **Banned/blocked number:** pairing gotchas around new numbers are covered by WAHA's
  own note on [avoiding blocking](https://waha.devlike.pro/docs/overview/how-to-avoid-blocking/).
- **Ports:** if `3000` or `8000` are busy, change `WAHA_PORT`/`BACKEND_PORT` in `.env`
  and re-run `docker compose up -d`; every port reference above follows those variables.