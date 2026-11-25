# LLM Chatbot with Key Failover and Response Caching

Full-stack resilient AI chat application built with **React (Vite + TypeScript)**, **FastAPI**, **LiteLLM**, and **SQLite**.

---

## 1. Live URLs

> Replace these placeholders with your public deployment URLs once deployed (see deployment steps below):

- **Frontend (Public UI):** `https://<your-frontend-deployment-url>`
- **Backend (FastAPI Base URL):** `https://<your-backend-deployment-url>`
- **Health Check:** `https://<your-backend-deployment-url>/health`

---

## 2. Run locally

### Prerequisites
- At least 2 LLM API keys (e.g., Google Gemini, OpenAI, or Groq) configured in `.env`.
- Copy `.env.example` to `.env` and fill in your real keys:
  ```bash
  cp .env.example .env
  ```

### With Docker (Recommended)
Start both services with a persistent `/data` SQLite volume:
```bash
make up
```
- **Frontend:** http://localhost:3000
- **Backend:** http://localhost:8000 (`GET /health`)

Stop the stack (preserving the SQLite volume):
```bash
make down
```

### Without Docker

#### Quick Make Targets
```bash
# Terminal 1 (Backend on :8000 with local SQLite ./data/app.db)
make dev-backend

# Terminal 2 (Frontend on :3000)
make dev-frontend
```

#### Manual Setup
**Backend:**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL="sqlite:///./data/app.db"
export LLM_API_KEYS="your-key-1,your-key-2"
export LLM_MODEL="gemini/gemini-3.5-flash-lite"
export ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
export ENABLE_FAILURE_SIMULATION="true"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Run backend tests:
```bash
make test
```

**Frontend:**
```bash
cd frontend
npm install
export VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```
Open http://localhost:3000.

### Cloud Deployment (Render + Vercel)
1. **Backend (Render Docker Web Service):**
   - Root directory: `backend` (uses multi-stage `backend/Dockerfile`)
   - Add a **Persistent Disk** mounted at `/data`
   - Environment variables:
     - `LLM_API_KEYS=<key_1>,<key_2>`
     - `LLM_MODEL=gemini/gemini-3.5-flash-lite`
     - `DATABASE_URL=sqlite:////data/app.db`
     - `ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app`
     - `ENABLE_FAILURE_SIMULATION=true`
     - `CACHE_TYPE=local`
2. **Frontend (Vercel / Netlify):**
   - Root directory: `frontend`, Build command: `npm run build`, Output directory: `dist`
   - Environment variable: `VITE_API_BASE_URL=https://<your-render-backend>.onrender.com`

---

## 3. Design decisions

### Key fallback
- `LLM_API_KEYS` is parsed in order into `key_1`, `key_2`, ... inside [`backend/app/config.py`](backend/app/config.py). Each entry shares `LLM_MODEL` or overrides per-key models via `LLM_MODELS`.
- In [`backend/app/llm_gateway.py`](backend/app/llm_gateway.py), `LLMGateway.complete` iterates deterministically through the ordered key pool with `num_retries=0` and a configurable `timeout` (`LLM_TIMEOUT_SECONDS`) per key so a failing key immediately falls back to the next key rather than stalling on retries against a broken key.
- Failover triggers on HTTP `429`, quota/billing/exhaustion errors, `5xx` errors (`500`, `502`, `503`, `504`), timeouts, and connection errors. If every key fails, `AllProvidersUnavailableError` maps cleanly to HTTP `503 {"detail": "all providers unavailable"}` with zero stack traces.
- When `ENABLE_FAILURE_SIMULATION=true`, the `X-Simulate-Failure` header injects LiteLLM provider exceptions (`429`, `500`, `502`, `503`) for targeted keys (or `*`) for that single request, while setting `cache={"no-cache": True}` so cached replies never mask live failover verification.

### Caching
- Uses LiteLLM's built-in `litellm.caching.Cache` (`CACHE_TYPE=local` in-memory by default, or `CACHE_TYPE=disk` backed by `CACHE_DIR`).
- LiteLLM hashes `model` + full `messages` array (`system prompt + conversation history + new user message`) while excluding `api_key`, ensuring an identical first message in a new session returns `LLM_CACHE` even if a different key originally populated the cache.
- Cache hits return `resolution_source: "LLM_CACHE"`, `key_id: null`, `fallback_attempts: 0`, stored per-request token counts, and sub-millisecond/low-millisecond latency.

### SQLite session storage
- Pure SQLite via Python's built-in `sqlite3` (no external Redis or Postgres), stored at `DATABASE_URL` (mounted at `/data/app.db` via a named Docker volume `sqlite_data`).
- Schema is initialized automatically on startup in [`backend/app/db.py`](backend/app/db.py) with `messages` (matching the exact required columns + index on `(session_id, id)`) and `sessions` (to persist per-session `system_prompt` overrides across turns without exposing them in `GET /sessions/{session_id}/messages`).
- User and assistant turns are committed atomically in a single transaction **only after** a `200` reply is obtained, ensuring `422` and `503` errors never pollute session history.

---

## 4. Verification

### 1. Health check
```bash
curl -s http://localhost:8000/health | jq
# {"status": "ok", "keys_configured": 2, "cache_enabled": true}
```

### 2. Cache hit across sessions
Send the same first prompt in two different sessions:
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-demo-1","message":"What is the capital of France?"}' | jq

curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-demo-2","message":"What is the capital of France?"}' | jq
```
- First call: `resolution_source = "LIVE_INFERENCE"`, `key_id = "key_1"`, higher `latency_ms`.
- Second call: `resolution_source = "LLM_CACHE"`, `key_id = null`, `fallback_attempts = 0`, identical `reply` & token counts, significantly lower `latency_ms`.

### 3. Key failover via `X-Simulate-Failure`
```bash
# Force key_1 to return 429 -> key_2 answers
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"s-fail-1","message":"Say hello in 5 words."}' | jq
# metadata.key_id == "key_2", metadata.fallback_attempts == 1

# Force all keys to return 429 -> clean 503, nothing saved to SQLite
curl -i -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"s-fail-all","message":"Hello"}'
# HTTP/1.1 503 Service Unavailable -> {"detail":"all providers unavailable"}
```

### 4. Multi-turn state & per-message token usage
```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-demo-1","message":"What river runs through it?"}' | jq

curl -s http://localhost:8000/sessions/s-demo-1/messages | jq
```
- Each assistant reply reports only that request's `input_tokens` and `output_tokens`.
- In the React UI (`http://localhost:3000`), inspect `meta-source`, `meta-model`, `meta-latency`, `meta-input-tokens`, and `meta-output-tokens` rendered beneath every assistant reply.

---

## 5. Setup Used

- **Editor / IDE:** VS Code / Antigravity IDE
- **AI Tools:** Antigravity AI coding assistant

---

## 6. Bonus features (if any)

- **Failover Lab & SQLite Inspector in UI:** Collapsible toolbar in the React frontend allowing one-click `X-Simulate-Failure` header presets (`key_1=429`, `key_1=500`, `*=429`), quick starter prompts, error retry buttons, and direct loading of any persisted `session_id` from `GET /sessions/{session_id}/messages`.
- **Per-session `system_prompt` persistence:** Overriding `system_prompt` on turn 1 is persisted in SQLite's `sessions` table so subsequent turns in the same session keep the custom system prompt while `GET /sessions/{session_id}/messages` cleanly excludes system messages.
- **Multi-model pool & timeout failover (`LLM_MODELS`, `LLM_TIMEOUT_SECONDS`):** Supports heterogeneous model fallback pools and per-key request timeouts so stalled keys automatically fail over cleanly.
