# Deploy CareersLow (Render + Vercel)

Two ways to share the app publicly:

| Setup | URL | Best for |
|-------|-----|----------|
| **Render only** | `https://careerslow.onrender.com` | Simplest — API, UI, and WebSocket on one domain |
| **Vercel + Render** | `https://your-app.vercel.app` → API on Render | Custom frontend CDN; API stays on Render |

Voice mock interviews need **WebSockets** → backend must run on **Render** (not Vercel serverless).

---

## Prerequisites

1. **GitHub repo** with this project pushed
2. **[Google AI Studio](https://aistudio.google.com/apikey)** → `GOOGLE_API_KEY`
3. **[Sarvam](https://www.sarvam.ai/)** → `SARVAM_API_KEY` (voice STT/TTS)
4. **[Qdrant Cloud](https://cloud.qdrant.io/)** free cluster:
   - Cluster URL → `QDRANT_URL` (e.g. `https://xxxx.cloud.qdrant.io:6333`)
   - API key → `QDRANT_API_KEY`

---

## Option A — Render (all-in-one, recommended)

Serves **frontend + API + WebSocket** from one service.

### 1. Push to GitHub

```bash
git add .
git commit -m "Add Render/Vercel deploy config"
git push origin main
```

### 2. Create Render Blueprint

1. Go to [render.com](https://render.com) → **New** → **Blueprint**
2. Connect your GitHub repo
3. Render reads `render.yaml` and creates:
   - Web service `careerslow` (Docker)
   - Redis `careerslow-redis`

### 3. Set secrets in Render dashboard

For the **careerslow** web service → **Environment**:

| Key | Value |
|-----|--------|
| `GOOGLE_API_KEY` | your Gemini key |
| `SARVAM_API_KEY` | your Sarvam key |
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |

`REDIS_URL` is injected automatically from the linked Redis service.

### 4. Deploy

After deploy, open:

```
https://careerslow.onrender.com
```

Health check: `https://careerslow.onrender.com/health`

**Note:** Free tier sleeps after ~15 min idle. First request may take 30–60s (cold start).

---

## Option B — Vercel (frontend) + Render (backend)

### 1. Deploy backend on Render first

Follow **Option A** above. Copy your Render URL, e.g.:

```
https://careerslow.onrender.com
```

### 2. Deploy frontend on Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New** → **Project**
2. Import the same GitHub repo
3. **Environment variable** (Production):

| Key | Value |
|-----|--------|
| `API_BASE_URL` | `https://careerslow.onrender.com` (no trailing slash) |

4. Deploy — Vercel runs `node scripts/write-frontend-config.js` which sets `frontend/config.js`

Your public UI:

```
https://your-project.vercel.app
```

All API and voice traffic goes to Render via `API_BASE_URL`.

### 3. CORS

Backend already allows `allow_origins=["*"]`. No extra CORS setup needed.

---

## Local test (production-like)

```bash
# Terminal 1 — infra
docker compose up -d

# Terminal 2 — API
export $(grep -v '^#' .env | xargs)
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# Browser
open http://localhost:8000
```

Test Vercel-style frontend pointing at local API:

```bash
API_BASE_URL=http://localhost:8000 node scripts/write-frontend-config.js
# Serve frontend/ with any static server, or open via Render locally
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| 503 / Redis errors | Check `REDIS_URL` on Render; Redis service must be running |
| Qdrant connection failed | Use HTTPS cluster URL + `QDRANT_API_KEY` from Qdrant Cloud |
| Voice not working on Vercel | Expected — WebSocket must hit Render URL; check `API_BASE_URL` |
| Cold start timeout | Free Render tier; retry after spin-up or upgrade plan |
| PDF parse fails | Ensure PDF is text-based, not scanned images only |

---

## Files added

- `Dockerfile` — Render Docker build
- `render.yaml` — Render Blueprint (web + Redis)
- `vercel.json` — Vercel static frontend build
- `scripts/write-frontend-config.js` — injects `API_BASE_URL` for Vercel
- `frontend/config.js` — runtime API base (empty = same origin)
- `.env.example` — local env template
