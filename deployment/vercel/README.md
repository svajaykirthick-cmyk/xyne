# Xyne on Vercel (Direct Serverless Mode)

This setup runs Xyne in a Vercel-first mode:

- Frontend is served as a static SPA from `frontend/dist`.
- Backend calls are handled by `api/v1/[...path].ts` as Vercel serverless functions.
- Model inference is direct from serverless to OpenAI (no Bun server, no Vespa, no Docker).

## What works in this mode

- Auth bootstrap for UI routing (`/api/v1/me`)
- Chat streaming endpoint (`/api/v1/message/create`) via SSE
- Retry endpoint compatibility (`/api/v1/message/retry`)
- Basic chat history/bookmark/rename/delete in in-memory serverless state
- Models endpoint (`/api/v1/chat/models`)

## What is limited in this mode

- Data will not persist across cold starts (in-memory state)
- Advanced integrations and infra-dependent features are stubbed
- Full enterprise backend features (queues, Vespa, workers) are not active

## Vercel project settings

Vercel uses `vercel.json` at repo root:

- Build command: `cd frontend && npm install --legacy-peer-deps && npm run build`
- Output directory: `frontend/dist`
- Serverless function: `api/v1/[...path].ts`

## Required environment variables

Set these in Vercel Project Settings -> Environment Variables:

```dotenv
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_MODEL=gpt-4o-mini
DIRECT_MODEL_LABEL=GPT-4o Mini
DIRECT_SYSTEM_PROMPT=You are Xyne running in Vercel Direct Mode. Be concise and helpful.
DIRECT_USER_EMAIL=admin@yourcompany.com
DIRECT_USER_NAME=Admin
```

## Deploy

1. Push changes to your repo.
2. Import project in Vercel.
3. Set env variables listed above.
4. Deploy.

## Health check

- `GET /api/v1/health`
