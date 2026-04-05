# Deploying Xyne on Render

This repository now includes a Render Blueprint at [render.yaml](../../../render.yaml).

## What gets deployed

- `xyne-db` (Render Postgres)
- `xyne-vespa` (private Docker service)
- `xyne-app` (public Docker web service)

## Steps

1. Push your branch with [render.yaml](../../../render.yaml).
2. In Render, click **New +** -> **Blueprint**.
3. Select this repository/branch.
4. Render will create the three resources.
5. After first deploy, set required env vars for `xyne-app` that are marked `sync: false`:
   - `HOST`
   - `GOOGLE_REDIRECT_URI`
   - `GOOGLE_CLIENT_ID`
   - `GOOGLE_CLIENT_SECRET`
   - `ENCRYPTION_KEY` (must be base64-encoded 32-byte key)
   - `SERVICE_ACCOUNT_ENCRYPTION_KEY` (must be base64-encoded 32-byte key)
   - `JWT_SECRET`
   - `ACCESS_TOKEN_SECRET`
   - `REFRESH_TOKEN_SECRET`
   - `USER_SECRET`
6. Redeploy `xyne-app` after updating those values.

## OAuth values

For Google OAuth, set values similar to:

- `HOST=https://<your-render-domain>`
- `GOOGLE_REDIRECT_URI=https://<your-render-domain>/v1/auth/callback`

In Google Cloud Console, ensure both are allowed:

- Authorized JavaScript origin: `https://<your-render-domain>`
- Authorized redirect URI: `https://<your-render-domain>/v1/auth/callback`

## Notes

- First startup can take longer because the app may run initial migrations and Vespa deployment.
- `xyne-app` health check is `GET /health`.
- Vespa is resource-heavy; move to higher Render plans for production workloads.

Example base64 32-byte keys for local testing:

- `ENCRYPTION_KEY=MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=`
- `SERVICE_ACCOUNT_ENCRYPTION_KEY=ZmVkY2JhOTg3NjU0MzIxMGZlZGNiYTk4NzY1NDMyMTA=`
