# Deploying Portfolio Visualiser (internal hosting)

Audience: whoever runs this for the advisor team (infra / platform). Developer
orientation lives in [CLAUDE.md](CLAUDE.md); this file is only about running it.

> **Status:** the `Dockerfile` in this repo has not been build-tested yet
> (it was written on a machine without Docker). First thing to do: build it and
> smoke-test the three pages and a PDF export (see [Smoke test](#smoke-test)).

## Why internal hosting

The Client Portfolio Visualizer and the Advisory Tool look clients up through
Metabase at `https://metabase.internal.syfe.com`, which is only reachable on the
Syfe network/VPN. Streamlit Community Cloud cannot reach it, so a host inside
the network is required for the client lookup. The app also handles client
holdings, so access should be restricted to Syfe staff.

## What the container needs

| Need | Detail |
|---|---|
| Image | `docker build -t portfolio-visualiser .` (Python 3.11, port **8501**) |
| Entry point | `streamlit run Advisory_Tool.py`; the Builder and Client PV pages are discovered from `pages/` |
| Memory | Start with 2 GiB; Builder optimisation and the Excel/PDF export are the heavy paths |
| Writable app dir | The app writes temp PDFs/charts to its working directory (`/app`), and `Data/saved_portfolios.json` if Supabase is not configured. The image runs as non-root user `app` (uid 1000) with `/app` writable. If you set `readOnlyRootFilesystem`, mount an `emptyDir` at `/app` or the PDF export will fail |
| Health check | `GET /_stcore/health` returns `200` (the image's `HEALTHCHECK` uses it; use it for k8s probes too) |
| WebSockets | Streamlit needs them. The ingress / load balancer must allow WebSocket upgrades on `/_stcore/stream` |

### Network egress

| Destination | Used for | Required? |
|---|---|---|
| `metabase.internal.syfe.com` (internal) | Client lookup (cards 1642 / 4905 / 1470) | Yes, for client lookup |
| `generativelanguage.googleapis.com` | Gemini: recommender, diversification insights, brokerage-PDF parsing | Yes, for LLM features |
| Yahoo Finance (`yfinance`) | Price history when data source is Yahoo; PDF holding enrichment | Yes, for those features |
| `*.supabase.co` | Shared saved Builder portfolios | Only if `SUPABASE_*` are set |

## Secrets

All secrets are read by `utils/app_secrets.py`: `st.secrets` first, then
environment variables. In a container the simplest route is environment
variables from a Kubernetes `Secret` (or your secret manager). Alternatively,
mount a `secrets.toml` at `/app/.streamlit/secrets.toml`. Template:
[`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example).

| Name | Required | Notes |
|---|---|---|
| `GEMINI_API_KEY` | For LLM features | Use a **team/service** key, not a personal one |
| `METABASE_API_KEY` | For client lookup | Create under a service account in Metabase; the old key was tied to the previous developer |
| `BUILDER_PASSWORD` | To use the Builder page | The page stays **locked** when unset. Pick a strong value |
| `SUPABASE_URL`, `SUPABASE_KEY` | Optional | Without them, saved portfolios go to a local JSON file that is **lost when the container restarts** (unless `Data/` is a persistent volume) |

**Before going live:** the previous Gemini, Metabase and Supabase keys and the
old Builder password are still in git history. Rotate them and use
only new values here. Do not reuse the old ones.

## Access control (important)

Streamlit has no built-in user authentication, and the Builder password is a
single shared secret, not an access-control system. Put the app behind
company SSO at the ingress (for example oauth2-proxy / your existing identity
provider) and restrict it to the advisor group. Treat the app as exposing
client data to anyone who can reach it.

## Run it

```bash
# local container, secrets from a git-ignored env file
printf 'GEMINI_API_KEY=...\nMETABASE_API_KEY=...\nBUILDER_PASSWORD=...\n' > .env
docker build -t portfolio-visualiser .
docker run --rm -p 8501:8501 --env-file .env portfolio-visualiser
# open http://localhost:8501  (Metabase lookup only works if you are on the VPN)
```

Kubernetes sketch (adapt to your standards; names are placeholders):

```yaml
containers:
  - name: portfolio-visualiser
    image: <registry>/portfolio-visualiser:<tag>
    ports: [{containerPort: 8501}]
    envFrom: [{secretRef: {name: portfolio-visualiser-secrets}}]
    readinessProbe: {httpGet: {path: /_stcore/health, port: 8501}, initialDelaySeconds: 20}
    livenessProbe:  {httpGet: {path: /_stcore/health, port: 8501}, initialDelaySeconds: 40}
    resources: {requests: {cpu: 500m, memory: 1Gi}, limits: {memory: 2Gi}}
```

Scaling: Streamlit sessions are in-process and per-websocket. One replica is
simplest. If you run several, enable session affinity on the ingress.

## Smoke test

1. Open `/`: the Advisory Tool loads with no "Inv_Data_Hardcoded.xlsx not found" error (the data file ships in `Data/`).
2. Existing client → Metabase lookup returns a client (confirms VPN reach and `METABASE_API_KEY`).
3. Generate a recommendation (confirms `GEMINI_API_KEY` and outbound access).
4. Generate a PDF report with charts (confirms the Chromium libraries for kaleido installed correctly; if charts are missing, the apt libraries in the `Dockerfile` need adjusting).
5. `/Builder` asks for the password and accepts `BUILDER_PASSWORD`.

## Known limitations

- **Shared temp files.** PDF generation writes fixed filenames (for example `temp_report.pdf`) into the working directory, so two advisors exporting at the same moment can overwrite each other's file. Low risk with a small team; fixing it means using per-session temp paths in `utils/pdf_gen.py`.
- **Reference data ships in the image.** `Data/Inv_Data_Hardcoded.xlsx` is committed (about 8 MB). When it changes, rebuild and redeploy.
- **Dependencies are floor-pinned** (`>=`) in `requirements.txt`, so rebuilds can pick up newer library versions. Consider a lock file once the first deployment is working.

## Alternative: Streamlit Community Cloud

Fine if you give up the Metabase client lookup (CSV upload and PDF upload still
work): connect the repo, set the main file to `Advisory_Tool.py`, paste the
secrets into the app's Secrets settings, and restrict viewers by email. Check
that hosting client data outside Syfe's network is acceptable first.
