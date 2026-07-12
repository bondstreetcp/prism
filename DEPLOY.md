# Deploying the risk-report web app

The app (`app.py`) is a Streamlit UI over the same pipeline the CLI uses:
upload a position CSV → it prices the book and returns the two-page PDF.

## Run it locally first

```
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501. No password is required locally (the gate
only activates when `APP_PASSWORD` is set). This is the zero-hosting option —
nothing leaves your machine.

## Hosting for you + a few colleagues (private)

All good options run the **same container** (`Dockerfile`). Pick by how much
you want to manage. Two things every choice needs:

1. **A password.** Set an `APP_PASSWORD` environment variable / secret. The
   app shows a login gate whenever it is set.
2. **A persistent disk** mounted at `/data` so the market-data cache and daily
   snapshots survive restarts. Without it every run starts cold (a few minutes)
   and attribution history never accumulates. The Dockerfile points
   `RISK_CACHE_DIR` / `RISK_OUT_DIR` at `/data`.

### Option A — Render or Railway (easiest turnkey, ~$5–10/mo)

1. Push this folder to a **private** GitHub repo.
2. Render: New → **Web Service** → connect the repo → it detects the
   `Dockerfile`. Railway: New Project → Deploy from repo.
3. Add a **persistent disk** mounted at `/data` (Render: "Disks"; Railway:
   "Volumes").
4. Add env var `APP_PASSWORD` = something only your team knows.
5. Deploy. You get an HTTPS URL; share it + the password with colleagues.

### Option B — Fly.io (global, scale-to-zero, cheap)

```
fly launch            # detects the Dockerfile; say no to DB
fly volume create data --size 1
# in fly.toml: mount the volume at /data, set internal_port = 8501
fly secrets set APP_PASSWORD=your-password
fly deploy
```

### Option C — Small VPS + Tailscale (most private, ~$4–6/mo)

Best if you want **zero public exposure**: the app is only reachable by
devices on your Tailscale network.

1. Cheap VPS (Hetzner CX22, DigitalOcean droplet, etc.).
2. Install Docker and [Tailscale](https://tailscale.com); `tailscale up`.
3. `docker build -t riskapp . && docker run -d --restart unless-stopped \
   -p 8501:8501 -e APP_PASSWORD=your-password \
   -v /srv/riskdata:/data riskapp`
4. Reach it at `http://<tailscale-hostname>:8501` — no public port, so the
   Yahoo-data ToS concern effectively disappears (it is a private tool).

### Option D — Streamlit Community Cloud (free, with caveats)

Free and one-click from a private GitHub repo, but the disk is **ephemeral**
(cache/snapshots reset on every reboot → cold starts, no attribution history)
and it is meant for lighter apps. Fine for occasional use; add `APP_PASSWORD`
via the app's **Secrets** settings. For anything regular, prefer A–C.

## Important caveats (read before sharing a URL)

- **Market data / terms of service.** The tool uses `yfinance`, which pulls
  Yahoo Finance. Yahoo's terms allow personal use but not redistributing their
  data through a public service. Keeping the app private (password + Tailscale/
  internal network) stays within normal internal-tool practice. If you ever
  open it to the public, move to a licensed feed (Polygon, Tiingo, ~$30–100/mo)
  — that is a code change in `riskreport/marketdata.py`, not just hosting.
- **Not investment advice.** The app states this; keep that disclaimer.
- **Latency & concurrency.** A first run for a new book fetches hundreds of
  tickers and option chains — several minutes. Runs are cached afterward. A
  small instance handles a few users doing occasional runs; it is **not** built
  for many simultaneous heavy jobs (each blocks its session). If that becomes a
  need, move the pipeline behind a job queue.
- **Cost.** Options A–C are a few dollars a month. Option D is free but limited.

## Automating the daily snapshot (optional)

Attribution and history improve as daily snapshots accrue. On any always-on
host, add a cron/scheduled job that runs the CLI against the latest export:

```
python run_report.py /data/exports/latest.csv --cache /data/cache --out /data/reports
```
