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
   and Trends/attribution history never accumulates. The Dockerfile points
   `RISK_CACHE_DIR` / `RISK_OUT_DIR` / `RISK_SNAP_DIR` at `/data`.

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
via the app's **Secrets** settings. For anything regular, prefer A–C or E.

### Option E — Synology NAS (self-hosted, private, persistent) ⭐

Best of both worlds if you already own a NAS: **your position files never leave
your hardware**, and the `/data` volume lives on the NAS so cache + snapshots
persist (Trends and attribution build up run over run). Uses `docker-compose.yml`.

**Requirements.** An **x86-64** Synology (Intel/AMD — the `+`, `play`, `xs`,
and most `value` models: DS220+, DS224+, DS423+, DS920+, DS923+, DS1522+, etc.)
running **Container Manager** (DSM 7.2+; older DSM calls it "Docker"). ARM models
(`j`/`se`, e.g. DS120j, DS223) **can't** run this — check yours under Package
Center. 4 GB+ RAM recommended (the factor fit on the ~1,000-line book is
CPU/RAM-heavy; first run is slow on a NAS CPU, then cached).

**Steps.**
1. Install **Container Manager** from Package Center.
2. Copy this repo folder to a shared folder on the NAS, e.g.
   `/volume1/docker/portfolio-risk` (File Station, or `git clone` over SSH).
   It gitignores your CSVs/PDFs, so copy your position files in separately or
   just upload them through the web UI.
3. Edit `docker-compose.yml` → set `APP_PASSWORD` to a real password (and
   `LLM_API_KEY` if you want the AI tab).
4. Container Manager → **Project** → **Create** → Path = that folder → it reads
   `docker-compose.yml`, **builds** the image, and starts it. (First build pulls
   Python + wheels — a few minutes.)
5. Open **`http://<NAS-LAN-IP>:8501`**, log in, upload your CSV(s), Generate.
   That's the "upload a file → get an updated page back" flow — now on your NAS.

**Reaching it from outside the house — pick one:**
- **Most private (recommended for holdings): keep it LAN-only + VPN.** Don't
  expose port 8501. Install the **Tailscale** package (or Synology **VPN
  Server**) on the NAS; connect your phone/laptop to it and use the LAN URL from
  anywhere. Nothing is on the public internet.
- **Convenience: Synology reverse proxy + HTTPS.** Control Panel → Login Portal
  → Advanced → **Reverse Proxy** → route `https://risk.<you>.synology.me` (set up
  a DDNS hostname + Let's Encrypt cert under Security → Certificate) to
  `localhost:8501`. Then it's a normal HTTPS URL. Keep `APP_PASSWORD` on, and
  ideally restrict source IPs / enable the firewall + 2-step login. Note the
  Yahoo-data ToS caveat below applies once it's publicly reachable.

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
