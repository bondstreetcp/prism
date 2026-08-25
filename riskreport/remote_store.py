"""Optional off-box mirror of the snapshot store.

On hosts with an ephemeral filesystem (e.g. Streamlit Community Cloud) the
``snapshots/`` folder is wiped on every reboot, so the Trends and attribution
history never accumulates. This module mirrors that folder to a *private*
GitHub repo via the REST Contents API, so history survives restarts.

It uses ``requests`` (already a dependency) and no git CLI. Point it at a
repo you own that is SEPARATE from the app's own repo, so writing snapshots
does not trigger the app to redeploy.

Configured via environment variables or ``st.secrets``:

    SNAPSHOT_REPO    = "owner/name"   private repo that holds the history
    SNAPSHOT_TOKEN   = <PAT>          fine-grained token, Contents: read+write
    SNAPSHOT_BRANCH  = "main"         optional, defaults to "main"

When these are unset every function is a silent no-op, so local runs and the
Synology deploy (which have a real persistent disk) are completely unaffected.
Nothing here ever raises into the caller — a flaky network must not break a
report run — so callers can invoke pull()/push_date() unconditionally.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Callable

import requests

_API = "https://api.github.com"
_TIMEOUT = 20  # seconds per request


def _secret(name: str) -> str | None:
    """Read a setting from the environment, then st.secrets if present."""
    import os

    val = os.environ.get(name)
    if val:
        return val
    try:  # streamlit is optional at import time (CLI paths never call this)
        import streamlit as st

        return st.secrets.get(name)  # type: ignore[no-any-return]
    except Exception:
        return None


def _config() -> tuple[str, str, str] | None:
    repo = _secret("SNAPSHOT_REPO")
    token = _secret("SNAPSHOT_TOKEN")
    if not repo or not token:
        return None
    branch = _secret("SNAPSHOT_BRANCH") or "main"
    return repo.strip("/"), token, branch


def enabled() -> bool:
    return _config() is not None


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def pull(local_dir: str | Path, log: Callable[[str], None] | None = None) -> int:
    """Download every mirrored snapshot file into ``local_dir``.

    Returns the number of files written (0 when disabled, empty, or on error).
    Existing local files are overwritten so the remote is authoritative.
    """
    cfg = _config()
    if not cfg:
        return 0
    repo, token, branch = cfg
    base = Path(local_dir)
    headers = _headers(token)
    written = 0
    try:
        tree = requests.get(
            f"{_API}/repos/{repo}/git/trees/{branch}",
            params={"recursive": "1"},
            headers=headers,
            timeout=_TIMEOUT,
        )
        if tree.status_code in (404, 409):  # empty repo / no such branch yet
            return 0
        tree.raise_for_status()
        blobs = [
            n for n in tree.json().get("tree", [])
            if n.get("type") == "blob" and n.get("path", "").endswith(
                ("summary.json", "positions.csv")
            )
        ]
        for node in blobs:
            blob = requests.get(
                f"{_API}/repos/{repo}/git/blobs/{node['sha']}",
                headers=headers,
                timeout=_TIMEOUT,
            )
            blob.raise_for_status()
            payload = blob.json()
            if payload.get("encoding") != "base64":
                continue
            content = base64.b64decode(payload["content"])
            dest = base / node["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            written += 1
    except Exception as exc:  # never break the app on a sync hiccup
        if log:
            log(f"Snapshot pull skipped: {exc}")
        return written
    if log and written:
        log(f"Pulled {written} snapshot file(s) from {repo}")
    return written


def _put_file(repo: str, token: str, branch: str, path: str, data: bytes) -> bool:
    headers = _headers(token)
    url = f"{_API}/repos/{repo}/contents/{path}"
    # An update needs the current blob sha; a first write must omit it.
    sha = None
    existing = requests.get(
        url, params={"ref": branch}, headers=headers, timeout=_TIMEOUT
    )
    if existing.status_code == 200:
        sha = existing.json().get("sha")
    body = {
        "message": f"snapshot: {path}",
        "content": base64.b64encode(data).decode("ascii"),
        "branch": branch,
    }
    if sha:
        body["sha"] = sha
    resp = requests.put(url, headers=headers, json=body, timeout=_TIMEOUT)
    resp.raise_for_status()
    return True


def push_date(
    local_dir: str | Path, asof, log: Callable[[str], None] | None = None
) -> int:
    """Upload the snapshot files for a single as-of date to the mirror.

    ``asof`` may be a date or an ISO string. Returns the number of files
    pushed (0 when disabled or on error).
    """
    cfg = _config()
    if not cfg:
        return 0
    repo, token, branch = cfg
    day = asof.isoformat() if hasattr(asof, "isoformat") else str(asof)
    src = Path(local_dir) / day
    if not src.is_dir():
        return 0
    pushed = 0
    try:
        for f in sorted(src.iterdir()):
            if f.name not in ("summary.json", "positions.csv"):
                continue
            _put_file(repo, token, branch, f"{day}/{f.name}", f.read_bytes())
            pushed += 1
    except Exception as exc:  # a failed mirror must not fail the run
        if log:
            log(f"Snapshot push skipped: {exc}")
        return pushed
    if log and pushed:
        log(f"Mirrored {pushed} snapshot file(s) to {repo}")
    return pushed
