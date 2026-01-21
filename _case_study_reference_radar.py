from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable, Optional
from urllib.parse import urljoin

import requests


def iso_z(dt: datetime) -> str:
    """Format datetime as ...Z (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    # keep milliseconds like your example
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def download_telcosense_images(
    *,
    base_url: str = "https://telcosense.cz",
    list_endpoint: str = "/api/merge1h/list",  # or "/api/maxz/list"
    start: datetime,
    end: datetime,
    out_dir: str = "./downloads",
    min_rain_score: Optional[float] = None,
    overwrite: bool = False,
    timeout_s: int = 60,
):
    """
    Downloads images returned by telcosense list endpoint.

    The list response items are expected like:
      {"rain_score": 0.285, "timestamp": "...+00:00", "url": "/maxz/FILE.png"}

    Images are fetched from: base_url + "/api" + item["url"]
    Example item["url"] is "/maxz/..." -> final becomes "https://telcosense.cz/api/maxz/..."
    """
    os.makedirs(out_dir, exist_ok=True)

    params = {"start": iso_z(start), "end": iso_z(end)}

    with requests.Session() as s:
        list_url = urljoin(base_url, list_endpoint)
        r = s.get(list_url, params=params, timeout=timeout_s)
        r.raise_for_status()
        items = r.json()

        if not isinstance(items, list):
            raise ValueError(f"Unexpected list response type: {type(items)}")

        # optional filter by rain_score
        if min_rain_score is not None:
            items = [
                it for it in items if float(it.get("rain_score", 0.0)) >= min_rain_score
            ]

        print(f"List returned {len(items)} items")

        for it in items:
            rel = it.get("url")
            if not rel:
                continue

            # item["url"] looks like "/maxz/FILE.png"
            # actual fetch URL is "/api" + rel
            fetch_url = urljoin(base_url, "/api" + rel)

            # build local filename from the URL tail
            filename = rel.split("/")[-1]
            local_path = os.path.join(out_dir, filename)

            if (
                (not overwrite)
                and os.path.exists(local_path)
                and os.path.getsize(local_path) > 0
            ):
                continue

            # stream download
            with s.get(fetch_url, stream=True, timeout=timeout_s) as img_res:
                img_res.raise_for_status()
                tmp = local_path + ".part"
                with open(tmp, "wb") as f:
                    for chunk in img_res.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, local_path)

        return out_dir


if __name__ == "__main__":
    # start = datetime(2025, 7, 6, 16, 0, tzinfo=timezone.utc)
    # end = datetime(2025, 7, 6, 23, 0, tzinfo=timezone.utc)
    start = datetime(2025, 8, 29, 18, 0, tzinfo=timezone.utc)
    end = datetime(2025, 8, 30, 3, 0, tzinfo=timezone.utc)
    download_telcosense_images(
        base_url="https://telcosense.cz",
        list_endpoint="/api/maxz/list",
        start=start,
        end=end,
        out_dir="./case_study_runs/prague_1/maxz",
        min_rain_score=None,  # e.g. 0.3 if you want
    )
