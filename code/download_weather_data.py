import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "real" / "open_meteo_dallas"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    params = {
        "latitude": "32.7767",
        "longitude": "-96.7970",
        "start_date": "2021-01-01",
        "end_date": "2023-12-31",
        "hourly": "temperature_2m",
        "timezone": "America/Chicago",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    raw_path = DATA_DIR / "open_meteo_dallas_temperature.json"
    csv_path = DATA_DIR / "open_meteo_dallas_hourly_temperature.csv"

    if not raw_path.exists():
        print(f"Downloading {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            raw_path.write_bytes(response.read())
    else:
        print(f"Using existing {raw_path}")

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    hourly = payload["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
    }).dropna()
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} with {len(df)} hourly observations")


if __name__ == "__main__":
    main()
