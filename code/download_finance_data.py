import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "real" / "fred_sp500"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
CSV_PATH = DATA_DIR / "sp500_daily_fred.csv"


def main():
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 1000:
        print(f"Using existing {CSV_PATH}")
        return

    print(f"Downloading {URL}")
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        CSV_PATH.write_bytes(response.read())
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
