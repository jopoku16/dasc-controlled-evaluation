import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "real" / "uci_household_power"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
ZIP_PATH = DATA_DIR / "individual_household_power_consumption.zip"


def main():
    if not ZIP_PATH.exists():
        print(f"Downloading {URL}")
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            ZIP_PATH.write_bytes(response.read())
    else:
        print(f"Using existing {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)

    print(f"Data ready in {DATA_DIR}")


if __name__ == "__main__":
    main()
