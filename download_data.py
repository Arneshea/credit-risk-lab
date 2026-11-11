"""Download the two public credit datasets used in this project.

Taiwan  : "Default of Credit Card Clients" (UCI ML repository), CSV mirror on GitHub
German  : "Statlog (German Credit Data)" (UCI ML repository), CSV mirror on GitHub
Originals: https://archive.ics.uci.edu/dataset/350  and  https://archive.ics.uci.edu/dataset/144
"""
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
FILES = {
    "taiwan_credit.csv": "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/credit.csv",
    "german_credit.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv",
}

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dest = DATA_DIR / name
        if dest.exists():
            print(f"{name}: already present")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved {dest} ({dest.stat().st_size/1024:.0f} KB)")
