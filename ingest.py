import os
import requests

DATA_URL = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AB_NYC_2019.csv"
SAVE_PATH = "data/raw/listings.csv"

os.makedirs("data/raw", exist_ok=True)
print("Downloading rental dataset...")

response = requests.get(DATA_URL, timeout=30)
response.raise_for_status()

with open(SAVE_PATH, "wb") as f:
    f.write(response.content)

print(f"Data saved successfully to {SAVE_PATH}!")
