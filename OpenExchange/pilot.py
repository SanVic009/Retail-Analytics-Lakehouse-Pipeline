import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_exchange_rates():
    API_KEY = os.getenv("APP_ID")
    url = f"https://openexchangerates.org/api/latest.json?app_id={API_KEY}"

    response = requests.get(url)
    data = response.json()
    return data

# data = get_exchange_rates()
# print(data.keys())
# print(data['base'])
# print(list(data['rates'].items()))
