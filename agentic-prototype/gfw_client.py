import requests
import json
from config import GFW_TOKEN, BASE_URL

class GFWClient:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {GFW_TOKEN}"
        }

    def request(self, endpoint, params=None, body=None):
        if body:
            print("POST body:", body)  
            r = requests.post(
                f"{BASE_URL}/{endpoint}",
                headers={**self.headers, "Content-Type": "application/json"},
                params=params, 
                json=body
            )
        else:
            r = requests.get(
                f"{BASE_URL}/{endpoint}",
                headers=self.headers,
                params=params
            )

        if not r.ok:
            print("API error response:", r.text)
        r.raise_for_status()
        return r.json()

    def get_fishing_events(self, start_date, end_date, region_id=5668):
        params = {
            "datasets[0]": "public-global-fishing-effort:latest",
            "date-range": f"{start_date},{end_date}",
            "spatial-resolution": "LOW",
            "temporal-resolution": "DAILY",
            "group-by": "VESSEL_ID",
            "format": "JSON",
            "region-id": region_id,
            "region-dataset": "public-eez-areas",
        }
        return self.request("4wings/report", params=params)
        