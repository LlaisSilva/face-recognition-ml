import requests

def send_event(event, api_url):
    response = requests.post(api_url, json=event, timeout=5)

    response.raise_for_status()
    return response.json()