import requests
from typing import Any, Dict


def fetch_data(url: str) -> Dict[str, Any]:
    response = requests.get(url)
    response.raise_for_status()  # Lève une exception pour les erreurs HTTP
    return response.json()


def post_data(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Lève une exception pour les erreurs HTTP
    return response.json()
