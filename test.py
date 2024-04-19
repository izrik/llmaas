import json

import requests

with open('test.json') as f:
    body = json.load(f)

resp = requests.get('http://localhost:5000/chat', json=body)
resp.raise_for_status()
print(resp)
print(resp.json())
