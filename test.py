import requests
response = requests.get("https://www.reddit.com", timeout=10)
print(response.status_code)
