import requests
import json
import os

url = 'http://localhost:8080/detect'
files = {
    'image_before': open('test_images/real_before.png', 'rb'),
    'image_after': open('test_images/real_after.png', 'rb')
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
finally:
    files['image_before'].close()
    files['image_after'].close()
