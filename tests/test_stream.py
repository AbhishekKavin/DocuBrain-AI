import requests
import json

url = "http://127.0.0.1:8000/ask-stream"
data = {"question": "Tell me about Abhishek's experience at Telstra."}

# The 'stream=True' flag tells the client not to wait for the end
with requests.post(url, json=data, stream=True) as response:
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            print(chunk.decode("utf-8"), end="", flush=True)