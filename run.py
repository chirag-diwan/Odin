from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import socket
import json

tokenizer_path = hf_hub_download(
    repo_id="Qwen/Qwen2-7B-Instruct",
    filename="tokenizer.json"
)

print(f"Using tokenizer path: {tokenizer_path}")

tokenizer = Tokenizer.from_file(tokenizer_path)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(("127.0.0.1", 42069))

while True:
    text = input("> ")

    if not text:
        continue

    encoded = tokenizer.encode(text)
    token_ids = encoded.ids

    payload = json.dumps(token_ids).encode("utf-8")
    client_socket.sendall(payload)
    data = client_socket.recv(4096)

    if not data:
        continue

    received_ids = json.loads(data.decode("utf-8"))
    decoded = tokenizer.decode(received_ids)
    print(decoded)
