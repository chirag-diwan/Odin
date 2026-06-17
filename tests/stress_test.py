import socket
import threading
import struct
import time

SOCKET_PATH = "/tmp/odin0000.socket"
MAX_CLIENTS = 30

def stress_client(client_id):
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)

        # Read the client ID sent by server upon connect
        server_id = struct.unpack("I", sock.recv(4))[0]

        payload = f"STRESS_TEST_PROMPT_FROM_CLIENT_{client_id}".encode('utf-8')
        length_prefix = struct.pack("I", len(payload))

        sock.sendall(length_prefix + payload)

        # Await response
        response = sock.recv(1024)
        print(f"Client {client_id} (Server ID {server_id}) received: {response.decode('utf-8', errors='ignore')}")
        sock.close()
    except Exception as e:
        print(f"Client {client_id} failed: {e}")

threads = []
for i in range(MAX_CLIENTS + 5): # Exceed the 30 client limit to test rejection
    t = threading.Thread(target=stress_client, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
