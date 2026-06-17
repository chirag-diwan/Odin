import socket
import struct
import time

SOCKET_PATH = "/tmp/odin0000.socket"

def hit_and_forget_length_only():
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCKET_PATH)
    sock.recv(4) # Consume ID

    # Send a massive length prefix, but no payload
    fake_length = struct.pack("I", 4294967295) # UINT32_MAX
    sock.sendall(fake_length)
    # Drop connection immediately
    sock.close()
    print("Hit and forget (Length only) executed.")

def hit_and_forget_partial_payload():
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCKET_PATH)
    sock.recv(4)

    payload = b"UNFINISHED_PROMPT"
    length_prefix = struct.pack("I", 5000) # Claiming 5000 bytes
    sock.sendall(length_prefix + payload)

    # Sleep to keep the socket alive but doing nothing, tying up server resources
    time.sleep(5)
    sock.close()
    print("Hit and forget (Partial payload) executed.")

hit_and_forget_length_only()
hit_and_forget_partial_payload()
