import socket
import struct

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

TOKENS_IN = 1
TOKENS_IN = 1
TOKEN_OUT = 2
END_STREAM = 3
ERROR_MSG = 4

client_socket.connect(("127.0.0.1", 42069))

tokens = [12, 12, 123, 123, 234, 46, 567, 678]

# Payload:
# [token_count:u32]
# [tokens:u32[token_count]]

payload = (
    struct.pack("!I", len(tokens)) +
    struct.pack(f"!{len(tokens)}I", *tokens)
)

# Frame:
# [length:u32]
# [type:u8]
# [payload]

packet = (
    struct.pack("!I", 1 + len(payload)) +   # frame length
    struct.pack("!B", TOKENS_IN) +          # message type
    payload
)

client_socket.sendall(packet)
