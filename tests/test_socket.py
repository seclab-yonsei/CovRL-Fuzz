import socket
import time
from struct import unpack

HOST = '127.0.0.1'
PORT = 1111


def send_finetune_message():
    try:
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect((HOST, PORT))

        print("Connected to server")

        client_sock.sendall(b"finetune")
        print("Sent 'finetune' message")

        data = client_sock.recv(1024)
        if data:
            finetune_result = unpack("<H", data)[0]
            print(f"Received finetune response: {finetune_result}")

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        client_sock.close()
        print("Connection closed")


for _ in range(3):
    send_finetune_message()
    time.sleep(1)
