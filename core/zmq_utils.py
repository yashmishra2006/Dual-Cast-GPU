# core/zmq_utils.py
import zmq
import json
import numpy as np

def send_frame(socket, frame: np.ndarray, metadata: dict):
    header = json.dumps(metadata)
    socket.send_string(header, zmq.SNDMORE)
    socket.send(frame.tobytes(), zmq.NOBLOCK)

def recv_frame(socket):
    header = json.loads(socket.recv_string(zmq.NOBLOCK))
    frame_bytes = socket.recv(zmq.NOBLOCK)
    shape = tuple(header['shape'])
    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(shape)
    return header, frame