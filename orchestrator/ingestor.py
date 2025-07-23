# ingestor.py
import cv2
import zmq
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import importlib.util
import gc

# Load zmq_utils
spec = importlib.util.spec_from_file_location("zmq_utils", Path("core/zmq_utils.py"))
zmq_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zmq_utils)
send_frame = zmq_utils.send_frame

PHASE_PORTS = {
    "AGENT_PHASE": 5555,
    "BUY_PHASE": 5556,
    "GAME_PHASE": 5557
}

class Ingestor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Ingestor")

        self.source = input("🎥 Video source (0 for webcam or path): ")
        self.heartbeat_interval = int(input("🔁 Heartbeat interval (frames): "))
        self.skip_rate = 3  # Send 1 in 3 frames

        self.cap = self._init_video_capture(self.source)
        self.context = zmq.Context()
        self.classifier_socket = self._init_classifier_socket()
        self.phase_socket = None
        self.current_phase = None
        self.frame_count = 0

    def _init_video_capture(self, src):
        cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"❌ Could not open source: {src}")
        return cap

    def _init_classifier_socket(self):
        sock = self.context.socket(zmq.REQ)
        sock.RCVTIMEO = 2000
        sock.connect("tcp://localhost:5554")
        return sock

    def _classify_phase(self, frame):
        header = {"shape": list(frame.shape)}
        self.classifier_socket.send_string(json.dumps([header]), zmq.SNDMORE)
        self.classifier_socket.send(frame.tobytes())

        try:
            reply = self.classifier_socket.recv_string()
            result = json.loads(reply)
            return result.get("phase"), result.get("confidence", 0.0)
        except zmq.Again:
            self.logger.warning("⚠️ Classifier timeout")
            return self.current_phase, 0.0

    def _switch_phase(self, new_phase):
        if self.phase_socket:
            self.phase_socket.close()
        self.phase_socket = self.context.socket(zmq.PUSH)
        self.phase_socket.connect(f"tcp://localhost:{PHASE_PORTS[new_phase]}")
        self.current_phase = new_phase

    def run(self):
        self.logger.info("🚀 Ingestor started")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("📴 End of stream or read error.")
                    break

                self.frame_count += 1

                if self.frame_count == 1 or self.frame_count % self.heartbeat_interval == 0:
                    phase, conf = self._classify_phase(frame)
                    if phase and phase != self.current_phase:
                        self._switch_phase(phase)

                # Send only 1 in every 3 frames
                if self.phase_socket and self.frame_count % self.skip_rate == 0:
                    header = {
                        "frame_id": self.frame_count,
                        "timestamp": datetime.now().isoformat(),
                        "shape": list(frame.shape)
                    }
                    send_frame(self.phase_socket, frame, header)
                    self.logger.info(f"📤 Frame {self.frame_count} sent to {self.current_phase}")

        except KeyboardInterrupt:
            self.logger.info("⛔ Interrupted.")
        finally:
            self.cleanup()

    def cleanup(self):
        self.logger.info("🧹 Cleaning up...")
        self.cap.release()
        if self.phase_socket:
            self.phase_socket.close()
        self.classifier_socket.close()
        self.context.term()
        gc.collect()
        self.logger.info("✅ Cleanup done.")

if __name__ == "__main__":
    Ingestor().run()
