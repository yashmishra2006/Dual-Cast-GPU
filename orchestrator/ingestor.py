# orchestrator/ingestor.py
import cv2
import zmq
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import importlib.util

# Dynamic import
spec_zmq = Path("core/zmq_utils.py").resolve()
spec = importlib.util.spec_from_file_location("zmq_utils", spec_zmq)
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
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.INFO
        )
        self.logger = logging.getLogger("Ingestor")

        self.source = input("🎥 Enter video source (0 for webcam, or file path): ")
        self.heartbeat_interval = int(input("🔁 Enter heartbeat interval (in frames): "))

        self.cap = self._init_video_capture(self.source)
        self.context = zmq.Context()
        self.classifier_socket = self._init_classifier_socket()
        self.phase_socket = None
        self.current_phase = None
        self.frame_count = 0

    def _init_video_capture(self, src):
        self.logger.info("Initializing video capture...")
        cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)

        if not cap.isOpened():
            self.logger.error(f"❌ Failed to open video source: {src}")
            exit(1)

        self.logger.info(f"✅ Video source opened: {src}")
        return cap

    def _init_classifier_socket(self):
        self.logger.info("Setting up classifier socket...")
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
            phase = result.get("phase")
            confidence = result.get("confidence")
            self.logger.info(f"🧠 Phase classified as '{phase}' (confidence: {confidence:.2f})")
            return phase, confidence
        except zmq.Again:
            self.logger.warning("⚠️ Classifier did not respond in time.")
            return self.current_phase, 0.0

    def _switch_phase(self, new_phase):
        if self.phase_socket:
            self.phase_socket.close()
            self.logger.info("🔌 Closed previous phase socket.")
        self.phase_socket = self.context.socket(zmq.PUSH)
        self.phase_socket.connect(f"tcp://localhost:{PHASE_PORTS[new_phase]}")
        self.logger.info(f"🔀 Switched to phase: {new_phase}")
        self.current_phase = new_phase

    def run(self):
        self.logger.info("🚀 Ingestor started")
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("📴 End of stream or read error.")
                    break

                self.frame_count += 1

                if self.frame_count == 1 or self.frame_count % self.heartbeat_interval == 0:
                    self.logger.debug(f"💓 Heartbeat check at frame {self.frame_count}")
                    phase, conf = self._classify_phase(frame)
                    if phase and phase != self.current_phase:
                        self._switch_phase(phase)

                if self.phase_socket:
                    header = {
                        "frame_id": self.frame_count,
                        "timestamp": datetime.now().isoformat(),
                        "shape": list(frame.shape)
                    }
                    send_frame(self.phase_socket, frame, header)
                    self.logger.info(f"📤 Sent frame {self.frame_count} to {self.current_phase}")

            except KeyboardInterrupt:
                self.logger.info("⛔ Interrupted by user.")
                break
            except Exception as e:
                self.logger.exception(f"💥 Ingestor error: {e}")
                continue

        self.cleanup()

    def cleanup(self):
        self.logger.info("🧹 Cleaning up resources...")
        self.cap.release()
        if self.phase_socket:
            self.phase_socket.close()
        if self.classifier_socket:
            self.classifier_socket.close()
        self.context.term()
        self.logger.info("✅ Cleanup complete.")

if __name__ == "__main__":
    ingestor = Ingestor()
    ingestor.run()
