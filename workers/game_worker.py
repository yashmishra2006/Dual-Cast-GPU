import logging
from datetime import datetime
from pathlib import Path
import sys
import time
import importlib.util
import cv2
import numpy as np
import zmq
from ultralytics import YOLO
from rich.logging import RichHandler
from rich.console import Console
import GPUtil
import torch

# Setup rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("GameWorker")
console = Console()

# === Dynamically import core modules ===
def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

device_module = import_from_path("device", Path("core/device.py").resolve())
zmq_utils = import_from_path("zmq_utils", Path("core/zmq_utils.py").resolve())

DeviceManager = device_module.DeviceManager
recv_frame = zmq_utils.recv_frame
send_frame = zmq_utils.send_frame

EVENT_CLASSES = ["KILL_FEED", "SPIKE_PLANT", "SPIKE_DEFUSE", "ROUND_END"]

def log_gpu_usage(prefix=""):
    """Logs the GPU memory usage."""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            used = gpus[0].memoryUsed
            total = gpus[0].memoryTotal
            logger.info(f"{prefix}GPU Usage: {used:.1f} / {total:.1f} MB")
    except Exception as e:
        logger.warning(f"GPU usage logging failed: {e}")

class GameWorker:
    def __init__(self, input_port=5557, output_port=5561):
        self.logger = logger
        self.device = DeviceManager().get()
        console.log(f"[cyan]Using device: {self.device}[/cyan]")

        base_path = Path(__file__).resolve().parent.parent / "models"
        self.main_model_path = base_path / "gamephase.pt"
        self.kill_model_path = base_path / "gamephasekill.pt"

        self.logger.info(f"Loading main model from: {self.main_model_path}")
        self.main_model = YOLO(str(self.main_model_path)).to(self.device)

        self.logger.info(f"Loading kill model from: {self.kill_model_path}")
        self.kill_model = YOLO(str(self.kill_model_path)).to(self.device)

        self.context = zmq.Context()
        self.input_socket = self.context.socket(zmq.PULL)
        self.input_socket.RCVTIMEO = 1000
        self.input_socket.bind(f"tcp://*:{input_port}")
        console.log(f"[green]Listening for frames on tcp://*:{input_port}[/green]")

        self.output_socket = self.context.socket(zmq.PUSH)
        self.output_socket.connect(f"tcp://localhost:{output_port}")
        console.log(f"[green]Output sending to tcp://localhost:{output_port}[/green]")

    def detect_events(self, frame):
        try:
            results_main = self.main_model(frame, conf=0.4, verbose=False)
            kill_boxes = []
            events = []

            for box in results_main[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                event_type = self.main_model.names[class_id]

                if event_type == "KILL_FEED":
                    kill_boxes.append((x1, y1, x2, y2))
                elif event_type in EVENT_CLASSES:
                    events.append({
                        "type": event_type,
                        "confidence": round(confidence, 4),
                        "bbox": [x1, y1, x2, y2]
                    })

            for (x1, y1, x2, y2) in kill_boxes:
                kill_crop = frame[y1:y2, x1:x2]
                results_kill = self.kill_model(kill_crop, conf=0.4, verbose=False)

                for box in results_kill[0].boxes:
                    kx1, ky1, kx2, ky2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    event_type = self.kill_model.names[class_id]

                    events.append({
                        "type": event_type,
                        "confidence": round(confidence, 4),
                        "bbox": [x1 + kx1, y1 + ky1, x1 + kx2, y1 + ky2]
                    })

            return events

        except Exception as e:
            self.logger.exception(f"Event detection failed: {e}")
            return []

    def run(self):
        self.logger.info("🎮 Game Phase Worker started")
        frame_id = 0
        while True:
            try:
                header, frame = recv_frame(self.input_socket)
                frame_id += 1

                # Resize to YOLO's default input size (640x640)
                frame = cv2.resize(frame, (640, 640))

                events = self.detect_events(frame)
                result = {
                    "frame_id": header["frame_id"],
                    "timestamp": datetime.now().isoformat(),
                    "phase": "GAME_PHASE",
                    "events": events,
                    "input_timestamp": header["timestamp"]
                }
                send_frame(self.output_socket, frame, result)

                # GPU Memory Cleanup after every 500 frames
                if frame_id % 500 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    log_gpu_usage("[Cleanup] ")

                self.logger.info(f"📦 Frame {header['frame_id']}: {len(events)} event(s)")

            except zmq.Again:
                time.sleep(0.1)  # wait briefly
            except Exception as e:
                self.logger.exception(f"Game worker error: {e}")

if __name__ == "__main__":
    worker = GameWorker()
    worker.run()
