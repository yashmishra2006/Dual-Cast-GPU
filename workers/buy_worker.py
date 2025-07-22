import logging
from datetime import datetime
from pathlib import Path
import sys
import importlib.util

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import zmq
from ultralytics import YOLO

from rich.console import Console
from rich.logging import RichHandler
import GPUtil

# Setup rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("BuyWorker")
console = Console()

# === Dynamically import core modules ===
def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

core_path = Path(__file__).resolve().parent.parent / "core"
DeviceManager = import_from_path("device", core_path / "device.py").DeviceManager
load_resnet = import_from_path("model_loader", core_path / "model_loader.py").load_resnet
get_imagenet_transform = import_from_path("transforms", core_path / "transforms.py").get_imagenet_transform
zmq_utils = import_from_path("zmq_utils", core_path / "zmq_utils.py")
recv_frame = zmq_utils.recv_frame
send_frame = zmq_utils.send_frame

# === Labels ===
WEAPON_CLASSES = {
    0: "phantom", 1: "vandal", 2: "classic", 3: "spectre", 4: "judge", 5: "ghost",
    6: "bucky", 7: "operator", 8: "ares", 9: "sheriff", 10: "odin", 11: "shorty"
}

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

class BuyWorker:
    def __init__(self, input_port=5556, output_port=5560):
        self.logger = logger
        self.device = DeviceManager().get()
        console.log(f"[cyan]Using device: {self.device}[/cyan]")

        # Load models
        yolo_path = Path(__file__).resolve().parent.parent / "models" / "buyphase.pt"
        resnet_path = Path(__file__).resolve().parent.parent / "models" / "buyphase_resnet.pt"

        console.log(f"[green]Loading YOLO model from {yolo_path}[/green]")
        self.yolo = YOLO(str(yolo_path)).to(self.device)

        console.log(f"[green]Loading ResNet model from {resnet_path}[/green]")
        self.resnet = load_resnet(str(resnet_path), len(WEAPON_CLASSES), self.device, half=True)
        self.transform = get_imagenet_transform()

        # Setup ZMQ
        self.context = zmq.Context()
        self.input_socket = self.context.socket(zmq.PULL)
        self.input_socket.RCVTIMEO = 1000
        self.input_socket.bind(f"tcp://*:{input_port}")
        console.log(f"[green]Listening on tcp://*:{input_port}[/green]")

        self.output_socket = self.context.socket(zmq.PUSH)
        self.output_socket.connect(f"tcp://localhost:{output_port}")
        console.log(f"[green]Pushing output to tcp://localhost:{output_port}[/green]")

    def detect_boxes(self, frame):
        try:
            results = self.yolo(frame, conf=0.5, verbose=False)
            return results[0].boxes if results and results[0].boxes is not None else []
        except Exception as e:
            self.logger.exception(f"YOLO detection failed: {e}")
            return []

    def classify_crop(self, crop):
        try:
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            if self.device.type == 'cuda':
                tensor = tensor.half()
            with torch.no_grad():
                output = self.resnet(tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                label = WEAPON_CLASSES.get(pred.item(), "unknown")
                return label, conf.item()
        except Exception as e:
            self.logger.exception(f"Classification failed: {e}")
            return "unknown", 0.0

    def run(self):
        self.logger.info("💰 Buy Phase Worker started")
        frame_id = 0
        while True:
            try:
                header, frame = recv_frame(self.input_socket)
                frame_id += 1

                # Resize to YOLO's default input size (640x640)
                frame = cv2.resize(frame, (640, 640))

                detections = self.detect_boxes(frame)
                weapons = []

                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    crop = frame[y1:y2, x1:x2]
                    label, confidence = self.classify_crop(crop)
                    weapons.append({
                        "label": label,
                        "confidence": round(confidence, 4),
                        "bbox": [x1, y1, x2, y2]
                    })
                    self.logger.info(f"🪖 {label} ({confidence:.2f}) at [{x1}, {y1}, {x2}, {y2}]")

                result = {
                    "frame_id": header["frame_id"],
                    "timestamp": datetime.now().isoformat(),
                    "phase": "BUY_PHASE",
                    "weapons": weapons,
                    "input_timestamp": header["timestamp"]
                }

                send_frame(self.output_socket, frame, result)
                self.logger.info(f"📦 Frame {header['frame_id']}: {len(weapons)} weapons detected")

                # GPU Memory Cleanup every 500 frames
                if frame_id % 500 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    log_gpu_usage("[Cleanup] ")

            except zmq.Again:
                continue  # timeout waiting for frame
            except Exception as e:
                self.logger.exception(f"Buy worker error: {e}")

if __name__ == "__main__":
    worker = BuyWorker()
    worker.run()
