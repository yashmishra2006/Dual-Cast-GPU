# agent_worker.py

import logging
import time
from datetime import datetime
from pathlib import Path
import importlib.util

import cv2
import torch
import torch.nn.functional as F
import zmq
import numpy as np
from rich.logging import RichHandler
import GPUtil

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("AgentWorker")

# === Dynamic Imports ===
def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

core_path = Path(__file__).resolve().parent.parent / "core"
DeviceManager = import_from_path("device", core_path / "device.py").DeviceManager
load_resnet = import_from_path("model_loader", core_path / "model_loader.py").load_resnet
get_imagenet_transform = import_from_path("transforms", core_path / "transforms.py").get_imagenet_transform
zmq_utils = import_from_path("zmq_utils", core_path / "zmq_utils.py")
recv_frame = zmq_utils.recv_frame
send_frame = zmq_utils.send_frame

# === Agent Classes ===
CLASS_NAMES = [
    "Astra", "Breach", "Brimstone", "Chamber", "Clove", "Cypher", "Deadlock", "Fade", "Gekko", "Harbor",
    "Iso", "Jett", "KAY-O", "Killjoy", "Neon", "Omen", "Phoenix", "Raze", "Reyna", "Sage",
    "Skye", "Sova", "Tejo", "Viper", "Vyse", "Waylay", "Yoru"
]
NUM_CLASSES = len(CLASS_NAMES)

torch.backends.cudnn.benchmark = True  # Optimized conv kernel selection

# === Agent Worker Class ===
class AgentWorker:
    def __init__(self, input_port=5555, output_port=5559):
        self.device = DeviceManager().get()
        logger.info(f"🧠 Using device: {self.device}")

        # Load model
        model_path = Path(__file__).resolve().parent.parent / "models" / "agent_resnet.pt"
        self.model = load_resnet(str(model_path), NUM_CLASSES, self.device, half=True)

        # ZMQ sockets
        self.context = zmq.Context()
        self.input_socket = self.context.socket(zmq.PULL)
        self.input_socket.RCVTIMEO = 1000  # 1s timeout
        self.input_socket.bind(f"tcp://*:{input_port}")

        self.output_socket = self.context.socket(zmq.PUSH)
        self.output_socket.connect(f"tcp://localhost:{output_port}")

        self.transform = get_imagenet_transform()
        logger.info("🚀 Agent Worker initialized.")

    def infer(self, frame):
        resized = cv2.resize(frame, (224, 224))
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()

        return CLASS_NAMES[pred], confidence

    def run(self):
        logger.info("🎬 Agent Worker started")
        frame_id = 0

        try:
            while True:
                try:
                    header, frame = recv_frame(self.input_socket)
                    frame_id += 1

                    agent, conf = self.infer(frame)

                    result = {
                        "frame_id": header["frame_id"],
                        "timestamp": datetime.now().isoformat(),
                        "phase": "AGENT_PHASE",
                        "agent": agent,
                        "confidence": round(conf, 4),
                        "input_timestamp": header["timestamp"]
                    }

                    send_frame(self.output_socket, frame, result)
                    logger.info(f"✅ Frame {header['frame_id']}: {agent} ({conf:.2f})")

                    if frame_id % 500 == 0 and torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                            logger.info("[Cleanup] GPU cache cleared.")
                        except Exception as e:
                            logger.warning(f"[Cleanup] GPU cleanup failed: {e}")    

                except zmq.Again:
                    continue  # Timed out waiting for a frame
                except Exception as e:
                    logger.exception(f"❌ Processing error: {e}")

        except Exception as e:
            logger.exception(f"🚨 Worker crashed: {e}")
            
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("🧹 Cleaning up...")
        self.input_socket.close()
        self.output_socket.close()
        self.context.term()
        torch.cuda.empty_cache()
        logger.info("✅ Cleanup complete.")

# === Entry Point ===
if __name__ == "__main__":
    AgentWorker().run()
