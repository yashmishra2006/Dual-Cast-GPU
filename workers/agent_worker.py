import logging
from datetime import datetime
from pathlib import Path
import importlib.util

import cv2
import torch
import torch.nn.functional as F
import zmq
import numpy as np
from rich.console import Console
from rich.logging import RichHandler

# Rich logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("AgentWorker")
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

CLASS_NAMES = [
    "Astra", "Breach", "Brimstone", "Chamber", "Clove", "Cypher", "Deadlock", "Fade", "Gekko", "Harbor",
    "Iso", "Jett", "KAY-O", "Killjoy", "Neon", "Omen", "Phoenix", "Raze", "Reyna", "Sage",
    "Skye", "Sova", "Tejo", "Viper", "Vyse", "Waylay", "Yoru"
]

class AgentWorker:
    def __init__(self, input_port=5555, output_port=5559):
        self.logger = logger
        self.device = DeviceManager().get()
        console.log(f"[cyan]Using device: {self.device}[/cyan]")

        self.transform = get_imagenet_transform()
        model_path = Path(__file__).resolve().parent.parent / "models" / "agent_resnet.pt"
        console.log(f"[green]Loading agent model from: {model_path}[/green]")
        self.model = load_resnet(str(model_path), len(CLASS_NAMES), self.device, half=True)

        self.context = zmq.Context()
        self.input_socket = self.context.socket(zmq.PULL)
        self.input_socket.RCVTIMEO = 1000
        self.input_socket.bind(f"tcp://*:{input_port}")
        console.log(f"[green]Listening on tcp://*:{input_port}[/green]")

        self.output_socket = self.context.socket(zmq.PUSH)
        self.output_socket.connect(f"tcp://localhost:{output_port}")
        console.log(f"[green]Sending output to tcp://localhost:{output_port}[/green]")

    def infer(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.transform(image).unsqueeze(0).to(self.device)
        if self.device.type == 'cuda':
            image = image.half()

        with torch.no_grad():
            logits = self.model(image)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()

        return CLASS_NAMES[pred], confidence

    def run(self):
        self.logger.info("🧠 Agent Worker started")
        while True:
            try:
                header, frame = recv_frame(self.input_socket)
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
                self.logger.info(f"🎯 Frame {header['frame_id']}: {agent} ({conf:.2f})")

            except zmq.Again:
                continue
            except Exception as e:
                self.logger.exception(f"Agent worker error: {e}")

if __name__ == "__main__":
    worker = AgentWorker()
    worker.run()
