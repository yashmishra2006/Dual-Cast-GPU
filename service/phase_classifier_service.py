import zmq
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import signal
import sys
import time

from torchvision import models, transforms
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest
from pathlib import Path
from typing import List
from rich.console import Console
import GPUtil

console = Console()

# Prometheus metrics
phase_requests_total = Counter('phase_requests_total', 'Total phase classification requests')
inference_seconds = Histogram('inference_seconds', 'Time spent on inference')

PHASE_CLASSES = ["AGENT_PHASE", "BUY_PHASE", "GAME_PHASE"]

def log_gpu_usage(prefix=""):
    """Logs the GPU memory usage."""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            used = gpus[0].memoryUsed
            total = gpus[0].memoryTotal
            console.log(f"{prefix}GPU Usage: {used:.1f} / {total:.1f} MB")
    except Exception as e:
        console.warning(f"GPU usage logging failed: {e}")

# ----------------- PHASE CLASSIFIER ----------------- #
class PhaseClassifier:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.log(f"[cyan]📦 Using device: {self.device}[/cyan]")
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

    def _load_model(self, model_path: str):
        console.log(f"[yellow]📥 Loading model from {model_path}...[/yellow]")
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device).eval()
        if self.device.type == "cuda":
            model = model.half()
        console.log("[green]✅ Model loaded successfully[/green]")
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def infer(self, frames: List[np.ndarray]):
        if not frames:
            raise ValueError("No frames received for inference")

        console.log(f"[blue]🔍 Running inference on {len(frames)} frame(s)[/blue]")
        batch = [self.transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        batch = torch.stack(batch).to(self.device)

        if self.device.type == 'cuda':
            batch = batch.half()

        with inference_seconds.time():
            with torch.no_grad():
                logits = self.model(batch)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()

        counts = np.bincount(preds, minlength=3)
        winning = np.argmax(counts)
        confidence = probs[:, winning].mean().item()

        console.log(f"[magenta]🏁 Predicted Phase: {PHASE_CLASSES[winning]}, Confidence: {confidence:.2f}[/magenta]")
        return PHASE_CLASSES[winning], confidence

# ----------------- ZMQ SERVICE ----------------- #
class PhaseClassifierService:
    def __init__(self, model_path: str, port: int = 5554):
        console.log("[yellow]🚀 Initializing Phase Classifier Service...[/yellow]")
        self.classifier = PhaseClassifier(model_path)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.RCVTIMEO = 1000
        self.socket.bind(f"tcp://*:{port}")
        console.log(f"[green]🔌 ZMQ REP socket bound on tcp://*:{port}[/green]")

        self.running = True
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, *_):
        console.log("[red]🛑 Phase Classifier Service shutting down...[/red]")
        self.running = False

    def serve(self):
        console.log("[bold green]✅ Phase Classifier Service is now running[/bold green]")
        frame_id = 0
        while self.running:
            try:
                if self.socket.poll(1000):
                    message = self.socket.recv_multipart()
                    headers = json.loads(message[0].decode())

                    frames = [
                        np.frombuffer(m, dtype=np.uint8).reshape(tuple(h['shape']))
                        for h, m in zip(headers, message[1:])
                    ]
                    console.log(f"[blue]📨 Received {len(frames)} frame(s)[/blue]")

                    phase, conf = self.classifier.infer(frames)
                    self.socket.send_string(json.dumps({"phase": phase, "confidence": conf}))
                    phase_requests_total.inc()

                    # GPU Memory Cleanup every 500 frames
                    frame_id += 1
                    if frame_id % 500 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        log_gpu_usage("[Cleanup] ")

                else:
                    time.sleep(0.1)

            except zmq.Again:
                continue
            except Exception as e:
                console.log(f"[bold red]❌ Inference Error: {e}[/bold red]")
                try:
                    self.socket.send_string(json.dumps({"error": str(e)}))
                except zmq.ZMQError:
                    console.log("[red]⚠️ Failed to send error response over socket[/red]")

        # Cleanup ZMQ resources
        self.socket.close()
        self.context.term()

# ----------------- FASTAPI FOR MONITORING ----------------- #
app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest())

# ----------------- ENTRY POINT ----------------- #
if __name__ == "__main__":
    model_path = str(Path(__file__).resolve().parent.parent / "models" / "phase_classifier.pt")
    service = PhaseClassifierService(model_path)
    service.serve()
    if service.classifier.device.type == 'cuda':
        torch.cuda.empty_cache()
