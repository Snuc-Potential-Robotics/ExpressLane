"""FastAPI application serving the Robot Vision & Telemetry Dashboard."""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.core.detector import FaceDetector
from src.core.emotion_engine import EmotionEngine
from src.core.temporal_smoother import TemporalSmoother
from src.comm.udp_transmitter import UDPTransmitter

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

# Emotion colors — muted, warm-minimal palette (BGR for OpenCV drawing).
# Matches the light frontend theme; no neon / cyan.
EMOTION_COLORS = {
    "Happy": (61, 128, 21),       # #15803D muted green
    "Sad": (105, 85, 71),         # #475569 slate
    "Angry": (28, 28, 185),       # #B91C1C muted red
    "Surprised": (12, 65, 194),   # #C2410C warm clay
    "Neutral": (108, 113, 120),   # #78716C stone gray
    "Fear": (78, 83, 87),         # #57534E stone
    "Disgust": (15, 124, 77),     # #4D7C0F olive
    "Contempt": (14, 64, 146),    # #92400E amber brown
}


class SettingsUpdate(BaseModel):
    target_ip: Optional[str] = None
    target_port: Optional[int] = None
    speed: Optional[int] = None
    turn_speed: Optional[int] = None
    activation_threshold: Optional[float] = None
    deactivation_threshold: Optional[float] = None
    min_consecutive_frames: Optional[int] = None
    expression_gain: Optional[float] = None
    mode: Optional[str] = None  # "AI" or "MANUAL"
    mapping: Optional[Dict[str, str]] = None


class ManualCommand(BaseModel):
    action: str  # FORWARD, BACKWARD, LEFT, RIGHT, STOP
    speed: Optional[int] = None


class RobotVisionSystem:
    def __init__(self):
        self.detector = FaceDetector()
        self.emotion_engine = EmotionEngine()
        self.smoother = TemporalSmoother()
        self.udp_tx = UDPTransmitter()

        # Operational parameters
        self.mode = "AI"  # "AI" or "MANUAL"
        self.default_speed = 200
        self.turn_speed = 130  # Slow and precise turn speed (PWM 130 vs 200)
        self.cap = None
        self.is_running = False

        # Telemetry cache
        self.fps = 0.0
        self.inference_latency_ms = 0.0
        self.face_detected = False
        self.active_emotion = "Neutral"
        self.active_action = "STOP"
        self.active_confidence = 1.0
        self.scores: Dict[str, float] = {}
        self.latest_frame_bytes: Optional[bytes] = None

        # Camera index
        self.camera_index = 0

    def start(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.udp_tx.start()
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False
        self.udp_tx.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def get_action_speed(self, action: str, requested_speed: Optional[int] = None) -> int:
        """Calculate targeted speed: slow and precise for turns, standard for linear travel."""
        if action == "STOP":
            return 0
        if requested_speed is not None:
            return max(0, min(255, int(requested_speed)))
        if action in ("LEFT", "RIGHT"):
            return self.turn_speed
        return self.default_speed

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run face detection, emotion recognition, and draw a minimal overlay."""
        h, w = frame.shape[:2]
        faces = self.detector.detect(frame)
        # Reliability gate: tiny/far faces are low-signal — hold through the
        # smoother's dropout grace instead of trusting a noisy crop.
        reliable = bool(faces) and self.detector.is_reliable(faces[0], frame.shape)

        if reliable:
            self.face_detected = True
            primary_face = faces[0]
            x, y, bw, bh = primary_face.bbox

            # Crop and align face
            cropped = self.detector.align_and_crop(frame, primary_face)
            dom_emo, dom_score, raw_scores, lat_ms = self.emotion_engine.predict(cropped)
            self.inference_latency_ms = lat_ms

            # Temporal smoothing & debouncing
            smooth_emo, smooth_act, smooth_conf, smooth_scores = self.smoother.process(
                raw_scores, face_detected=True
            )
            self.active_emotion = smooth_emo
            self.active_confidence = smooth_conf
            self.scores = smooth_scores

            # If in AI mode, apply smoothed action; if in manual, keep manual command
            if self.mode == "AI":
                self.active_action = smooth_act
                self.udp_tx.update_command(
                    action=self.active_action,
                    speed=self.get_action_speed(self.active_action),
                    emotion=self.active_emotion,
                    confidence=self.active_confidence,
                )

            # Draw overlay graphics
            color = EMOTION_COLORS.get(self.active_emotion, (108, 113, 120))
            
            # Corner Reticles
            reticle_len = max(15, int(bw * 0.18))
            thick = 2
            # Top-Left
            cv2.line(frame, (x, y), (x + reticle_len, y), color, thick)
            cv2.line(frame, (x, y), (x, y + reticle_len), color, thick)
            # Top-Right
            cv2.line(frame, (x + bw, y), (x + bw - reticle_len, y), color, thick)
            cv2.line(frame, (x + bw, y), (x + bw, y + reticle_len), color, thick)
            # Bottom-Left
            cv2.line(frame, (x, y + bh), (x + reticle_len, y + bh), color, thick)
            cv2.line(frame, (x, y + bh), (x, y + bh - reticle_len), color, thick)
            # Bottom-Right
            cv2.line(frame, (x + bw, y + bh), (x + bw - reticle_len, y + bh), color, thick)
            cv2.line(frame, (x + bw, y + bh), (x + bw, y + bh - reticle_len), color, thick)

            # Draw Facial Landmarks (muted tones)
            if primary_face.right_eye and primary_face.left_eye:
                cv2.circle(frame, (int(primary_face.right_eye[0]), int(primary_face.right_eye[1])), 3, (23, 25, 28), -1)
                cv2.circle(frame, (int(primary_face.left_eye[0]), int(primary_face.left_eye[1])), 3, (23, 25, 28), -1)
            if primary_face.nose_tip:
                cv2.circle(frame, (int(primary_face.nose_tip[0]), int(primary_face.nose_tip[1])), 3, (61, 128, 21), -1)
            if primary_face.right_mouth and primary_face.left_mouth:
                cv2.circle(frame, (int(primary_face.right_mouth[0]), int(primary_face.right_mouth[1])), 3, (28, 28, 185), -1)
                cv2.circle(frame, (int(primary_face.left_mouth[0]), int(primary_face.left_mouth[1])), 3, (28, 28, 185), -1)

            # Draw Emotion Badge & Action Header (light minimal badge)
            badge_text = f"{self.active_emotion.upper()} ({int(self.active_confidence * 100)}%) -> {self.active_action}"
            font_scale = 0.55
            (text_w, text_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.rectangle(frame, (x, max(0, y - 28)), (x + text_w + 10, y), (250, 249, 246), -1)
            cv2.rectangle(frame, (x, max(0, y - 28)), (x + text_w + 10, y), color, 1)
            cv2.putText(frame, badge_text, (x + 5, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (23, 25, 28), 2)
        else:
            self.face_detected = False
            # Grace-aware failsafe: brief dropouts hold the last command
            # (smoother rides through max_missed_frames), sustained loss
            # decays to Neutral/STOP. Command stream keeps the watchdog fed.
            smooth_emo, smooth_act, smooth_conf, smooth_scores = self.smoother.process(
                None, face_detected=False
            )
            self.active_emotion = smooth_emo
            self.active_confidence = smooth_conf
            self.scores = smooth_scores

            if self.mode == "AI":
                self.active_action = smooth_act
                self.udp_tx.update_command(
                    action=self.active_action,
                    speed=self.get_action_speed(self.active_action),
                    emotion=self.active_emotion,
                    confidence=self.active_confidence,
                )

            # Display status line: holding vs searching
            if faces and not reliable:
                cv2.putText(frame, "[LOW SIGNAL - HOLDING...]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 70, 60), 2)
            else:
                cv2.putText(frame, "[SEARCHING FOR FACE...]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 70, 60), 2)

        # Top Watermark: Mode & FPS
        hud_top = f"MODE: {self.mode}  |  FPS: {self.fps:.1f}  |  AI LATENCY: {self.inference_latency_ms:.1f}ms"
        cv2.putText(frame, hud_top, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)

        return frame


vision_system = RobotVisionSystem()
app = FastAPI(title="AI Face Emotion Bot Cockpit")


@app.on_event("startup")
async def startup_event():
    vision_system.start(camera_index=0)


@app.on_event("shutdown")
async def shutdown_event():
    vision_system.stop()


def generate_video_frames():
    """Generator capturing webcam frames, processing AI HUD, and yielding MJPEG chunks."""
    cap = cv2.VideoCapture(vision_system.camera_index)
    if not cap.isOpened():
        logger.warning(f"Could not open webcam index {vision_system.camera_index}. Generating placeholder canvas.")
        while True:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "WEBCAM OFFLINE", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            _, encoded = cv2.imencode(".jpg", blank)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")
            time.sleep(0.1)

    prev_time = time.time()
    while vision_system.is_running:
        success, frame = cap.read()
        if not success:
            time.sleep(0.02)
            continue

        now = time.time()
        dt = now - prev_time
        if dt > 0:
            vision_system.fps = 0.9 * vision_system.fps + 0.1 * (1.0 / dt)
        prev_time = now

        # Process frame with AI
        processed = vision_system.process_frame(frame)
        _, encoded = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")

    cap.release()


@app.get("/")
async def get_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_file)


@app.get("/video_feed")
async def video_feed():
    """Stream real-time video feed with AR overlay."""
    return StreamingResponse(
        generate_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Real-time bi-directional telemetry websocket connection."""
    await websocket.accept()
    try:
        while True:
            telemetry = {
                "fps": round(vision_system.fps, 1),
                "latency_ms": round(vision_system.inference_latency_ms, 1),
                "face_detected": vision_system.face_detected,
                "active_emotion": vision_system.active_emotion,
                "active_action": vision_system.active_action,
                "confidence": round(vision_system.active_confidence, 2),
                "scores": {k: round(v, 2) for k, v in vision_system.scores.items()},
                "mode": vision_system.mode,
                "speed": vision_system.default_speed,
                "turn_speed": vision_system.turn_speed,
                "udp": vision_system.udp_tx.get_telemetry(),
                "mapping": vision_system.smoother.mapping,
                "tuning": {
                    "activation_threshold": vision_system.smoother.activation_threshold,
                    "deactivation_threshold": vision_system.smoother.deactivation_threshold,
                    "min_consecutive_frames": vision_system.smoother.min_consecutive_frames,
                    "expression_gain": vision_system.emotion_engine.expression_gain,
                    "turn_speed": vision_system.turn_speed,
                },
            }
            await websocket.send_text(json.dumps(telemetry))
            await asyncio.sleep(0.04)  # ~25 Hz telemetry stream
    except WebSocketDisconnect:
        pass


@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate):
    """Update system settings dynamically."""
    if settings.target_ip and settings.target_port:
        vision_system.udp_tx.set_target(settings.target_ip, settings.target_port)
    if settings.speed is not None:
        vision_system.default_speed = max(0, min(255, settings.speed))
        if settings.turn_speed is None:
            vision_system.turn_speed = max(60, min(200, int(vision_system.default_speed * 0.65)))
    if settings.turn_speed is not None:
        vision_system.turn_speed = max(0, min(255, settings.turn_speed))
    if settings.activation_threshold is not None:
        vision_system.smoother.activation_threshold = float(settings.activation_threshold)
    if settings.deactivation_threshold is not None:
        vision_system.smoother.deactivation_threshold = float(settings.deactivation_threshold)
    if settings.min_consecutive_frames is not None:
        vision_system.smoother.min_consecutive_frames = max(1, int(settings.min_consecutive_frames))
    if settings.expression_gain is not None:
        # Raise if your expressions never trigger; lower if the bot twitches
        # while your face is at rest.
        vision_system.emotion_engine.expression_gain = max(1.0, float(settings.expression_gain))
    if settings.mode is not None and settings.mode in ["AI", "MANUAL"]:
        vision_system.mode = settings.mode
    if settings.mapping is not None:
        vision_system.smoother.update_mapping(settings.mapping)
    return {"status": "ok", "settings": settings.dict()}


@app.post("/api/manual_command")
async def send_manual_command(cmd: ManualCommand):
    """Execute manual drive command."""
    if vision_system.mode != "MANUAL":
        vision_system.mode = "MANUAL"
    
    vision_system.active_action = cmd.action.upper()
    speed = vision_system.get_action_speed(vision_system.active_action, cmd.speed)
    vision_system.udp_tx.update_command(
        action=vision_system.active_action,
        speed=speed,
        emotion="Manual",
        confidence=1.0,
    )
    return {"status": "ok", "action": vision_system.active_action, "speed": speed}


@app.post("/api/emergency_stop")
async def emergency_stop():
    """Immediately stop robot and set mode to MANUAL."""
    vision_system.mode = "MANUAL"
    vision_system.active_action = "STOP"
    vision_system.udp_tx.send_emergency_stop()
    return {"status": "ok", "message": "Emergency STOP executed"}


# Mount static assets
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
