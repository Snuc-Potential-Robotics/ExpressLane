"""
Single-file real-time 4-expression recognizer for an Expression Race.

Pipeline: webcam -> OpenCV Haar cascade face detection -> pretrained ONNX
FER+ emotion model (via OpenCV's own DNN module) -> temporal smoothing ->
game action.

Dependencies (only these two):
    pip install opencv-python numpy

No TensorFlow, no Keras, no `fer` package — this avoids the dependency
issues those bring (missing exports, pkg_resources, version clashes).
The only thing that touches the network is a ONE-TIME download of the
~35MB pretrained model on first run; every run after that is fully
offline, and no webcam data ever leaves the machine.

Usage:
    python emotion_race.py
Press 'q' to quit.
"""
import os
import time
import urllib.request
from collections import deque, Counter

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config — tweak these freely
# ---------------------------------------------------------------------------
MODEL_PATH = "emotion-ferplus-8.onnx"
MODEL_URL = (
    "https://github.com/spmallick/learnopencv/raw/master/"
    "Facial-Emotion-Recognition/emotion-ferplus-8.onnx"
)

# The FER+ model's 8 output labels, in its fixed output order.
FERPLUS_LABELS = [
    "neutral", "happy", "surprised", "sad", "angry", "disgust", "fear", "contempt",
]

# Only these 4 matter for the race; everything else (neutral/disgust/
# fear/contempt) is simply ignored.
TARGET_EXPRESSIONS = ["happy", "sad", "angry", "surprised"]

EXPRESSION_ACTIONS = {
    "happy": "MOVE_FORWARD",
    "sad": "MOVE_BACKWARD",
    "angry": "LEFT",
    "surprised": "RIGHT",
}

CAMERA_INDEX = 0
SMOOTHING_WINDOW = 8      # recent frames to majority-vote over
MIN_CONFIDENCE = 0.4      # ignore per-frame predictions below this
ACTION_COOLDOWN = 0.6     # seconds before the same action can fire again


# ---------------------------------------------------------------------------
# One-time model download (cached locally next to this script after that)
# ---------------------------------------------------------------------------
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading emotion model (one-time, ~35MB) to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done. Every run after this is fully offline.")


# ---------------------------------------------------------------------------
# Face detection: OpenCV's bundled Haar cascade — ships with opencv-python,
# nothing extra to download or install for this part.
# ---------------------------------------------------------------------------
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face(frame_bgr):
    """Return (cropped_grayscale_face, bbox) for the largest face, or (None, None)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
    return gray[y:y + h, x:x + w], (x, y, w, h)


# ---------------------------------------------------------------------------
# Expression classifier: pretrained ONNX FER+ model via OpenCV's DNN module
# ---------------------------------------------------------------------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_expression(net, face_gray):
    """Run the FER+ model on a cropped grayscale face. Returns (label, confidence)."""
    face_resized = cv2.resize(face_gray, (64, 64))
    blob = face_resized.astype(np.float32).reshape(1, 1, 64, 64)
    net.setInput(blob)
    raw_scores = net.forward().flatten()
    probs = softmax(raw_scores)

    # Restrict to our 4 target expressions and renormalize among just those.
    scored = {
        label: probs[i] for i, label in enumerate(FERPLUS_LABELS) if label in TARGET_EXPRESSIONS
    }
    total = sum(scored.values()) or 1e-8
    scored = {k: v / total for k, v in scored.items()}

    label = max(scored, key=scored.get)
    return label, scored[label]


# ---------------------------------------------------------------------------
# Temporal smoothing: majority vote over a sliding window of recent frames,
# so one noisy/wrong frame can't fire an action by itself.
# ---------------------------------------------------------------------------
class Smoother:
    def __init__(self, window_size=SMOOTHING_WINDOW, min_confidence=MIN_CONFIDENCE):
        self.buffer = deque(maxlen=window_size)
        self.min_confidence = min_confidence
        self.window_size = window_size

    def update(self, label, confidence):
        if confidence >= self.min_confidence:
            self.buffer.append(label)
        if not self.buffer:
            return None
        top_label, votes = Counter(self.buffer).most_common(1)[0]
        ratio = votes / len(self.buffer)
        if len(self.buffer) >= max(3, self.window_size // 2) and ratio < 0.5:
            return None
        return top_label


# ---------------------------------------------------------------------------
# Game action firing, with a cooldown so holding an expression doesn't spam
# the same action every single frame.
# ---------------------------------------------------------------------------
class GameController:
    def __init__(self, cooldown=ACTION_COOLDOWN):
        self.cooldown = cooldown
        self.last_time = 0.0
        self.last_expression = None

    def handle(self, expression):
        if expression is None:
            return
        action = EXPRESSION_ACTIONS.get(expression)
        if action is None:
            return
        now = time.time()
        if expression == self.last_expression and (now - self.last_time) < self.cooldown:
            return
        # <-- Replace this print with your real race hook (socket, serial,
        # game-engine call, etc.)
        print(f"[GAME] expression={expression!r} -> action={action!r}")
        self.last_time = now
        self.last_expression = expression


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ensure_model()
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}")

    smoother = Smoother()
    controller = GameController()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            face_gray, bbox = detect_face(frame)

            label, confidence = None, 0.0
            if face_gray is not None:
                label, confidence = predict_expression(net, face_gray)
                smoothed = smoother.update(label, confidence)
                controller.handle(smoothed)

            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if label is not None:
                    cv2.putText(
                        frame, f"{label} ({confidence:.2f})", (x, max(y - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )

            cv2.imshow("Expression Race", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()