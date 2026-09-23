"""
Single-file real-time expression recognizer for an Expression Race.

Pipeline: webcam -> OpenCV Haar cascade face detection -> pretrained ONNX
FER+ emotion model (via OpenCV's own DNN module) -> temporal smoothing ->
game action.

v5 changes (fixing issues from v4):
  - FALSE PAUSE ON BACKGROUND OBJECTS: the wood-toned chairs in the
    background fell inside the skin-color HSV range and, being roughly
    hand-sized, satisfied the old "any blob of the right size" check.
    Fixed by requiring the blob to actually look like an open hand:
      1. Finger counting via convexity defects (properly converted from
         OpenCV's fixed-point depth units this time — v3 had a units bug
         where raw depth wasn't divided by 256). A candidate needs at
         least 4 finger-shaped gaps (~5 fingertips) at a plausible
         angle/depth to count.
      2. Solidity check (contour area / convex-hull area) — an open hand
         has a lower solidity than the solid or blocky background shapes
         that were false-triggering before.
      3. Area bounds are now scaled relative to the detected face size
         (a hand should be roughly comparable in size to the face at the
         same distance from camera), not a fixed absolute pixel range —
         so an oddly-sized background blob can't sneak into the old fixed
         window.
    The debug overlay now also shows the live finger count so you can see
    exactly why a candidate blob was accepted or rejected.

Dependencies (only these two):
    pip install opencv-python numpy

Usage:
    python emotion_race.py
Press 'q' to quit. Show an open hand (5 fingers spread) to the camera to
pause; remove it and press 'c' to resume.
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

FERPLUS_LABELS = [
    "neutral", "happy", "surprised", "sad", "angry", "disgust", "fear", "contempt",
]

TARGET_EXPRESSIONS = ["happy", "sad", "angry", "surprised", "disgust"]

EXPRESSION_ACTIONS = {
    "happy": "MOVE_FORWARD",
    "sad": "MOVE_BACKWARD",
    "angry": "LEFT",
    "surprised": "RIGHT",
    "disgust": "BOOST",
}

CAMERA_INDEX = 0
SMOOTHING_WINDOW = 8
MIN_CONFIDENCE = 0.4
PER_EXPRESSION_MIN_CONFIDENCE = {
    "disgust": 0.22,
}
DISGUST_LOGIT_BIAS = 1.5
ACTION_COOLDOWN = 0.6

SHOW_DEBUG_OVERLAY = True
SHOW_HAND_MASK_PREVIEW = True

STOP_CONSEC_FRAMES = 5   # frames of sustained open-hand before pausing
BBOX_TTL = 1.0            # seconds to keep excluding the last known face box

# Hand-shape requirements
REQUIRED_FINGER_GAPS = 4        # 4 gaps ~= 5 fingertips spread open
DEFECT_ANGLE_MAX_RAD = 1.75     # ~100 degrees; wider = more lenient
DEFECT_DEPTH_MIN_RATIO = 0.10   # min defect depth, as a fraction of the
                                 # candidate's own bounding-box diagonal
SOLIDITY_RANGE = (0.45, 0.92)   # contour area / convex-hull area
# A hand's on-screen area, relative to the detected face's area, when
# both are roughly the same distance from the camera.
HAND_TO_FACE_AREA_RANGE = (0.15, 3.5)
FALLBACK_MIN_AREA = 6000        # used only if no face size is known at all
FALLBACK_MAX_AREA_FRAC = 0.35   # ...as a fraction of the frame area


# ---------------------------------------------------------------------------
# One-time model download
# ---------------------------------------------------------------------------
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading emotion model (one-time, ~35MB) to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done. Every run after this is fully offline.")


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray[y:y + h, x:x + w], (x, y, w, h)


# ---------------------------------------------------------------------------
# Expression classifier
# ---------------------------------------------------------------------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_expression(net, face_gray):
    face_resized = cv2.resize(face_gray, (64, 64))
    blob = face_resized.astype(np.float32).reshape(1, 1, 64, 64)
    net.setInput(blob)
    raw_scores = net.forward().flatten().copy()

    disgust_idx = FERPLUS_LABELS.index("disgust")
    raw_scores[disgust_idx] += DISGUST_LOGIT_BIAS

    probs = softmax(raw_scores)
    scored = {
        label: float(probs[i])
        for i, label in enumerate(FERPLUS_LABELS)
        if label in TARGET_EXPRESSIONS
    }
    total = sum(scored.values()) or 1e-8
    scored = {k: v / total for k, v in scored.items()}

    label = max(scored, key=scored.get)
    return label, scored[label], scored


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------
class Smoother:
    def __init__(self, window_size=SMOOTHING_WINDOW, min_confidence=MIN_CONFIDENCE):
        self.buffer = deque(maxlen=window_size)
        self.min_confidence = min_confidence
        self.window_size = window_size

    def update(self, label, confidence):
        floor = PER_EXPRESSION_MIN_CONFIDENCE.get(label, self.min_confidence)
        if confidence >= floor:
            self.buffer.append(label)
        if not self.buffer:
            return None
        top_label, votes = Counter(self.buffer).most_common(1)[0]
        ratio = votes / len(self.buffer)
        if len(self.buffer) >= max(3, self.window_size // 2) and ratio < 0.5:
            return None
        return top_label

    def reset(self):
        self.buffer.clear()


# ---------------------------------------------------------------------------
# Game action firing
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
        print(f"[GAME] expression={expression!r} -> action={action!r}")
        self.last_time = now
        self.last_expression = expression


# ---------------------------------------------------------------------------
# Open-hand detector: skin-color blob outside the face box that ALSO
# passes finger-count + solidity + face-relative-size checks.
# ---------------------------------------------------------------------------
class HandStopDetector:
    def __init__(self, consec_frames=STOP_CONSEC_FRAMES):
        self.consec_frames = consec_frames
        self.hit_streak = 0

    @staticmethod
    def _skin_mask(frame_bgr, exclude_bbox=None):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 20, 70], np.uint8), np.array([25, 180, 255], np.uint8))
        m2 = cv2.inRange(hsv, np.array([160, 20, 70], np.uint8), np.array([180, 180, 255], np.uint8))
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        if exclude_bbox is not None:
            x, y, w, h = exclude_bbox
            pad = int(0.5 * w)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(mask.shape[1], x + w + pad)
            y1 = min(mask.shape[0], y + h + pad)
            mask[y0:y1, x0:x1] = 0
        return mask

    @staticmethod
    def _finger_gaps(cnt):
        """Count finger-shaped concavities via convex-hull defects.
        Returns (gap_count, solidity)."""
        hull_pts = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull_pts)
        cnt_area = cv2.contourArea(cnt)
        solidity = (cnt_area / hull_area) if hull_area > 0 else 1.0

        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 4:
            return 0, solidity
        defects = cv2.convexityDefects(cnt, hull_idx)
        if defects is None:
            return 0, solidity

        x, y, w, h = cv2.boundingRect(cnt)
        diag = (w ** 2 + h ** 2) ** 0.5
        depth_min_px = DEFECT_DEPTH_MIN_RATIO * diag

        gaps = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth_px = d / 256.0  # OpenCV returns depth in fixed-point (x256)
            if depth_px < depth_min_px:
                continue
            start, end, far = cnt[s][0], cnt[e][0], cnt[f][0]
            a = np.linalg.norm(start - far)
            b = np.linalg.norm(end - far)
            c = np.linalg.norm(start - end)
            if a * b == 0:
                continue
            angle = np.arccos(np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0))
            if angle <= DEFECT_ANGLE_MAX_RAD:
                gaps += 1
        return gaps, solidity

    def reset(self):
        self.hit_streak = 0

    def update(self, frame_bgr, exclude_bbox=None, face_area=None):
        """Returns (sustained_now, hand_seen_this_frame, contour_or_None,
        mask, finger_gaps)."""
        mask = self._skin_mask(frame_bgr, exclude_bbox)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if face_area:
            min_area = HAND_TO_FACE_AREA_RANGE[0] * face_area
            max_area = HAND_TO_FACE_AREA_RANGE[1] * face_area
        else:
            min_area = FALLBACK_MIN_AREA
            max_area = FALLBACK_MAX_AREA_FRAC * frame_bgr.shape[0] * frame_bgr.shape[1]

        best_cnt, best_gaps = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            gaps, solidity = self._finger_gaps(cnt)
            if not (SOLIDITY_RANGE[0] <= solidity <= SOLIDITY_RANGE[1]):
                continue
            if gaps >= REQUIRED_FINGER_GAPS and gaps > best_gaps:
                best_cnt, best_gaps = cnt, gaps

        hand_seen = best_cnt is not None
        self.hit_streak = self.hit_streak + 1 if hand_seen else 0
        sustained = self.hit_streak >= self.consec_frames
        return sustained, hand_seen, best_cnt, mask, best_gaps


# ---------------------------------------------------------------------------
# Small on-screen debug helpers
# ---------------------------------------------------------------------------
def draw_expression_debug(frame, scores):
    y = 25
    for label in TARGET_EXPRESSIONS:
        val = scores.get(label, 0.0)
        color = (0, 255, 255) if label != "disgust" else (255, 150, 0)
        cv2.putText(frame, f"{label}: {val:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20


def draw_mask_preview(frame, mask):
    h, w = frame.shape[:2]
    thumb_w, thumb_h = 160, 120
    thumb = cv2.resize(mask, (thumb_w, thumb_h))
    thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    x0, y0 = w - thumb_w - 10, 10
    frame[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb_bgr
    cv2.rectangle(frame, (x0, y0), (x0 + thumb_w, y0 + thumb_h), (255, 255, 255), 1)
    cv2.putText(frame, "skin mask", (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


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
    hand_stop = HandStopDetector()

    state = "RUNNING"
    last_bbox = None
    last_bbox_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()

            face_gray, bbox = detect_face(frame)
            if bbox is not None:
                last_bbox, last_bbox_time = bbox, now
            exclude_bbox = bbox
            if exclude_bbox is None and (now - last_bbox_time) < BBOX_TTL:
                exclude_bbox = last_bbox

            face_area = None
            active_bbox = bbox if bbox is not None else (
                last_bbox if (now - last_bbox_time) < BBOX_TTL else None
            )
            if active_bbox is not None:
                face_area = active_bbox[2] * active_bbox[3]

            label, confidence, scores = None, 0.0, {}
            if state == "RUNNING" and face_gray is not None:
                label, confidence, scores = predict_expression(net, face_gray)
                smoothed = smoother.update(label, confidence)
                controller.handle(smoothed)

            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if label is not None:
                    tag = f"{label} ({confidence:.2f})"
                    if label == "disgust":
                        tag += " BOOST!"
                    cv2.putText(frame, tag, (x, max(y - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if SHOW_DEBUG_OVERLAY and scores:
                draw_expression_debug(frame, scores)

            sustained, hand_seen, hand_cnt, mask, finger_gaps = hand_stop.update(
                frame, exclude_bbox=exclude_bbox, face_area=face_area
            )
            if hand_cnt is not None:
                cv2.drawContours(frame, [hand_cnt], -1, (255, 0, 0), 2)

            if SHOW_DEBUG_OVERLAY:
                cv2.putText(frame, f"finger gaps: {finger_gaps}", (10, y := 25 + 20 * len(TARGET_EXPRESSIONS) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            if state == "RUNNING":
                if hand_seen:
                    cv2.putText(frame, "Open hand detected",
                                (20, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if sustained:
                    state = "PAUSED"
                    print("[GAME] Open hand sustained -> PAUSED.")
            else:  # PAUSED
                msg = "PAUSED - remove hand, press C to continue (Q to quit)"
                if hand_seen:
                    msg = "Remove hand to continue..."
                cv2.putText(frame, msg, (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            if SHOW_HAND_MASK_PREVIEW:
                draw_mask_preview(frame, mask)

            cv2.imshow("Expression Race", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and state == "PAUSED" and not hand_seen:
                state = "RUNNING"
                hand_stop.reset()
                smoother.reset()
                print("[GAME] Resumed.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()