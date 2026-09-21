"""Face detection and alignment module using OpenCV YuNet with Haar fallback and robustness hardening."""

import os
import urllib.request
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "face_detection_yunet_2023mar.onnx")


@dataclass
class FaceInfo:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    score: float
    right_eye: Optional[Tuple[float, float]] = None
    left_eye: Optional[Tuple[float, float]] = None
    nose_tip: Optional[Tuple[float, float]] = None
    right_mouth: Optional[Tuple[float, float]] = None
    left_mouth: Optional[Tuple[float, float]] = None


class FaceDetector:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf_threshold: float = 0.6,
        bbox_alpha: float = 0.6,
        bbox_snap_iou: float = 0.3,
        min_face_px: int = 64,
        min_face_area_ratio: float = 0.015,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        # Bounding-box temporal smoothing: blend weight for the newest
        # detection. Kills 1-2px YuNet jitter so the emotion crop is stable.
        self.bbox_alpha = bbox_alpha
        # If the new box overlaps the track below this IoU, snap instead of
        # blending (handles jumps / new person entering the frame).
        self.bbox_snap_iou = bbox_snap_iou
        # Reliability gate: faces smaller than this are too low-signal to
        # drive the robot and would only inject noise.
        self.min_face_px = min_face_px
        self.min_face_area_ratio = min_face_area_ratio
        self.yunet: Optional[cv2.FaceDetectorYN] = None
        self.haar_cascade: Optional[cv2.CascadeClassifier] = None
        self.current_input_size: Tuple[int, int] = (320, 320)

        # Primary face tracking for multi-person consistency
        self.last_primary_bbox: Optional[Tuple[int, int, int, int]] = None
        self.frames_since_last_seen = 0
        # Smoothed track state (floats for stable blending)
        self.smoothed_bbox: Optional[Tuple[float, float, float, float]] = None
        self.smoothed_angle: Optional[float] = None
        self.smoothed_eye_center: Optional[Tuple[float, float]] = None

        self._init_detector()

    def _init_detector(self) -> None:
        """Initialize YuNet detector, downloading model if absent, with Haar fallback."""
        try:
            if not os.path.exists(self.model_path):
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                logger.info(f"Downloading YuNet model to {self.model_path}...")
                urllib.request.urlretrieve(YUNET_MODEL_URL, self.model_path)
                logger.info("YuNet model downloaded successfully.")

            self.yunet = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=self.current_input_size,
                score_threshold=self.conf_threshold,
                nms_threshold=0.3,
                top_k=5000,
            )
            logger.info("OpenCV YuNet face detector initialized successfully.")
        except Exception as exc:
            logger.warning(f"Failed to initialize YuNet ({exc}). Falling back to Haar cascade.")
            self.yunet = None
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)

    def _preprocess_lighting(self, image: np.ndarray) -> np.ndarray:
        """Adaptive CLAHE equalization when illumination is dim/underexposed (< 65)."""
        if image is None or image.size == 0 or len(image.shape) < 3:
            return image
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            mean_l = float(np.mean(l_chan))
            if mean_l < 65.0:
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                l_eq = clahe.apply(l_chan)
                equalized_lab = cv2.merge((l_eq, a_chan, b_chan))
                return cv2.cvtColor(equalized_lab, cv2.COLOR_LAB2BGR)
        except Exception:
            pass
        return image

    def _reset_track(self) -> None:
        """Clear all temporal track state after a prolonged absence."""
        self.last_primary_bbox = None
        self.smoothed_bbox = None
        self.smoothed_angle = None
        self.smoothed_eye_center = None

    def _smooth_bbox(self, raw: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """EMA-blend the new box into the track; snap on large jumps."""
        if self.smoothed_bbox is None:
            self.smoothed_bbox = tuple(float(v) for v in raw)
        else:
            prev_int = tuple(int(round(v)) for v in self.smoothed_bbox)
            if self._compute_iou(prev_int, raw) < self.bbox_snap_iou:
                self.smoothed_bbox = tuple(float(v) for v in raw)
            else:
                a = self.bbox_alpha
                self.smoothed_bbox = tuple(
                    s + a * (n - s) for s, n in zip(self.smoothed_bbox, raw)
                )
        return tuple(int(round(v)) for v in self.smoothed_bbox)

    def is_reliable(self, face: FaceInfo, frame_shape: Tuple[int, ...]) -> bool:
        """True when a face is large/clear enough for trustworthy emotion input."""
        try:
            _, _, bw, bh = face.bbox
            if bw < self.min_face_px or bh < self.min_face_px:
                return False
            h, w = int(frame_shape[0]), int(frame_shape[1])
            if h <= 0 or w <= 0:
                return False
            area_ratio = (bw * bh) / float(w * h)
            return area_ratio >= self.min_face_area_ratio
        except Exception:
            return False

    @staticmethod
    def _compute_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union between two bounding boxes (x, y, w, h)."""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / float(union_area)

    def detect(self, image: np.ndarray) -> List[FaceInfo]:
        """Detect faces in an image with channel normalization and primary driver lock."""
        if image is None or image.size == 0:
            self.frames_since_last_seen += 1
            if self.frames_since_last_seen > 15:
                self._reset_track()
            return []

        # Ensure 3-channel BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Apply adaptive low-light enhancement
        proc_img = self._preprocess_lighting(image)

        h, w = proc_img.shape[:2]
        faces_out: List[FaceInfo] = []

        if self.yunet is not None:
            if (w, h) != self.current_input_size:
                self.current_input_size = (w, h)
                self.yunet.setInputSize(self.current_input_size)

            _, faces = self.yunet.detect(proc_img)
            if faces is not None:
                for f in faces:
                    x, y, bw, bh = map(int, f[0:4])
                    re = (float(f[4]), float(f[5]))
                    le = (float(f[6]), float(f[7]))
                    nt = (float(f[8]), float(f[9]))
                    rm = (float(f[10]), float(f[11]))
                    lm = (float(f[12]), float(f[13]))
                    score = float(f[14])
                    if score >= self.conf_threshold:
                        # Safe coordinate clamping
                        cx = max(0, min(w - 1, x))
                        cy = max(0, min(h - 1, y))
                        cbw = max(1, min(w - cx, bw))
                        cbh = max(1, min(h - cy, bh))
                        faces_out.append(
                            FaceInfo(
                                bbox=(cx, cy, cbw, cbh),
                                score=score,
                                right_eye=re,
                                left_eye=le,
                                nose_tip=nt,
                                right_mouth=rm,
                                left_mouth=lm,
                            )
                        )
        elif self.haar_cascade is not None:
            gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            rects = self.haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            for (x, y, bw, bh) in rects:
                cx = max(0, min(w - 1, x))
                cy = max(0, min(h - 1, y))
                cbw = max(1, min(w - cx, bw))
                cbh = max(1, min(h - cy, bh))
                faces_out.append(FaceInfo(bbox=(cx, cy, cbw, cbh), score=0.9))

        if not faces_out:
            self.frames_since_last_seen += 1
            if self.frames_since_last_seen > 15:
                self._reset_track()
            return []

        # Sort initially by area descending
        faces_out.sort(key=lambda item: item.bbox[2] * item.bbox[3], reverse=True)

        # Primary Driver Lock: maintain tracking continuity across multiple faces
        if self.last_primary_bbox is not None and len(faces_out) > 1:
            best_idx = 0
            best_iou = -1.0
            for i, face in enumerate(faces_out):
                iou = self._compute_iou(face.bbox, self.last_primary_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            # If previous driver matched with reasonable overlap, keep as primary face
            if best_iou >= 0.25:
                faces_out.insert(0, faces_out.pop(best_idx))

        # Update tracking state
        self.last_primary_bbox = faces_out[0].bbox
        self.frames_since_last_seen = 0

        # Temporal box smoothing on the primary face so the emotion crop
        # doesn't jitter. Landmarks/scores stay raw (they're re-estimated).
        primary = faces_out[0]
        smooth_box = self._smooth_bbox(primary.bbox)
        faces_out[0] = FaceInfo(
            bbox=smooth_box,
            score=primary.score,
            right_eye=primary.right_eye,
            left_eye=primary.left_eye,
            nose_tip=primary.nose_tip,
            right_mouth=primary.right_mouth,
            left_mouth=primary.left_mouth,
        )

        return faces_out

    def align_and_crop(
        self, image: np.ndarray, face: FaceInfo, target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """Align face using eye landmarks (if available) and crop safely to target_size."""
        if image is None or image.size == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # Ensure 3-channel BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        h, w = image.shape[:2]
        x, y, bw, bh = face.bbox

        # If eye coordinates are available, perform affine rotation alignment
        # with temporally smoothed angle/center to avoid warp flicker.
        if face.right_eye and face.left_eye:
            rx, ry = face.right_eye
            lx, ly = face.left_eye
            dx = lx - rx
            dy = ly - ry
            angle = np.degrees(np.arctan2(dy, dx))

            # Center between eyes
            eye_center = ((rx + lx) / 2.0, (ry + ly) / 2.0)

            if self.smoothed_angle is None or self.smoothed_eye_center is None:
                self.smoothed_angle = float(angle)
                self.smoothed_eye_center = (float(eye_center[0]), float(eye_center[1]))
            else:
                center_jump = abs(eye_center[0] - self.smoothed_eye_center[0]) + abs(
                    eye_center[1] - self.smoothed_eye_center[1]
                )
                angle_jump = abs(float(angle) - self.smoothed_angle)
                jump_limit = max(bw, bh) * 0.6
                if angle_jump > 25.0 or center_jump > jump_limit:
                    # Snap: head moved fast or a different face took over
                    self.smoothed_angle = float(angle)
                    self.smoothed_eye_center = (float(eye_center[0]), float(eye_center[1]))
                else:
                    a = 0.5
                    self.smoothed_angle = self.smoothed_angle + a * (float(angle) - self.smoothed_angle)
                    self.smoothed_eye_center = (
                        self.smoothed_eye_center[0] + a * (eye_center[0] - self.smoothed_eye_center[0]),
                        self.smoothed_eye_center[1] + a * (eye_center[1] - self.smoothed_eye_center[1]),
                    )

            angle = self.smoothed_angle
            eye_center = self.smoothed_eye_center
            rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
            rotated = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)

            box_size = int(max(bw, bh) * 1.25)
            x1 = max(0, min(w - 1, int(eye_center[0] - box_size // 2)))
            y1 = max(0, min(h - 1, int(eye_center[1] - box_size // 2.2)))
            x2 = max(x1 + 1, min(w, x1 + box_size))
            y2 = max(y1 + 1, min(h, y1 + box_size))

            crop = rotated[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] > 0 and crop.shape[1] > 0:
                return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

        # Safe fallback crop with 15% margin and bounds protection
        margin_x = int(bw * 0.15)
        margin_y = int(bh * 0.15)
        x1 = max(0, min(w - 1, x - margin_x))
        y1 = max(0, min(h - 1, y - margin_y))
        x2 = max(x1 + 1, min(w, x + bw + margin_x))
        y2 = max(y1 + 1, min(h, y + bh + margin_y))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
