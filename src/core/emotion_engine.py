"""Emotion recognition engine using HSEmotion AffectNet EfficientNet-B0 ONNX model."""

import time
import urllib.request  # Fix for hsemotion-onnx Python 3.12 urllib dependency
import logging
from typing import Dict, Tuple
import cv2
import numpy as np

try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
except ImportError:
    HSEmotionRecognizer = None

logger = logging.getLogger(__name__)

# Mapping from HSEmotion 8-class output to standard friendly names
EMOTION_NAME_MAP = {
    "Anger": "Angry",
    "Contempt": "Contempt",
    "Disgust": "Disgust",
    "Fear": "Fear",
    "Happiness": "Happy",
    "Neutral": "Neutral",
    "Sadness": "Sad",
    "Surprise": "Surprised",
}


class EmotionEngine:
    def __init__(
        self,
        model_name: str = "enet_b0_8_best_vgaf",
        sharpness_threshold: float = 5.0,
        tta_margin: float = 0.08,
        low_light_luma: float = 70.0,
        expression_gain: float = 1.6,
    ):
        self.model_name = model_name
        # Contrast-normalized sharpness below this => motion-blurred /
        # defocused crop, damped toward Neutral instead of trusted.
        # Calibrated against measured crops: a sharp 224x224 webcam face
        # scores ~13, Gaussian/motion blur scores 1-5. See _sharpness.
        self.sharpness_threshold = sharpness_threshold
        # When the top-2 margin is tighter than this, average a horizontally
        # flipped second pass (cheap test-time augmentation on hard frames).
        self.tta_margin = tta_margin
        # Mean gray below this => gentle CLAHE lift before inference.
        self.low_light_luma = low_light_luma
        # AffectNet models are Neutral-biased on webcam input (a person at a
        # screen reads ~0.75 Neutral at rest), which starves the directional
        # classes of probability mass. Scale non-Neutral classes by this gain
        # and renormalize so real expressions can clear the activation
        # threshold. 1.0 disables the correction.
        self.expression_gain = max(1.0, float(expression_gain))
        self.recognizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load HSEmotion ONNX model."""
        if HSEmotionRecognizer is None:
            raise RuntimeError("hsemotion-onnx is not installed. Please run pip install hsemotion-onnx")
        
        logger.info(f"Loading HSEmotion ONNX model ({self.model_name})...")
        try:
            self.recognizer = HSEmotionRecognizer(model_name=self.model_name)
            logger.info("HSEmotion model loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load model {self.model_name}: {exc}")
            raise

    @staticmethod
    def _sharpness(face_bgr: np.ndarray) -> float:
        """Contrast-normalized sharpness (higher = sharper).

        Raw Laplacian variance scales with image contrast, so a perfectly
        sharp *dim* face scores the same as a heavily motion-blurred bright
        one. Dividing by the gray variance removes the contrast term and
        leaves a pure focus measure. Measured on real 224x224 crops:
        sharp ~13, sharp-but-dim 15-23, Gaussian/motion blur 1-5.
        """
        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
            contrast = float(gray.var())
            if contrast <= 1.0:
                return 0.0  # featureless crop carries no expression signal
            return float(cv2.Laplacian(gray, cv2.CV_64F).var()) / contrast * 1000.0
        except Exception:
            return 0.0

    def _apply_expression_gain(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Counteract the model's Neutral prior, then renormalize to sum 1."""
        if self.expression_gain <= 1.0:
            return scores
        adjusted = {
            key: (val if key == "Neutral" else val * self.expression_gain)
            for key, val in scores.items()
        }
        total = sum(adjusted.values())
        if total <= 0.0:
            return scores
        return {key: val / total for key, val in adjusted.items()}

    def _normalize_illumination(self, face_bgr: np.ndarray) -> np.ndarray:
        """Gentle CLAHE lift for dim crops so shadows don't skew features."""
        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            if float(np.mean(gray)) >= self.low_light_luma:
                return face_bgr
            lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            merged = cv2.merge((clahe.apply(l_chan), a_chan, b_chan))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        except Exception:
            return face_bgr

    def _infer(self, face_bgr: np.ndarray):
        """Single forward pass; returns (raw_class, raw_scores list)."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return self.recognizer.predict_emotions(face_rgb, logits=False)

    def _to_scores_dict(self, raw_scores) -> Dict[str, float]:
        scores_dict: Dict[str, float] = {}
        for idx, score in enumerate(raw_scores):
            raw_name = self.recognizer.idx_to_class.get(idx, "Neutral")
            friendly_name = EMOTION_NAME_MAP.get(raw_name, raw_name)
            scores_dict[friendly_name] = float(score)
        return scores_dict

    def predict(self, face_bgr: np.ndarray) -> Tuple[str, float, Dict[str, float], float]:
        """
        Run emotion recognition on a cropped face BGR image.
        Returns:
            dominant_emotion: e.g. "Happy", "Sad", "Angry", "Surprised", "Neutral"
            dominant_score: float confidence in range [0.0, 1.0]
            scores_dict: dictionary mapping each friendly emotion name to its score
            latency_ms: time taken for inference in milliseconds
        """
        if face_bgr is None or face_bgr.size == 0:
            scores = {name: (1.0 if name == "Neutral" else 0.0) for name in EMOTION_NAME_MAP.values()}
            return "Neutral", 1.0, scores, 0.0

        t0 = time.perf_counter()

        # Normalize dim lighting, then run the base forward pass.
        face_bgr = self._normalize_illumination(face_bgr)
        _, raw_scores = self._infer(face_bgr)
        scores_dict = self._to_scores_dict(raw_scores)

        # Correct the Neutral prior before any tie-breaking, so the margin
        # and the dominant class are judged on the same scale the smoother
        # thresholds against.
        scores_dict = self._apply_expression_gain(scores_dict)

        # Selective flip-TTA: on near-tie frames, average a mirrored pass.
        # Faces are ~symmetric, so disagreement between views flags noise and
        # averaging measurably steadies the winner without slowing easy frames.
        # Skipped when Neutral simply leads (the common resting case) -- a
        # second pass there only doubles latency and halves the frame rate.
        ranked = sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True)
        margin = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 1.0
        if margin < self.tta_margin and ranked[0][0] != "Neutral":
            try:
                flipped = cv2.flip(face_bgr, 1)
                _, flip_scores = self._infer(flipped)
                flip_dict = self._to_scores_dict(flip_scores)
                flip_dict = self._apply_expression_gain(flip_dict)
                for key in scores_dict:
                    scores_dict[key] = 0.5 * (scores_dict[key] + flip_dict.get(key, scores_dict[key]))
            except Exception:
                pass

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Blur gate: heavily blurred crops carry little expression signal, so
        # damp them toward Neutral instead of letting a smeared frame spike a
        # directional command. The ramp is deliberately narrow -- only
        # genuinely defocused frames are damped, and an in-focus crop (however
        # dim) passes through untouched.
        sharpness = self._sharpness(face_bgr)
        if sharpness < self.sharpness_threshold:
            floor = 1.0  # at/below this the crop is featureless -> pure Neutral
            span = max(1e-6, self.sharpness_threshold - floor)
            weight = max(0.0, min(1.0, (sharpness - floor) / span))
            for key in scores_dict:
                target = 1.0 if key == "Neutral" else 0.0
                scores_dict[key] = weight * scores_dict[key] + (1.0 - weight) * target

        friendly_dominant = max(scores_dict, key=lambda k: scores_dict[k])
        dominant_score = scores_dict.get(friendly_dominant, 0.0)

        return friendly_dominant, dominant_score, scores_dict, latency_ms
