"""Stability / anti-glitch test suite for the hardened perception pipeline.

Covers: dropout grace boundaries, two-frame burst rejection, low-confidence
flat-line rejection, bbox track smoothing, reliability gating, and blur-gated
emotion damping.
"""

import unittest
import cv2
import numpy as np

from src.core.temporal_smoother import TemporalSmoother
from src.core.detector import FaceDetector, FaceInfo


def _scores(dominant: str, strength: float = 0.9) -> dict:
    base = {"Happy": 0.0, "Sad": 0.0, "Angry": 0.0, "Surprised": 0.0,
            "Neutral": 0.0, "Fear": 0.0, "Disgust": 0.0, "Contempt": 0.0}
    base[dominant] = strength
    base["Neutral"] = round(1.0 - strength, 3)
    return base


class TestDropoutGrace(unittest.TestCase):
    def setUp(self):
        self.smoother = TemporalSmoother(max_missed_frames=3)

    def test_grace_boundary_holds_then_failsafes(self):
        for _ in range(4):
            self.smoother.process(_scores("Happy"), face_detected=True)
        self.assertEqual(self.smoother.active_action, "FORWARD")

        # Exactly max_missed_frames misses must still hold
        for _ in range(3):
            emo, action, _, _ = self.smoother.process(None, face_detected=False)
            self.assertEqual(action, "FORWARD")

        # One more miss trips the failsafe
        emo, action, conf, _ = self.smoother.process(None, face_detected=False)
        self.assertEqual(emo, "Neutral")
        self.assertEqual(action, "STOP")
        self.assertEqual(conf, 1.0)

    def test_recovery_after_brief_dropout(self):
        for _ in range(4):
            self.smoother.process(_scores("Happy"), face_detected=True)
        self.smoother.process(None, face_detected=False)
        self.smoother.process(None, face_detected=False)
        # Face returns: still driving without re-debounce delay
        emo, action, _, _ = self.smoother.process(_scores("Happy"), face_detected=True)
        self.assertEqual(emo, "Happy")
        self.assertEqual(action, "FORWARD")


class TestBurstRejection(unittest.TestCase):
    def setUp(self):
        self.smoother = TemporalSmoother(
            activation_threshold=0.55,
            deactivation_threshold=0.38,
            min_consecutive_frames=3,
        )

    def test_two_frame_bursts_never_drive(self):
        """2-frame bursts (below the 3-frame debounce) must never move the bot."""
        burst = [_scores("Happy"), _scores("Happy"),
                 _scores("Angry"), _scores("Angry"),
                 _scores("Surprised"), _scores("Surprised"),
                 _scores("Sad"), _scores("Sad")] * 3
        transitions = 0
        last_action = "STOP"
        for scores in burst:
            _, action, _, _ = self.smoother.process(scores, face_detected=True)
            self.assertEqual(action, "STOP")
            if action != last_action:
                transitions += 1
                last_action = action
        self.assertEqual(transitions, 0)

    def test_flat_low_confidence_line_never_drives(self):
        """A flat uncertain distribution must never activate motion."""
        flat = {"Happy": 0.22, "Sad": 0.18, "Angry": 0.15, "Surprised": 0.15,
                "Neutral": 0.20, "Fear": 0.05, "Disgust": 0.03, "Contempt": 0.02}
        for _ in range(10):
            emo, action, _, _ = self.smoother.process(dict(flat), face_detected=True)
            self.assertEqual(action, "STOP")
            self.assertEqual(emo, "Neutral")

    def test_sustained_expression_still_fast(self):
        """Legit expressions must engage within 4 frames (no sluggishness)."""
        for i in range(4):
            emo, action, _, _ = self.smoother.process(_scores("Sad"), face_detected=True)
            if i >= 2:
                self.assertEqual(emo, "Sad")
                self.assertEqual(action, "BACKWARD")


class TestCleanSwitches(unittest.TestCase):
    def setUp(self):
        self.smoother = TemporalSmoother()

    def test_deliberate_switch_has_no_stop_dip(self):
        """Happy -> Angry must crossfade FORWARD -> LEFT without a STOP dip."""
        seq = [_scores("Happy")] * 5 + [_scores("Angry")] * 5
        actions = [self.smoother.process(s, face_detected=True)[1] for s in seq]
        self.assertEqual(actions[-1], "LEFT")
        # After the first activation, STOP must never reappear mid-run
        first_drive = next(i for i, a in enumerate(actions) if a != "STOP")
        self.assertNotIn("STOP", actions[first_drive:])

    def test_faded_expression_with_no_alternative_releases(self):
        """Active emotion fading with nothing strong behind it must release."""
        for _ in range(4):
            self.smoother.process(_scores("Happy"), face_detected=True)
        self.assertEqual(self.smoother.active_action, "FORWARD")
        weak = {"Happy": 0.30, "Sad": 0.10, "Angry": 0.10, "Surprised": 0.10,
                "Neutral": 0.30, "Fear": 0.03, "Disgust": 0.03, "Contempt": 0.04}
        emo, action, _, _ = self.smoother.process(dict(weak), face_detected=True)
        self.assertEqual(emo, "Neutral")
        self.assertEqual(action, "STOP")


class TestBboxTrack(unittest.TestCase):
    def setUp(self):
        self.detector = FaceDetector()

    def test_jitter_is_damped(self):
        """Small box jitter must shrink through the EMA track."""
        raw_boxes = [(100 + ((i * 7) % 5 - 2), 100 + ((i * 11) % 5 - 2), 120, 120)
                     for i in range(12)]
        smoothed = [self.detector._smooth_bbox(b) for b in raw_boxes]
        raw_var = float(np.var([b[0] for b in raw_boxes]) + np.var([b[1] for b in raw_boxes]))
        sm_var = float(np.var([b[0] for b in smoothed]) + np.var([b[1] for b in smoothed]))
        self.assertLess(sm_var, raw_var)

    def test_large_jump_snaps(self):
        """A big jump (new person) must snap, not drag the old track."""
        self.detector._smooth_bbox((100, 100, 120, 120))
        snapped = self.detector._smooth_bbox((400, 300, 120, 120))
        self.assertEqual(snapped, (400, 300, 120, 120))

    def test_reliability_gate(self):
        frame_shape = (480, 640, 3)
        tiny = FaceInfo(bbox=(10, 10, 40, 40), score=0.9)
        solid = FaceInfo(bbox=(200, 150, 140, 140), score=0.9)
        self.assertFalse(self.detector.is_reliable(tiny, frame_shape))
        self.assertTrue(self.detector.is_reliable(solid, frame_shape))

    def test_alignment_jitter_stays_shaped(self):
        """Slightly jittered landmarks must still yield a valid crop."""
        img = np.full((480, 640, 3), 110, dtype=np.uint8)
        for i in range(4):
            face = FaceInfo(
                bbox=(200 + i, 150, 120, 120), score=0.92,
                right_eye=(230.0 + i, 180.0), left_eye=(280.0 - i, 180.0 + i),
                nose_tip=(255.0, 205.0),
            )
            crop = self.detector.align_and_crop(img, face, (224, 224))
            self.assertEqual(crop.shape, (224, 224, 3))


class TestBlurGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.core.emotion_engine import EmotionEngine
        cls.engine = EmotionEngine()

    def test_heavily_blurred_crop_damps_to_neutral(self):
        """A featureless blurred crop must not spike a directional command."""
        flat = np.full((224, 224, 3), 128, dtype=np.uint8)
        dom, score, scores, latency = self.engine.predict(flat)
        self.assertEqual(dom, "Neutral")
        self.assertGreater(latency, 0.0)
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=3)

    def test_output_distribution_stays_normalized(self):
        rng = np.random.RandomState(7)
        noisy = rng.randint(40, 220, (224, 224, 3)).astype(np.uint8)
        _, _, scores, _ = self.engine.predict(noisy)
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=3)
        for v in scores.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


class TestSharpnessCalibration(unittest.TestCase):
    """Regression guards for the emotion-gate calibration.

    The gate previously used raw Laplacian variance against a threshold of
    35. A sharp real webcam crop measures ~13 on that scale, so *every*
    frame was treated as blurred and pulled ~60% toward Neutral -- no
    expression could ever reach the activation threshold. Raw variance also
    scales with contrast, so a sharp-but-dim face scored the same as a
    smeared one.
    """

    @classmethod
    def setUpClass(cls):
        from src.core.emotion_engine import EmotionEngine
        cls.cls_engine = EmotionEngine
        rng = np.random.RandomState(11)
        # Face-like crop: band-limited texture over smooth shading, i.e. a
        # natural image spectrum rather than hard synthetic step edges
        # (which survive blurring and would not exercise the gate).
        tex = cv2.GaussianBlur(
            rng.randint(0, 255, (224, 224)).astype(np.float32), (5, 5), 0
        )
        tex = cv2.normalize(tex, None, 60, 200, cv2.NORM_MINMAX)
        yy, xx = np.mgrid[0:224, 0:224]
        shading = 40 * np.sin(xx / 70.0) + 30 * np.cos(yy / 55.0)
        gray = np.clip(tex + shading, 0, 255).astype(np.uint8)
        cls.sharp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_sharp_crop_is_above_threshold(self):
        """An in-focus crop must pass the gate untouched."""
        engine = self.cls_engine
        self.assertGreater(engine._sharpness(self.sharp), 5.0)

    def test_blur_is_separated_from_sharp(self):
        """Sharpness must fall monotonically with blur, crossing the gate."""
        scores = [
            self.cls_engine._sharpness(cv2.GaussianBlur(self.sharp, (k, k), 0))
            for k in (5, 9, 15)
        ]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[0], 5.0)   # mild blur still usable
        self.assertLess(scores[-1], 5.0)     # heavy blur is gated out

    def test_dim_and_bright_score_alike(self):
        """The metric must measure focus only, independent of exposure."""
        bright = self.cls_engine._sharpness(self.sharp)
        dim = self.cls_engine._sharpness((self.sharp * 0.25).astype(np.uint8))
        self.assertAlmostEqual(dim / bright, 1.0, delta=0.25)

    def test_dim_but_sharp_is_not_treated_as_blur(self):
        """Low light must not be mistaken for defocus (the contrast confound)."""
        dim = (self.sharp.astype(np.float64) * 0.25).astype(np.uint8)
        self.assertGreater(self.cls_engine._sharpness(dim), 5.0)
        # And it must beat a genuinely blurred (but bright) crop.
        blurred = cv2.GaussianBlur(self.sharp, (15, 15), 0)
        self.assertGreater(
            self.cls_engine._sharpness(dim), self.cls_engine._sharpness(blurred)
        )

    def test_expression_gain_preserves_normalization(self):
        from src.core.emotion_engine import EmotionEngine
        engine = EmotionEngine.__new__(EmotionEngine)
        engine.expression_gain = 1.6
        scores = {"Neutral": 0.75, "Angry": 0.10, "Sad": 0.08, "Happy": 0.07}
        out = engine._apply_expression_gain(scores)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=6)
        # Neutral still wins at rest -- the gain must not invent expressions.
        self.assertEqual(max(out, key=out.get), "Neutral")
        # ...but the expression classes gain ground on it.
        self.assertGreater(out["Angry"] / out["Neutral"], scores["Angry"] / scores["Neutral"])


class TestHoldStability(unittest.TestCase):
    """A held expression must not be chopped up by single noisy frames."""

    def setUp(self):
        self.smoother = TemporalSmoother()

    def test_single_neutral_blip_does_not_break_hold(self):
        for _ in range(4):
            self.smoother.process(_scores("Happy"), face_detected=True)
        self.assertEqual(self.smoother.active_action, "FORWARD")
        # One mid-expression frame where Neutral edges ahead (a blink, a
        # partial relax). Previously this dropped to STOP and forced a full
        # 3-frame re-debounce -- the visible stutter.
        blip = {"Happy": 0.38, "Neutral": 0.58, "Sad": 0.02, "Angry": 0.02}
        _, action, _, _ = self.smoother.process(blip, face_detected=True)
        self.assertEqual(action, "FORWARD")
        # The expression resuming keeps it driving.
        _, action, _, _ = self.smoother.process(_scores("Happy"), face_detected=True)
        self.assertEqual(action, "FORWARD")

    def test_sustained_neutral_still_releases(self):
        """The blip tolerance must not defeat a genuine relaxation."""
        for _ in range(4):
            self.smoother.process(_scores("Happy"), face_detected=True)
        calm = {"Happy": 0.30, "Neutral": 0.62, "Sad": 0.04, "Angry": 0.04}
        actions = [self.smoother.process(dict(calm), face_detected=True)[1] for _ in range(4)]
        self.assertEqual(actions[-1], "STOP")

    def test_borderline_candidate_still_activates(self):
        """Ambiguity must block a frame, not permanently stall the debounce.

        Frames alternate between a clear Angry and a near-tie. The guard
        previously rewrote the candidate to Neutral on ambiguous frames,
        resetting the counter, so the candidate could never accumulate the
        consecutive frames it needed and the bot never moved.
        """
        clear = {"Angry": 0.72, "Neutral": 0.16, "Sad": 0.06, "Happy": 0.06}
        tie = {"Angry": 0.44, "Sad": 0.40, "Neutral": 0.10, "Happy": 0.06}
        actions = []
        for i in range(10):
            scores = clear if i % 2 == 0 else tie
            actions.append(self.smoother.process(dict(scores), face_detected=True)[1])
        self.assertIn("LEFT", actions)


class TestTurnSlowAndPrecise(unittest.TestCase):
    def setUp(self):
        self.smoother = TemporalSmoother()

    def test_turn_speed_is_slower_and_distinct_from_linear(self):
        """Web server must configure turning speed to be slower and precise."""
        from src.web.server import RobotVisionSystem
        sys = RobotVisionSystem()
        self.assertLess(sys.turn_speed, sys.default_speed)
        self.assertEqual(sys.get_action_speed("FORWARD"), sys.default_speed)
        self.assertEqual(sys.get_action_speed("BACKWARD"), sys.default_speed)
        self.assertEqual(sys.get_action_speed("LEFT"), sys.turn_speed)
        self.assertEqual(sys.get_action_speed("RIGHT"), sys.turn_speed)
        self.assertEqual(sys.get_action_speed("STOP"), 0)

    def test_turn_precise_relaxation_halts_immediately(self):
        """Relaxing expression during a turn halts immediately on the next frame to prevent overshoot."""
        for _ in range(4):
            self.smoother.process(_scores("Angry"), face_detected=True)
        self.assertEqual(self.smoother.active_action, "LEFT")

        # User relaxes expression (moderate Neutral surge ~0.65)
        relaxed = {"Angry": 0.20, "Neutral": 0.65, "Happy": 0.05, "Sad": 0.05,
                   "Surprised": 0.05, "Fear": 0.0, "Disgust": 0.0, "Contempt": 0.0}
        emo, action, _, _ = self.smoother.process(relaxed, face_detected=True)
        self.assertEqual(emo, "Neutral")
        self.assertEqual(action, "STOP")

    def test_right_turn_precise_stop(self):
        """Surprised -> RIGHT also halts on a dime when relaxed."""
        for _ in range(4):
            self.smoother.process(_scores("Surprised"), face_detected=True)
        self.assertEqual(self.smoother.active_action, "RIGHT")

        relaxed = {"Surprised": 0.15, "Neutral": 0.65, "Happy": 0.05, "Sad": 0.05,
                   "Angry": 0.05, "Fear": 0.0, "Disgust": 0.0, "Contempt": 0.0}
        emo, action, _, _ = self.smoother.process(relaxed, face_detected=True)
        self.assertEqual(emo, "Neutral")
        self.assertEqual(action, "STOP")

    def test_turn_dropout_grace_tight_to_avoid_blind_spin(self):
        """Turns hold for at most 1 dropped frame, then failsafe STOP to avoid spinning blind."""
        for _ in range(4):
            self.smoother.process(_scores("Angry"), face_detected=True)
        self.assertEqual(self.smoother.active_action, "LEFT")

        # 1 missed frame rides through
        emo, action, _, _ = self.smoother.process(None, face_detected=False)
        self.assertEqual(action, "LEFT")

        # 2nd missed frame immediately failsafes to STOP (unlike linear which waits 3)
        emo, action, _, _ = self.smoother.process(None, face_detected=False)
        self.assertEqual(action, "STOP")
        self.assertEqual(emo, "Neutral")


if __name__ == "__main__":
    unittest.main()
