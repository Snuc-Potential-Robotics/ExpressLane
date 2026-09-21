"""Comprehensive edge-case test suite for TemporalSmoother decision logic."""

import unittest
from src.core.temporal_smoother import TemporalSmoother


class TestEdgeCasesSmoother(unittest.TestCase):
    def setUp(self):
        self.smoother = TemporalSmoother(
            activation_threshold=0.55,
            deactivation_threshold=0.38,
            min_consecutive_frames=3,
            ambiguity_margin=0.10,
        )

    def test_ambiguous_flat_distribution_rejected(self):
        """When two non-neutral emotions are nearly tied, system stays in STOP."""
        # Happy 0.36 vs Surprised 0.34 (margin = 0.02, < 0.10)
        ambiguous_scores = {
            "Happy": 0.36,
            "Surprised": 0.34,
            "Neutral": 0.20,
            "Sad": 0.05,
            "Angry": 0.05,
        }
        for _ in range(5):
            emo, action, _, _ = self.smoother.process(ambiguous_scores, face_detected=True)
            self.assertEqual(emo, "Neutral")
            self.assertEqual(action, "STOP")

    def test_alternating_single_frame_flicker_never_drives(self):
        """Alternating noisy frames (1 frame Happy, 1 frame Angry, etc.) must never move bot."""
        flicker_sequence = [
            {"Happy": 0.9, "Neutral": 0.1},
            {"Angry": 0.9, "Neutral": 0.1},
            {"Surprised": 0.9, "Neutral": 0.1},
            {"Sad": 0.9, "Neutral": 0.1},
            {"Happy": 0.9, "Neutral": 0.1},
            {"Angry": 0.9, "Neutral": 0.1},
        ]
        for scores in flicker_sequence:
            _, action, _, _ = self.smoother.process(scores, face_detected=True)
            self.assertEqual(action, "STOP")

    def test_rapid_release_halts_immediately(self):
        """When user relaxes back to Neutral, system halts immediately on the next frame."""
        happy_scores = {"Happy": 0.95, "Neutral": 0.05}
        for _ in range(3):
            self.smoother.process(happy_scores, face_detected=True)
        self.assertEqual(self.smoother.active_action, "FORWARD")

        # User relaxes expression to Neutral
        neutral_scores = {"Happy": 0.10, "Neutral": 0.90}
        emo, action, _, _ = self.smoother.process(neutral_scores, face_detected=True)
        # Immediate halt without waiting 3 debounce frames!
        self.assertEqual(emo, "Neutral")
        self.assertEqual(action, "STOP")

    def test_smooth_directional_transition(self):
        """Holding Happy, then holding Angry transitions cleanly from FORWARD to LEFT."""
        happy_scores = {"Happy": 0.95, "Neutral": 0.05, "Angry": 0.0}
        for _ in range(3):
            self.smoother.process(happy_scores, face_detected=True)
        self.assertEqual(self.smoother.active_action, "FORWARD")

        angry_scores = {"Angry": 0.95, "Neutral": 0.05, "Happy": 0.0}
        # Hold Angry for 4 frames (1 frame for EMA crossover + 3 debounce frames)
        for _ in range(4):
            self.smoother.process(angry_scores, face_detected=True)
        self.assertEqual(self.smoother.active_emotion, "Angry")
        self.assertEqual(self.smoother.active_action, "LEFT")

    def test_face_disappearance_instant_failsafe(self):
        """Brief camera flicker holds; sustained disappearance failsafes STOP."""
        happy_scores = {"Happy": 0.95, "Neutral": 0.05}
        for _ in range(3):
            self.smoother.process(happy_scores, face_detected=True)
        self.assertEqual(self.smoother.active_action, "FORWARD")

        # 1-2 dropped frames must ride through without stuttering
        emo, action, _, _ = self.smoother.process(None, face_detected=False)
        self.assertEqual(emo, "Happy")
        self.assertEqual(action, "FORWARD")

        # Camera feed truly interrupted -> failsafe STOP after grace window
        for _ in range(self.smoother.max_missed_frames + 1):
            emo, action, conf, _ = self.smoother.process(None, face_detected=False)
        self.assertEqual(emo, "Neutral")
        self.assertEqual(action, "STOP")
        self.assertEqual(conf, 1.0)


if __name__ == "__main__":
    unittest.main()
