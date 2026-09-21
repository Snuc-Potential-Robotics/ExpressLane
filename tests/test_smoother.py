"""Automated tests for TemporalSmoother debouncing and hysteresis logic."""

import unittest
from src.core.temporal_smoother import TemporalSmoother


class TestTemporalSmoother(unittest.TestCase):
    def setUp(self):
        self.smoother = TemporalSmoother(
            activation_threshold=0.60,
            deactivation_threshold=0.35,
            min_consecutive_frames=3,
        )

    def test_single_frame_spike_rejected(self):
        """A single frame spike should NOT trigger state change from Neutral."""
        spike_scores = {"Happy": 0.95, "Neutral": 0.05, "Sad": 0.0, "Angry": 0.0, "Surprised": 0.0}
        emo, action, conf, _ = self.smoother.process(spike_scores, face_detected=True)
        # Should stay Neutral on frame 1
        self.assertEqual(emo, "Neutral")
        self.assertEqual(action, "STOP")

    def test_sustained_emotion_activates(self):
        """Sustained emotion for 3 consecutive frames activates the motion command."""
        happy_scores = {"Happy": 0.90, "Neutral": 0.10, "Sad": 0.0, "Angry": 0.0, "Surprised": 0.0}
        # Frame 1
        self.smoother.process(happy_scores, face_detected=True)
        # Frame 2
        self.smoother.process(happy_scores, face_detected=True)
        # Frame 3: threshold reached and held >= 3 frames
        emo, action, conf, _ = self.smoother.process(happy_scores, face_detected=True)
        self.assertEqual(emo, "Happy")
        self.assertEqual(action, "FORWARD")

    def test_face_lost_resets_to_stop(self):
        """Brief dropout holds the command; sustained loss failsafes to STOP."""
        happy_scores = {"Happy": 0.90, "Neutral": 0.10}
        for _ in range(3):
            self.smoother.process(happy_scores, face_detected=True)

        # 1-2 missed frames (detector flicker) must NOT stutter the robot
        emo, action, _, _ = self.smoother.process(None, face_detected=False)
        self.assertEqual(emo, "Happy")
        self.assertEqual(action, "FORWARD")

        # Sustained loss beyond the grace window must failsafe to STOP
        for _ in range(self.smoother.max_missed_frames + 1):
            emo, action, conf, _ = self.smoother.process(None, face_detected=False)
        self.assertEqual(emo, "Neutral")
        self.assertEqual(action, "STOP")

    def test_remapping_updates_action(self):
        """Updating action mapping dynamically reflects in output."""
        self.smoother.update_mapping({"Happy": "BACKWARD"})
        happy_scores = {"Happy": 0.95, "Neutral": 0.05}
        for _ in range(3):
            self.smoother.process(happy_scores, face_detected=True)
        emo, action, _, _ = self.smoother.process(happy_scores, face_detected=True)
        self.assertEqual(action, "BACKWARD")


if __name__ == "__main__":
    unittest.main()
