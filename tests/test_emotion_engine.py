"""Automated tests for EmotionEngine."""

import unittest
import numpy as np
from src.core.emotion_engine import EmotionEngine, EMOTION_NAME_MAP


class TestEmotionEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = EmotionEngine()

    def test_predict_shape_and_keys(self):
        """Verify prediction outputs on a synthetic 224x224 RGB image."""
        dummy_face = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        dom_emo, dom_score, scores, latency_ms = self.engine.predict(dummy_face)

        self.assertIsInstance(dom_emo, str)
        self.assertGreaterEqual(dom_score, 0.0)
        self.assertLessEqual(dom_score, 1.0)
        self.assertGreater(latency_ms, 0.0)

        # Check all expected friendly emotions exist
        expected_emotions = set(EMOTION_NAME_MAP.values())
        for emo in expected_emotions:
            self.assertIn(emo, scores)
            self.assertGreaterEqual(scores[emo], 0.0)
            self.assertLessEqual(scores[emo], 1.0)

    def test_predict_empty_image(self):
        """Verify graceful fallback on empty image."""
        dom_emo, dom_score, scores, latency_ms = self.engine.predict(np.zeros((0, 0, 3), dtype=np.uint8))
        self.assertEqual(dom_emo, "Neutral")
        self.assertEqual(dom_score, 1.0)


if __name__ == "__main__":
    unittest.main()
