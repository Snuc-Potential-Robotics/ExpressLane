"""Comprehensive edge-case test suite for Computer Vision & Face Detection pipeline."""

import unittest
import numpy as np
import cv2
from src.core.detector import FaceDetector, FaceInfo


class TestEdgeCasesVision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = FaceDetector()

    def test_pitch_black_frame(self):
        """Pitch black underexposed frame should not raise exceptions."""
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = self.detector.detect(black_img)
        self.assertIsInstance(faces, list)
        self.assertEqual(len(faces), 0)

    def test_pure_white_overexposed_frame(self):
        """Pure white frame should not raise exceptions."""
        white_img = np.full((480, 640, 3), 255, dtype=np.uint8)
        faces = self.detector.detect(white_img)
        self.assertIsInstance(faces, list)
        self.assertEqual(len(faces), 0)

    def test_none_and_empty_images(self):
        """None and empty image arrays must safely return empty lists."""
        self.assertEqual(self.detector.detect(None), [])
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(self.detector.detect(empty_img), [])

    def test_grayscale_single_channel_image(self):
        """Single-channel 2D grayscale image is safely handled."""
        gray_img = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        faces = self.detector.detect(gray_img)
        self.assertIsInstance(faces, list)

    def test_four_channel_bgra_image(self):
        """4-channel BGRA image is safely handled."""
        bgra_img = np.random.randint(0, 255, (300, 300, 4), dtype=np.uint8)
        faces = self.detector.detect(bgra_img)
        self.assertIsInstance(faces, list)

    def test_non_standard_aspect_ratios(self):
        """Tests varying aspect ratios: 16:9, 1:1, 9:16 vertical, 4:3."""
        ratios = [(1080, 1920), (512, 512), (1280, 720), (240, 320)]
        for h, w in ratios:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            faces = self.detector.detect(frame)
            self.assertIsInstance(faces, list)

    def test_border_clipping_align_and_crop(self):
        """Faces located at or outside frame borders must crop safely to target size."""
        img = np.full((480, 640, 3), 120, dtype=np.uint8)

        # 1. Top-Left corner clipping
        face_tl = FaceInfo(bbox=(0, 0, 80, 80), score=0.95)
        crop_tl = self.detector.align_and_crop(img, face_tl, (224, 224))
        self.assertEqual(crop_tl.shape, (224, 224, 3))

        # 2. Bottom-Right corner clipping (extending past width and height)
        face_br = FaceInfo(bbox=(600, 440, 100, 100), score=0.95)
        crop_br = self.detector.align_and_crop(img, face_br, (224, 224))
        self.assertEqual(crop_br.shape, (224, 224, 3))

    def test_extreme_head_roll_tilt_alignment(self):
        """Test eye landmark alignment with +45 and -45 degree tilted head poses."""
        img = np.full((480, 640, 3), 100, dtype=np.uint8)
        # Create a face with eye coordinates tilted at 45 degrees
        face_tilted = FaceInfo(
            bbox=(200, 150, 120, 120),
            score=0.92,
            right_eye=(230.0, 180.0),
            left_eye=(280.0, 230.0),  # dy = +50, dx = +50 -> 45 degrees
            nose_tip=(250.0, 210.0),
        )
        crop = self.detector.align_and_crop(img, face_tilted, (224, 224))
        self.assertEqual(crop.shape, (224, 224, 3))

    def test_low_light_clahe_preprocessing(self):
        """Verify that dim images (< 65 luminance) are enhanced by CLAHE."""
        dim_img = np.full((200, 200, 3), 20, dtype=np.uint8)  # Very dim
        enhanced = self.detector._preprocess_lighting(dim_img)
        self.assertEqual(enhanced.shape, dim_img.shape)

    def test_iou_computation(self):
        """Test bounding box Intersection over Union computation."""
        b1 = (10, 10, 50, 50)
        b2 = (10, 10, 50, 50)
        self.assertAlmostEqual(self.detector._compute_iou(b1, b2), 1.0)

        # Completely disjoint boxes
        b3 = (200, 200, 50, 50)
        self.assertAlmostEqual(self.detector._compute_iou(b1, b3), 0.0)


if __name__ == "__main__":
    unittest.main()
