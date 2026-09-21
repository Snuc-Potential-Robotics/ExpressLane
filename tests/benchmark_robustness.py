"""Benchmark stress test for vision pipeline, emotion engine, and smoother."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.core.detector import FaceDetector, FaceInfo
from src.core.emotion_engine import EmotionEngine
from src.core.temporal_smoother import TemporalSmoother


def run_benchmark(iterations: int = 100):
    print("================================================================")
    print("  AI FACE EMOTION BOT - ROBUSTNESS & PERFORMANCE BENCHMARK     ")
    print("================================================================")

    detector = FaceDetector()
    engine = EmotionEngine()
    smoother = TemporalSmoother()

    latencies = []
    actions_count = {}

    print(f"Running {iterations} stress iterations with synthetic transforms...")

    for i in range(iterations):
        # Generate synthetic frames with varying conditions
        base = np.random.randint(40, 220, (480, 640, 3), dtype=np.uint8)

        # Inject periodic edge cases:
        if i % 10 == 0:
            base = np.zeros((480, 640, 3), dtype=np.uint8)  # Black frame
        elif i % 10 == 1:
            base = np.full((480, 640, 3), 255, dtype=np.uint8)  # White frame
        elif i % 10 == 2:
            base = (base * 0.2).astype(np.uint8)  # Underexposed dim frame

        t0 = time.perf_counter()

        # Simulate face info with varying bounding boxes and eye angles
        angle_rad = (i % 8 - 4) * 0.1  # -0.4 to +0.4 rad (~ -23° to +23°)
        dx = 60 * np.cos(angle_rad)
        dy = 60 * np.sin(angle_rad)
        face = FaceInfo(
            bbox=(150 + (i % 20), 100 + (i % 20), 120, 120),
            score=0.92,
            right_eye=(200.0, 140.0),
            left_eye=(200.0 + dx, 140.0 + dy),
            nose_tip=(230.0, 170.0),
            right_mouth=(210.0, 200.0),
            left_mouth=(250.0, 200.0),
        )

        crop = detector.align_and_crop(base, face)
        dom_emo, dom_score, scores, _ = engine.predict(crop)
        emo, action, conf, _ = smoother.process(scores, face_detected=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        actions_count[action] = actions_count.get(action, 0) + 1

    avg_lat = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0

    print("----------------------------------------------------------------")
    print(f"  Iterations Completed:       {iterations}")
    print(f"  Mean Latency per Frame:     {avg_lat:.2f} ms")
    print(f"  Median (p50) Latency:       {p50:.2f} ms")
    print(f"  95th Percentile (p95):      {p95:.2f} ms")
    print(f"  99th Percentile (p99):      {p99:.2f} ms")
    print(f"  Effective Processing Speed: {fps:.1f} FPS")
    print("----------------------------------------------------------------")
    print(f"  Actions Distribution:       {actions_count}")
    print("================================================================")
    print("  [PASSED] Stress & edge-case benchmark completed successfully!  ")
    print("================================================================")


if __name__ == "__main__":
    run_benchmark(100)
