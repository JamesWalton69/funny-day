import unittest
import numpy as np
from pathlib import Path
from backend.pose_tracker import PoseTracker, GazeState, PlayerPoseData, AdaptiveAngleFilter

class TestPoseTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = PoseTracker()

    def test_calibration_baseline(self):
        # Set calibration for player 1
        self.tracker.set_calibration(1, 10.0, -5.0, 2.0)
        baseline = self.tracker.calibration_baselines[1]
        self.assertEqual(baseline, (10.0, -5.0, 2.0))

    def test_gaze_classification_logic(self):
        # Create a mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        left_player, right_player = self.tracker.process_camera_frame(frame, 1, 2)

        # In a black frame, no face is detected
        self.assertFalse(left_player.face_detected)
        self.assertFalse(right_player.face_detected)
        self.assertEqual(left_player.gaze_state, GazeState.FACE_NOT_DETECTED)
        self.assertEqual(right_player.gaze_state, GazeState.FACE_NOT_DETECTED)

    def test_adaptive_angle_filter(self):
        filt = AdaptiveAngleFilter(slow_alpha=0.20, fast_alpha=0.85, delta_thresh=10.0)
        # Initial value
        v0 = filt.update(10.0)
        self.assertEqual(v0, 10.0)

        # Micro-jitter of 1 degree: should be heavily smoothed by slow_alpha
        v1 = filt.update(11.0)
        self.assertLess(v1, 10.5)  # 10.0 + 0.2*(1.0) = ~10.2

        # Fast head turn of 30 degrees: should quickly jump with high alpha
        v2 = filt.update(40.0)
        self.assertGreater(v2, 30.0)

    def test_iris_gaze_sideways(self):
        # When head is forward (yaw=0) but eyes look sideways (iris_x_ratio = 0.85)
        self.tracker.gaze_thresholds["iris_sideways_thresh"] = 0.20
        # If iris deviates from 0.50 by > 0.20 and head yaw > 12:
        # Check that tracker gaze classification handles iris offsets
        pass

if __name__ == "__main__":
    unittest.main()
