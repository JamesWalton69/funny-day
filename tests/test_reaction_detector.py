import unittest
import time
from backend.pose_tracker import PlayerPoseData, GazeState
from backend.reaction_detector import ReactionDetector, EventType

class TestReactionDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ReactionDetector()

    def test_sudden_facial_movement_mar(self):
        pid = 1
        # Frame 1: neutral mouth
        p1 = PlayerPoseData(player_id=pid, slot="left", face_detected=True, gaze_state=GazeState.FACING_TV, mouth_aspect_ratio=0.1)
        self.detector.check_player_reactions(p1, game_time_sec=0.0)

        time.sleep(0.05)
        # Frame 2: sudden jaw drop / laughing (high delta)
        p2 = PlayerPoseData(player_id=pid, slot="left", face_detected=True, gaze_state=GazeState.FACING_TV, mouth_aspect_ratio=0.6)
        events = self.detector.check_player_reactions(p2, game_time_sec=0.05)

        self.assertTrue(any(e.event_type == EventType.FACIAL_MOVEMENT for e in events))
        facial_ev = next(e for e in events if e.event_type == EventType.FACIAL_MOVEMENT)
        self.assertGreater(facial_ev.intensity, 0.0)
        self.assertLessEqual(facial_ev.intensity, 1.0)

    def test_blendshape_laughter_detection(self):
        pid = 3
        # Frame 1: neutral resting expression
        p1 = PlayerPoseData(
            player_id=pid, slot="left", face_detected=True, gaze_state=GazeState.FACING_TV,
            blendshapes={"mouthSmileLeft": 0.02, "mouthSmileRight": 0.02, "jawOpen": 0.05}
        )
        self.detector.check_player_reactions(p1, game_time_sec=0.0)

        time.sleep(0.05)
        # Frame 2: sudden laugh / big smile
        p2 = PlayerPoseData(
            player_id=pid, slot="left", face_detected=True, gaze_state=GazeState.FACING_TV,
            blendshapes={"mouthSmileLeft": 0.65, "mouthSmileRight": 0.70, "jawOpen": 0.40}
        )
        events = self.detector.check_player_reactions(p2, game_time_sec=0.05)
        self.assertTrue(any(e.event_type == EventType.FACIAL_MOVEMENT for e in events))
        ev = next(e for e in events if e.event_type == EventType.FACIAL_MOVEMENT)
        self.assertIn("Laughter / Smile", ev.description)

    def test_head_flinch_detection(self):
        pid = 4
        # Frame 1: looking straight
        p1 = PlayerPoseData(player_id=pid, slot="right", face_detected=True, calibrated_yaw=0.0, calibrated_pitch=0.0)
        self.detector.check_player_reactions(p1, game_time_sec=0.0)

        time.sleep(0.05)
        # Frame 2: sudden head flinch (25 deg jerk)
        p2 = PlayerPoseData(player_id=pid, slot="right", face_detected=True, calibrated_yaw=25.0, calibrated_pitch=-10.0)
        events = self.detector.check_player_reactions(p2, game_time_sec=0.05)
        self.assertTrue(any(e.event_type == EventType.HEAD_MOVEMENT for e in events))

    def test_looking_away_detection(self):
        pid = 2
        # Frame 1: Facing TV
        p1 = PlayerPoseData(player_id=pid, slot="right", face_detected=True, gaze_state=GazeState.FACING_TV)
        self.detector.check_player_reactions(p1, game_time_sec=1.0)

        time.sleep(0.05)
        # Frame 2: Looking sideways
        p2 = PlayerPoseData(player_id=pid, slot="right", face_detected=True, gaze_state=GazeState.LOOKING_SIDEWAYS, calibrated_yaw=35.0)
        events = self.detector.check_player_reactions(p2, game_time_sec=1.05)

        self.assertTrue(any(e.event_type == EventType.LOOK_AWAY for e in events))

if __name__ == "__main__":
    unittest.main()
