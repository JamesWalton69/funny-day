import unittest
from backend.config import DEFAULT_CONFIG
from backend.video_manager import VideoManager
from backend.game_engine import GameEngine, MatchState
from backend.pose_tracker import PlayerPoseData, GazeState
from backend.reaction_detector import ReactionEvent, EventType

class TestGameEngine(unittest.TestCase):
    def setUp(self):
        vm = VideoManager()
        self.engine = GameEngine(config=DEFAULT_CONFIG, video_manager=vm)

    def test_initial_state(self):
        self.assertEqual(self.engine.match_state, MatchState.LOBBY)
        self.assertEqual(len(self.engine.player_stats), 6)
        for p in self.engine.player_stats.values():
            self.assertEqual(p.penalty_score, 0.0)
            self.assertEqual(p.tv_attention_pct, 100.0)

    def test_penalty_scoring_and_ranking(self):
        self.engine.start_match()
        self.engine.transition_to(MatchState.PLAYING_VIDEO)

        # Player 1 faces TV; Player 2 looks away
        poses = {
            1: PlayerPoseData(player_id=1, slot="left", face_detected=True, gaze_state=GazeState.FACING_TV),
            2: PlayerPoseData(player_id=2, slot="right", face_detected=True, gaze_state=GazeState.LOOKING_SIDEWAYS),
            3: PlayerPoseData(player_id=3, slot="left", face_detected=True, gaze_state=GazeState.FACING_TV),
            4: PlayerPoseData(player_id=4, slot="right", face_detected=True, gaze_state=GazeState.FACING_TV),
            5: PlayerPoseData(player_id=5, slot="left", face_detected=True, gaze_state=GazeState.FACING_TV),
            6: PlayerPoseData(player_id=6, slot="right", face_detected=True, gaze_state=GazeState.FACING_TV)
        }

        # Player 2 has a look away event
        events = [
            ReactionEvent(
                event_id="test-ev",
                player_id=2,
                timestamp=1.0,
                game_time_sec=1.0,
                video_round=1,
                event_type=EventType.LOOK_AWAY,
                intensity=0.8,
                description="Looked away test"
            )
        ]

        # Simulate 1 second of play
        self.engine.tick(poses, events)

        # Player 2 should have higher penalty than Player 1
        p1 = self.engine.player_stats[1]
        p2 = self.engine.player_stats[2]
        self.assertGreater(p2.penalty_score, p1.penalty_score)
        self.assertEqual(p1.rank, 1)  # Player 1 is rank 1
        self.assertGreater(p2.rank, p1.rank)  # Player 2 is worse

if __name__ == "__main__":
    unittest.main()
