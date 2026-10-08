import unittest
import tempfile
import shutil
from pathlib import Path
from backend.video_manager import VideoManager

class TestVideoManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.vm = VideoManager(video_dir=self.temp_dir, target_count=3)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_procedural_generation_and_playlist(self):
        # Should generate 3 synthetic challenge clips
        vids = self.vm.refresh_and_ensure_videos()
        self.assertEqual(len(vids), 3)

        # Playlist generation
        playlist = self.vm.start_new_match_playlist()
        self.assertEqual(len(playlist), 3)
        self.assertEqual(self.vm.current_index, 0)
        self.assertIsNotNone(self.vm.get_current_video())

        # Next video
        v2 = self.vm.next_video()
        self.assertEqual(self.vm.current_index, 1)
        self.assertEqual(v2["round_num"], 2)

if __name__ == "__main__":
    unittest.main()
