"""Offline tests for the low-VRAM phase split; no model servers or GPU required."""
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "omnidexgrasp"))
from recons import client, sequential
from recons.data import GSAMResult, HaMeRResult, TaskInput
from utils.camera import CameraIntrinsics


class PhaseSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        Image.new("RGB", (8, 8)).save(root / "generated_human_grasp.png")
        self.task = TaskInput("sample", root, root / "scene_image.png", root / "generated_human_grasp.png",
                              CameraIntrinsics(10, 10, 4, 4, 8, 8), "cup", root / "base.obj", root / "depth.png")
        self.cfg = OmegaConf.create({"phase": "gsam", "output": str(root / "out"),
            "servers": {"gsam": "unused", "hamer": "unused", "timeout": 1},
            "scale": {"depth_scale": .001, "max_depth_m": 3, "edge_erode_px": 3,
                      "stat_nb_neighbors": 3, "stat_std_ratio": 2}})
        self.gsam = GSAMResult("success", "ok", [{"is_hand": False, "score": .9, "bbox": [0, 0, 8, 8],
                                                  "mask_rle": {"size": [8, 8], "counts": "x"}}], [8, 8])

    def test_gsam_phase_caches_and_hamer_phase_reuses(self):
        with patch.object(client, "encode_image_file_b64", return_value=""), \
             patch.object(client, "call_gsam", return_value=self.gsam) as gsam, \
             patch.object(client, "call_hamer") as hamer:
            output = client.process_task(self.task, self.cfg)
        self.assertEqual(gsam.call_count, 2)
        hamer.assert_not_called()
        self.assertIsNone(output.hamer)
        self.assertTrue((Path(self.cfg.output) / "sample/data/recons/gsam_cache.json").is_file())

        self.cfg.phase = "hamer"
        with patch.object(client, "encode_image_file_b64", return_value=""), \
             patch.object(client, "call_gsam") as gsam, \
             patch.object(client, "call_hamer", return_value=HaMeRResult("success", "ok")) as hamer, \
             patch.object(client, "decode_mask_rle", return_value=np.ones((8, 8))), \
             patch.object(client, "depth_to_pointcloud", return_value=np.zeros((3, 3))), \
             patch.object(client, "denoise_pointcloud", return_value=np.zeros((3, 3))), \
             patch.object(client, "compute_obj_scale", return_value=(1., 1., 1.)), \
             patch.object(client, "scale_and_center_mesh"):
            output = client.process_task(self.task, self.cfg)
        gsam.assert_not_called()
        hamer.assert_called_once()
        self.assertEqual(output.gsam_grasp.status, "success")


class ServerLifecycleTests(unittest.TestCase):
    def test_stop_terminates_then_kills_on_timeout(self):
        proc = Mock(pid=1)
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired("server", 20), 0]
        with patch.object(sequential.os, "killpg") as kill:
            sequential.stop_server(proc)
        self.assertEqual([c.args[1] for c in kill.call_args_list], [signal.SIGTERM, signal.SIGKILL])

    def test_dead_server_fails_readiness(self):
        proc = Mock(returncode=1)
        proc.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "exited"):
            sequential.wait_ready(proc, "http://127.0.0.1:1", timeout=1)


if __name__ == "__main__":
    unittest.main()
