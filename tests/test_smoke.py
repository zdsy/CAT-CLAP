"""Fast CPU checks for the released CAT-CLAP method paths."""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1] / "Examples"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CLAP_BirdClef"))

from allway_core import adapt_catclap, tcam_audio_logits_from_frames
from proto_steps import eval_step_catclap


class DummyCLAP:
    def eval(self):
        return self

    def get_audio_features(self, wavs):
        return torch.stack([F.normalize(w.float(), dim=-1) for w in wavs])

    def get_audio_frame_features(self, wavs):
        return torch.stack([torch.stack((w.float(), w.float().roll(1))) for w in wavs])

    def get_text_anchors(self, class_names):
        return F.normalize(torch.eye(4)[:len(class_names)], dim=-1)


class MethodSmokeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def test_episodic_both_adapters(self):
        wavs = [[
            torch.tensor([1.0, 0.1, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.1, 0.0]),
            torch.tensor([0.9, 0.2, 0.0, 0.0]),
            torch.tensor([0.1, 0.9, 0.0, 0.0]),
        ]]
        for arch in ("mlp", "ln"):
            with self.subTest(arch=arch):
                result = eval_step_catclap(
                    clap_model=DummyCLAP(), batch_wavs=wavs, q_num=[1, 1],
                    y=torch.tensor([[0, 1, 0, 1]]),
                    batch_class_names=[["class_a", "class_b"]],
                    device=torch.device("cpu"), n_way=2, k_shot=1, q_queries=1,
                    distance="l2", ft_steps=2,
                    adapter_arch=arch,
                )
                self.assertEqual(len(result[3]), 1)
                self.assertTrue(0 <= result[2] <= 1)

    def test_allway_adaptation_and_tcam(self):
        support = F.normalize(torch.rand(4, 4), dim=-1)
        text = F.normalize(torch.rand(2, 4), dim=-1)
        labels = torch.tensor([0, 0, 1, 1])
        for arch in ("mlp", "ln"):
            with self.subTest(arch=arch):
                adapter, _, learned_text = adapt_catclap(
                    support_feats=support, support_labels=labels,
                    text_init=text, num_classes=2, k_shot=2,
                    ft_steps=2, adapter_arch=arch,
                )
                frames = F.normalize(torch.rand(2, 3, 4), dim=-1)
                logits = tcam_audio_logits_from_frames(
                    query_frames=frames, support_frame_proto=frames,
                    adapter=adapter, class_chunk_size=1,
                )
                self.assertEqual(tuple(logits.shape), (2, 2))
                self.assertEqual(tuple(learned_text.shape), (2, 4))
                self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
