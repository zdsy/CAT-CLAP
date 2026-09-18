"""Sample-wise all-way manifest loader; audio files are supplied separately."""

import csv
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

class SupervisedManifestDataset(Dataset):
    def __init__(self, manifest_path, split, class_to_id, example_dir, sample_rate=44100, clip_seconds=5.0):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.class_to_id = class_to_id
        self.example_dir = Path(example_dir)
        self.sample_rate = sample_rate
        self.clip_len = int(round(sample_rate * clip_seconds))
        self.rows = []

        with open(self.manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.rows.append(row)

        if not self.rows:
            raise RuntimeError(f"No rows for split={split!r} in {manifest_path}")

    def __len__(self):
        return len(self.rows)

    def _load_audio(self, wav_path):
        try:
            waveform, sr = torchaudio.load(wav_path)
        except Exception as exc:
            print(f"[Warning] Could not load audio {wav_path}: {exc}. Using silence.")
            return torch.zeros(self.clip_len, dtype=torch.float32)

        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0).float()

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform.unsqueeze(0)).squeeze(0)

        if waveform.numel() <= 0:
            return torch.zeros(self.clip_len, dtype=torch.float32)

        if waveform.numel() < self.clip_len:
            repeat = int(np.ceil(self.clip_len / waveform.numel()))
            waveform = waveform.repeat(repeat)[:self.clip_len]
        elif waveform.numel() > self.clip_len:
            max_start = waveform.numel() - self.clip_len
            if self.split == "train":
                start = int(torch.randint(0, max_start + 1, (1,)).item())
            else:
                start = max_start // 2
            waveform = waveform[start:start + self.clip_len]

        return waveform.contiguous()

    def __getitem__(self, index):
        row = self.rows[index]
        wav_path = self.example_dir / row["wav_path"]
        wav = self._load_audio(wav_path)
        label = self.class_to_id[row["class_name"]]
        return wav, label


def collate_raw(batch):
    wavs = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return wavs, labels


def load_class_names(manifest_path):
    names = set()
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.add(row["class_name"])
    return sorted(names)


def default_manifest_from_params(params):
    if "supervised_split_path" in params.get("data", {}):
        return params["data"]["supervised_split_path"]
    safe_name = params["data"]["name"].replace(" ", "_").replace("/", "_")
    return f"dataset_/splits/{safe_name}_supervised_sample_split.csv"
