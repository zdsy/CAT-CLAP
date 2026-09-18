"""
Paper-faithful supervised downstream training for Microsoft CLAP.

This implements the CLAP paper's downstream supervised setups on an explicit
sample-wise train/val/test manifest:

1. Freeze_L1 / Freeze_L3:
   freeze the CLAP audio encoder and train only an attached 1-layer or 3-layer
   fully-connected classifier. Paper setting: Adam, lr=1e-3, 30 epochs.

2. FineTune_L1 / FineTune_L3:
   unfreeze the CLAP audio encoder and train it together with the attached
   classifier. Paper setting: Adam, lr=1e-4, 30 epochs.

Model selection is by validation accuracy; the final printed number is test
accuracy from the best validation checkpoint.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.clap import ZeroShotCLAP
from utils_proto import set_seed


class FCClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, num_layers=1, hidden_dim=None, dropout=0.2):
        super().__init__()
        if num_layers == 1:
            self.net = nn.Linear(in_dim, num_classes)
        elif num_layers == 3:
            hidden_dim = hidden_dim or in_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError("num_layers must be 1 or 3 for CLAP Freeze_L1/L3 and FineTune_L1/L3")

    def forward(self, x):
        return self.net(x)


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


def set_paper_trainability(clap_core, mode):
    for param in clap_core.parameters():
        param.requires_grad = False

    if mode == "finetune":
        for param in clap_core.audio_encoder.parameters():
            param.requires_grad = True
    elif mode != "freeze":
        raise ValueError("mode must be 'freeze' or 'finetune'")


def run_epoch(clap_core, classifier, loader, optimizer, device, mode, phase, epoch=None, epochs=None):
    is_train = phase == "train"
    clap_core.train(is_train and mode == "finetune")
    classifier.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total = 0
    desc = f"CLAP {mode} {phase}"
    if epoch is not None:
        desc += f" epoch {epoch}/{epochs}"

    pbar = tqdm(loader, desc=desc)
    for wavs, labels in pbar:
        wavs = wavs.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        grad_audio = is_train and mode == "finetune"
        with torch.set_grad_enabled(grad_audio):
            audio_embed, _ = clap_core.audio_encoder(wavs)
            audio_embed = F.normalize(audio_embed, p=2, dim=-1)

        if not grad_audio:
            audio_embed = audio_embed.detach()

        with torch.set_grad_enabled(is_train):
            logits = classifier(audio_embed)
            loss = F.cross_entropy(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.numel()
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
        pbar.set_postfix(
            loss=f"{total_loss / max(total, 1):.4f}",
            acc=f"{total_correct / max(total, 1):.4f}",
        )

    return total_loss / max(total, 1), total_correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="proto_params.yaml")
    parser.add_argument("--sample-split", default=None)
    parser.add_argument("--mode", choices=["freeze", "finetune"], default="finetune")
    parser.add_argument("--classifier-layers", type=int, choices=[1, 3], default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint to load before continuing training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cuda", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 1e-3 if args.mode == "freeze" else 1e-4

    set_seed(args.seed)

    example_dir = Path.cwd()
    with open(args.params) as stream:
        params = yaml.safe_load(stream)

    if args.sample_split is None:
        args.sample_split = default_manifest_from_params(params)
    manifest_path = example_dir / args.sample_split

    cuda_idx = params["base"]["cuda"] if args.cuda is None else args.cuda
    device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")

    class_names = load_class_names(manifest_path)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    num_workers = params["training"]["num_workers"] if args.num_workers is None else args.num_workers
    eval_batch_size = args.batch_size if args.eval_batch_size is None else args.eval_batch_size

    datasets = {
        split: SupervisedManifestDataset(
            manifest_path=manifest_path,
            split=split,
            class_to_id=class_to_id,
            example_dir=example_dir,
            sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_raw, drop_last=False),
        "val": DataLoader(datasets["val"], batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_raw, drop_last=False),
        "test": DataLoader(datasets["test"], batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_raw, drop_last=False),
    }

    clap_model = ZeroShotCLAP(device=device, register_frame_hook=False)
    clap_core = clap_model.model.clap.to(device)
    set_paper_trainability(clap_core, args.mode)

    wav0, _ = next(iter(loaders["train"]))
    with torch.no_grad():
        emb0, _ = clap_core.audio_encoder(wav0[:1].to(device))

    classifier = FCClassifier(
        in_dim=emb0.shape[-1],
        num_classes=len(class_names),
        num_layers=args.classifier_layers,
        dropout=args.dropout,
    ).to(device)

    trainable_params = [p for p in clap_core.parameters() if p.requires_grad]
    trainable_params.extend(classifier.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val_acc = 0.0
    best_epoch = 0
    if args.resume is not None:
        resume_path = Path(args.resume)
        resume_ckpt = torch.load(resume_path, map_location=device)
        clap_core.load_state_dict(resume_ckpt["model"], strict=False)
        classifier.load_state_dict(resume_ckpt["classifier"], strict=True)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        else:
            print("Resume checkpoint has no optimizer state; continuing with a fresh Adam optimizer.")
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_val_acc = float(resume_ckpt.get("best_val_acc", resume_ckpt.get("val_acc", 0.0)))
        best_epoch = int(resume_ckpt.get("best_epoch", resume_ckpt.get("epoch", 0)))
        print(f"Resumed from {resume_path} at epoch {start_epoch}; previous best val acc={best_val_acc:.4f} at epoch {best_epoch}")

    if args.output is None:
        setup_name = f"{args.mode}_L{args.classifier_layers}"
        dataset_name = params["data"]["name"].replace(" ", "_").replace("/", "_")
        args.output = f"checkpoints/{dataset_name}_msclap_{setup_name}.pth"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = out_path.with_name(out_path.stem + "_best" + out_path.suffix)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = run_epoch(clap_core, classifier, loaders["train"], optimizer, device, args.mode, "train", epoch, args.epochs)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(clap_core, classifier, loaders["val"], optimizer, device, args.mode, "val")

        is_best = val_acc >= best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch

        ckpt = {
            "model": clap_core.state_dict(),
            "classifier": classifier.state_dict(),
            "mode": args.mode,
            "classifier_layers": args.classifier_layers,
            "class_names": class_names,
            "sample_split": str(args.sample_split),
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "args": vars(args),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, out_path)
        print(f"Epoch {epoch}: train_acc={train_acc:.4f}; val_acc={val_acc:.4f}; best_val_acc={best_val_acc:.4f} at epoch {best_epoch}")

        if is_best:
            torch.save(ckpt, best_path)

    print(f"Saved final checkpoint: {out_path}")
    print(f"Saved best checkpoint: {best_path}")
    print(f"Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    best_ckpt = torch.load(best_path, map_location=device)
    clap_core.load_state_dict(best_ckpt["model"], strict=False)
    classifier.load_state_dict(best_ckpt["classifier"], strict=True)
    with torch.no_grad():
        test_loss, test_acc = run_epoch(clap_core, classifier, loaders["test"], optimizer, device, args.mode, "test")
    print(f"Best-checkpoint test accuracy: {test_acc:.4f} (loss={test_loss:.4f})")


if __name__ == "__main__":
    main()
