"""Sample-wise all-way few-shot evaluation for Figure 2 methods."""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


DATASETS = {
    "kaggle18": "CLAP_Kaggle18",
    "voxceleb1": "CLAP_VoxCeleb1",
    "birdclef": "CLAP_BirdClef",
}
METHODS = (
    "zero_shot", "tip_f", "treff", "clap_s_plus", "catclap_mlp",
    "catclap_ln",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--shots", type=int, choices=(1, 5, 16), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--support-manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--support-batch-size", type=int, default=32)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--adapt-batch-size", type=int, default=64)
    parser.add_argument("--class-chunk-size", type=int, default=128)
    parser.add_argument("--validation-limit", type=int, default=1_000_000_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()
    if args.method != "zero_shot" and args.support_manifest is None:
        parser.error("--support-manifest is required except for zero_shot")
    support_path = args.support_manifest.resolve() if args.support_manifest else None


    workdir = Path(__file__).resolve().parent / DATASETS[args.dataset]
    os.chdir(workdir)
    sys.path.insert(0, str(workdir))
    sys.path.insert(0, str(workdir.parent))

    from allway_core import (
        SelectedRowsDataset, adapt_protoclap, build_support_frame_tensors,
        build_support_tensors, can_audio_logits_for_batch, get_embeddings,
        make_logits, read_support_manifest, select_validation_rows,
    )
    from cache_baselines import (
        adapt_tip_adapter_f, adapt_treff_official_full,
        cache_logits, official_clap_logit_scale, search_tip_adapter_hparams,
    )
    from clap_s_baseline import (
        clap_logit_scale, clap_s_plus_components, clap_s_plus_logits,
        clap_s_text_anchors, search_clap_s_hparams, train_clap_s_adapter,
    )
    from manifest_dataset import (
        SupervisedManifestDataset, collate_raw, default_manifest_from_params,
        load_class_names,
    )
    from models.clap import ZeroShotCLAP

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open("proto_params.yaml") as stream:
        params = yaml.safe_load(stream)
    manifest_path = workdir / default_manifest_from_params(params)
    class_names = load_class_names(manifest_path)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    use_tcam = args.method.startswith("catclap_")
    model = ZeroShotCLAP(device=device, register_frame_hook=use_tcam)
    model.eval()

    support_dataset = None
    if support_path is not None:
        support_rows = read_support_manifest(support_path)
        expected = num_classes * args.shots
        if len(support_rows) != expected:
            raise ValueError(f"Expected {expected} support rows, got {len(support_rows)}")
        support_dataset = SelectedRowsDataset(
            support_rows, class_to_id, workdir, 44100, 5.0,
        )
    query_dataset = SupervisedManifestDataset(
        manifest_path, "test", class_to_id, workdir, 44100, 5.0,
    )

    def loader(dataset, batch_size):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_raw,
        )

    query_loader = loader(query_dataset, args.batch_size)
    text_proto = None
    if args.method not in ("clap_s_plus",):
        with torch.no_grad():
            text_proto = F.normalize(model.get_text_anchors(class_names).to(device).float(), dim=-1)

    if args.method != "zero_shot":
        support_feats, support_labels = build_support_tensors(
            model, loader(support_dataset, args.support_batch_size), device,
        )

    if use_tcam:
        _, support_frame_proto, _ = build_support_frame_tensors(
            model, loader(support_dataset, args.frame_batch_size),
            num_classes, args.shots, device,
        )
        arch = "tclap" if args.method == "catclap_mlp" else "protoclip"
        adapter, _, text_proto = adapt_protoclap(
            support_feats=support_feats, support_labels=support_labels,
            text_init=text_proto, num_classes=num_classes, k_shot=args.shots,
            ft_steps=30, lr=1e-3, adapter_arch=arch, lambda_align=0.0,
            learn_audio_memory=False, symmetric_audio_memory=True,
            alpha=0.5, beta=5.0, distance="l2", weight_decay=0.05,
        )
    elif args.method == "tip_f":
        scale = official_clap_logit_scale(model)
    elif args.method == "clap_s_plus":
        scale = clap_logit_scale(model)
    elif args.method == "treff":
        scale = official_clap_logit_scale(model)

    if args.method in ("tip_f", "clap_s_plus"):
        val_rows = select_validation_rows(
            manifest_path, class_names, args.validation_limit, args.seed,
        )
        val_dataset = SelectedRowsDataset(val_rows, class_to_id, workdir, 44100, 5.0)
        val_feats, val_labels = build_support_tensors(
            model, loader(val_dataset, args.batch_size), device,
        )

    if args.method == "tip_f":
        cache, _, _ = adapt_tip_adapter_f(
            support=support_feats, labels=support_labels, text_anchors=text_proto,
            num_classes=num_classes, steps=20, lr=1e-3,
            batch_size=args.adapt_batch_size, alpha=1.17, beta=1.0,
            seed=args.seed, text_logit_scale=scale,
            validation=val_feats, validation_labels=val_labels,
        )
        records = []
        with torch.no_grad():
            for start in range(0, val_feats.size(0), args.batch_size):
                end = start + args.batch_size
                features = val_feats[start:end]
                records.append((
                    scale * (features @ text_proto.T), features @ cache.T,
                    support_labels, val_labels[start:end],
                ))
        alpha, beta, _ = search_tip_adapter_hparams(records, num_classes)
    elif args.method == "clap_s_plus":
        train_text = clap_s_text_anchors(model, class_names, device, stage="train")
        text_proto = clap_s_text_anchors(model, class_names, device, stage="inference")
        adapter = train_clap_s_adapter(
            support=support_feats, labels=support_labels, text_anchors=train_text,
            epochs=20, batch_size=128, seed=args.seed, logit_scale=scale,
            validation_features=val_feats, validation_labels=val_labels,
        )
        records = []
        with torch.no_grad():
            for start in range(0, val_feats.size(0), args.batch_size):
                end = start + args.batch_size
                text_logits, affinity = clap_s_plus_components(
                    val_feats[start:end], support_feats, support_labels,
                    text_proto, adapter, num_classes, scale,
                )
                records.append((text_logits, affinity, support_labels, val_labels[start:end]))
        alpha, beta, _ = search_clap_s_hparams(records, num_classes)
    elif args.method == "treff":
        projection, cache, alpha = adapt_treff_official_full(
            support=support_feats, labels=support_labels, text_anchors=text_proto,
            num_classes=num_classes, steps=20, lr=1e-4,
            batch_size=args.adapt_batch_size, alpha=1.0, beta=5.5,
            seed=args.seed, text_logit_scale=scale,
        )

    correct = total = 0
    for wavs, labels in tqdm(query_loader, desc=args.method):
        labels = labels.to(device)
        query_wavs = [w.detach().cpu() for w in wavs]
        with torch.no_grad():
            raw = get_embeddings(model, query_wavs, device)
            if args.method == "zero_shot":
                preds = (raw @ text_proto.T).argmax(dim=1)
            elif use_tcam:
                query = adapter(raw)
                text_logits = make_logits(query, text_proto, distance="l2", beta=5.0)
                audio_logits = can_audio_logits_for_batch(
                    clap_model=model, query_wavs=query_wavs,
                    support_frame_proto=support_frame_proto, adapter=adapter,
                    device=device, class_chunk_size=args.class_chunk_size,
                    can_temperature=0.025, tcam_score_mode="mean",
                    tcam_attn_mode="residual", beta=5.0, distance="l2",
                )
                probs = 0.5 * F.softmax(audio_logits, dim=1) + 0.5 * F.softmax(text_logits, dim=1)
                preds = probs.argmax(dim=1)
            elif args.method == "tip_f":
                text_logits = scale * (raw @ text_proto.T)
                retrieval = cache_logits(raw, cache, support_labels, num_classes, beta)
                preds = (text_logits + alpha * retrieval).argmax(dim=1)
            elif args.method == "treff":
                query = F.linear(F.normalize(raw, dim=-1), projection)
                text_logits = scale * (raw @ text_proto.T)
                retrieval = cache_logits(query, cache, support_labels, num_classes, 5.5)
                preds = (text_logits + alpha * retrieval).argmax(dim=1)
            else:
                preds = clap_s_plus_logits(
                    raw, support_feats, support_labels, text_proto,
                    adapter, num_classes, scale, alpha, beta,
                ).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    print(f"{args.dataset} {args.method} {args.shots}-shot seed={args.seed}: "
          f"accuracy={100 * correct / total:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
