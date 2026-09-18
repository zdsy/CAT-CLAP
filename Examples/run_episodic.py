"""MetaAudio episodic evaluation for the CLAP-based methods in the paper."""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import torch
import yaml
from tqdm import tqdm


DATASETS = {
    "kaggle18": "CLAP_Kaggle18",
    "voxceleb1": "CLAP_VoxCeleb1",
    "birdclef": "CLAP_BirdClef",
}
METHODS = (
    "zero_shot", "audio_protonet", "tip_f", "treff", "clap_s_plus",
    "catclap_mlp", "catclap_ln",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--shots", type=int, choices=(1, 5), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent / DATASETS[args.dataset]
    os.chdir(workdir)
    sys.path.insert(0, str(workdir))
    sys.path.insert(0, str(workdir.parent))

    from all_prep_batches import prep_var_eval
    from dataset_.DatasetClasses import FastDataLoader, NormDataset
    from dataset_.SetupClass import DatasetSetup
    from models.clap import ZeroShotCLAP
    from proto_steps import (
        eval_step_clap_audio_proto,
        eval_step_tmclap_no_audio,
        eval_step_zero_shot_var,
    )
    from task_sampling_classes import NShotTaskSampler
    from cache_baselines import (
        eval_step_tip_adapter_f_var,
        eval_step_treff_official_var,
        tune_tip_adapter_f_episodic,
    )
    from clap_s_baseline import (
        eval_step_clap_s_plus_var,
        tune_clap_s_plus_episodic,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open("proto_params.yaml") as stream:
        params = yaml.safe_load(stream)
    params["base"].update(n_way=5, k_shot=args.shots, q_queries=1)
    params["training"]["test_tasks"] = args.episodes

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model = ZeroShotCLAP(device=device)

    class_splits = np.load(params["data"]["fixed_path"], allow_pickle=True)
    setup = DatasetSetup(
        params=params,
        splits=[params["split"][key] for key in ("train", "val", "test")],
        seed=args.seed,
        class_splits=class_splits,
    )

    def make_loader(classes, episodes):
        dataset = NormDataset(
            data_path=params["data"]["data_path"],
            wav_path=params["data"]["wav_path"],
            classes=classes,
            norm=params["data"]["norm"],
            stats_file_path=setup.stats_file_path,
        )
        sampler = NShotTaskSampler(
            dataset=dataset,
            episodes_per_epoch=episodes,
            n_way=5,
            k_shot=args.shots,
            q_queries=1,
            num_tasks=1,
            seed=args.seed,
        )
        return FastDataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=params["training"]["num_workers"],
            collate_fn=lambda batch: (
                [item[0] for item in batch],
                [item[1] for item in batch],
                torch.LongTensor([item[2] for item in batch]),
                [item[3] for item in batch],
            ),
        )

    prep = prep_var_eval(
        n_way=5, k_shot=args.shots, q_queries=1, device=device,
        trans=params["training"]["trans_batch"],
    )
    val_loader = make_loader(setup.val, params["training"]["val_tasks"])
    test_loader = make_loader(setup.test, args.episodes)

    if args.method == "zero_shot":
        step = eval_step_zero_shot_var
    elif args.method == "audio_protonet":
        step = eval_step_clap_audio_proto
    elif args.method == "tip_f":
        epoch, alpha, beta, _ = tune_tip_adapter_f_episodic(
            model, val_loader, prep, device, 5, args.shots, 1,
            epochs=20, max_tasks=200,
        )
        step = lambda **kw: eval_step_tip_adapter_f_var(
            **kw, ft_steps=epoch, lr=1e-3, tip_alpha=alpha, tip_beta=beta,
        )
    elif args.method == "treff":
        step = lambda **kw: eval_step_treff_official_var(
            **kw, ft_steps=20, lr=1e-4,
        )
    elif args.method == "clap_s_plus":
        alpha, beta, _ = tune_clap_s_plus_episodic(
            model, val_loader, prep, device, 5, args.shots, 1,
            max_tasks=200,
        )
        step = lambda **kw: eval_step_clap_s_plus_var(
            **kw, alpha=alpha, beta=beta,
        )
    else:
        arch = "tclap" if args.method == "catclap_mlp" else "protoclip"
        step = lambda **kw: eval_step_tmclap_no_audio(
            **kw, lambda_align=0.0, adapter_arch=arch,
        )

    episode_acc = []
    for batch in tqdm(test_loader, total=args.episodes, desc=args.method):
        _, _, _, q_num, y, batch_wavs, class_names = prep(batch, 1)
        _, _, _, values = step(
            clap_model=model,
            batch_wavs=batch_wavs,
            q_num=q_num,
            y=y,
            batch_class_names=class_names,
            device=device,
            n_way=5,
            k_shot=args.shots,
            q_queries=1,
            distance=params["base"]["distance"],
        )
        episode_acc.extend(values)

    mean = float(np.mean(episode_acc))
    ci = float(scipy.stats.sem(episode_acc) * scipy.stats.t.ppf(0.975, len(episode_acc) - 1))
    print(f"accuracy={mean * 100:.2f} +/- {ci * 100:.2f} (95% CI; {len(episode_acc)} episodes)")


if __name__ == "__main__":
    main()
