# CAT-CLAP

Code for few-shot audio classification with frozen CLAP encoders. This repository
contains CAT-CLAP with MLP and LN adapters, plus the CLAP-based baselines used
in the paper: zero-shot CLAP, frozen-audio ProtoNet, TIP-Adapter-F, Treff
Adapter, and CLAP-S+.

## Setup

Create a Python environment and install `requirements.txt`. Install a PyTorch
and TorchAudio build compatible with your CPU or CUDA installation. The code
uses the `msclap` 2023 checkpoint; model weights are downloaded by `msclap`
and are not included in this repository.

```bash
pip install -r requirements.txt
```

Place the MetaAudio data under `Datasets/` using the paths listed in
`Datasets/README.md`. Audio and precomputed MetaAudio features are required for
episodic evaluation; audio is required for all-way evaluation. The class
splits and fixed all-way support manifests are included under `Examples/`.

## Episodic evaluation

Run from the repository root. This example evaluates CAT-CLAP-MLP on 10,000
5-way-5-shot BirdClef episodes:

```bash
python Examples/run_episodic.py --dataset birdclef --method catclap_mlp --shots 5 --seed 389
```

Use `--shots 1` for 5-way-1-shot. Available datasets are `kaggle18`,
`voxceleb1`, and `birdclef`. Available methods are `zero_shot`,
`audio_protonet`, `tip_f`, `treff`, `clap_s_plus`, `catclap_mlp`, and
`catclap_ln`. For a quick pipeline check, set `--episodes 2`; this is **not**
a paper-quality estimate. TIP-Adapter-F and CLAP-S+ use validation episodes
from the MetaAudio class split for their hyperparameter selection.

## All-way evaluation

The same test split is evaluated against the full class vocabulary. Choose a
shot count and one of the five fixed support manifests for that dataset. All
methods should use the same manifest when compared.

```bash
python Examples/run_allway.py --dataset birdclef --method catclap_mlp \
  --shots 5 --seed 1337 \
  --support-manifest Examples/CLAP_BirdClef/support_manifests/five_seeds/seed0/BirdClef_5shot_support.csv
```

The all-way methods are `zero_shot`, `tip_f`, `treff`, `clap_s_plus`,
`catclap_mlp`, and `catclap_ln`. Zero-shot needs no support manifest:

```bash
python Examples/run_allway.py --dataset birdclef --method zero_shot --shots 5 --seed 1337
```

The supplied all-way support sets are in `seed0` through `seed4`. Their
sampling seeds are recorded in each dataset's `metadata.json`. The `--seed`
argument controls the method RNG, not support-set selection. The frozen-CLAP
classifier reference is implemented separately in each dataset's
`finetune_clap_supervised.py`.
