# Data layout

Audio, extracted MetaAudio features, and pretrained model weights are not
distributed here. Obtain the datasets from their original providers and
prepare the 5-second/44.1-kHz audio and MetaAudio features before running the
experiments. The loaders expect this layout relative to the repository root:

```text
Datasets/
  BirdClef/
    audio/
    features/
  FSDKaggle2018/
    Sorted/
    Spec_5_second_npy/
  VoxCeleb1_Mirror/
  VoxCeleb1_features/
```

The exact feature and audio paths can be changed in each
`Examples/CLAP_*/proto_params.yaml`. Split metadata and fixed all-way support
manifests are included in `Examples/CLAP_*/dataset_/splits/` and
`Examples/CLAP_*/support_manifests/`; they do not contain audio.
