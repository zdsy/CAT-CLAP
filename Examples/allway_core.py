"""Support manifests, adaptation, and TCAM for paper all-way evaluation."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from manifest_dataset import SupervisedManifestDataset

def read_manifest_rows(manifest_path):
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def write_support_manifest(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class_name", "npy_path", "wav_path"])
        writer.writeheader()
        writer.writerows(rows)


def read_support_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def select_support_rows(manifest_path, class_names, k_shot, seed, support_split="train"):
    rows_by_class = defaultdict(list)
    for row in read_manifest_rows(manifest_path):
        if row["split"] == support_split:
            rows_by_class[row["class_name"]].append(row)

    rng = np.random.default_rng(seed)
    support_rows = []
    for class_name in class_names:
        rows = rows_by_class[class_name]
        if not rows:
            raise RuntimeError(f"No support rows for class {class_name!r} in split {support_split!r}")
        replace = len(rows) < k_shot
        picked = rng.choice(len(rows), size=k_shot, replace=replace)
        for idx in picked:
            support_rows.append(rows[int(idx)])
    return support_rows


def select_validation_rows(manifest_path, class_names, max_samples, seed):
    """Select a deterministic, class-balanced subset of the validation split."""
    rows_by_class = defaultdict(list)
    for row in read_manifest_rows(manifest_path):
        if row["split"] == "val":
            rows_by_class[row["class_name"]].append(row)

    rng = np.random.default_rng(seed)
    class_order = list(class_names)
    rng.shuffle(class_order)
    for name in class_order:
        rng.shuffle(rows_by_class[name])

    selected = []
    offset = 0
    while len(selected) < max_samples:
        added = False
        for name in class_order:
            rows = rows_by_class[name]
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) >= max_samples:
                    break
        if not added:
            break
        offset += 1
    if not selected:
        raise RuntimeError("CLAP-S+ requires a non-empty validation split")
    return selected


class SelectedRowsDataset(SupervisedManifestDataset):
    def __init__(self, rows, class_to_id, example_dir, sample_rate=44100, clip_seconds=5.0):
        self.manifest_path = None
        self.split = "selected"
        self.class_to_id = class_to_id
        self.example_dir = Path(example_dir)
        self.sample_rate = sample_rate
        self.clip_len = int(round(sample_rate * clip_seconds))
        self.rows = rows


def clear_frame_cache(clap_model):
    if hasattr(clap_model, "frame_features_cache"):
        clap_model.frame_features_cache = []


def get_embeddings(clap_model, wavs, device):
    # The frame hook is active for ProtoCLAP-CAN. Clip-level embedding calls also
    # trigger it, so clear the cache around these calls to avoid memory buildup.
    clear_frame_cache(clap_model)
    with torch.no_grad():
        z = clap_model.get_audio_features(wavs)
    clear_frame_cache(clap_model)
    return F.normalize(z.to(device).float(), p=2, dim=-1)


def get_frame_embeddings(clap_model, wavs, device):
    with torch.no_grad():
        z = clap_model.get_audio_frame_features(wavs)
    z = z.to(device).float()
    if z.dim() == 2:
        z = z.unsqueeze(1)
    return F.normalize(z, p=2, dim=-1)


def make_logits(query_feats, proto_feats, distance="l2", beta=5.0):
    query_feats = F.normalize(query_feats, p=2, dim=-1)
    proto_feats = F.normalize(proto_feats, p=2, dim=-1)
    if isinstance(distance, str) and distance.lower() in ["euclidean", "l2"]:
        return -beta * torch.cdist(query_feats, proto_feats, p=2).pow(2)
    return beta * (query_feats @ proto_feats.T)


def make_pairwise_logits(query_pair, proto_pair, distance="l2", beta=5.0):
    query_pair = F.normalize(query_pair, p=2, dim=-1)
    proto_pair = F.normalize(proto_pair, p=2, dim=-1)
    if isinstance(distance, str) and distance.lower() in ["euclidean", "l2"]:
        return -beta * (query_pair - proto_pair).pow(2).sum(dim=-1)
    return beta * (query_pair * proto_pair).sum(dim=-1)


def build_support_tensors(clap_model, support_loader, device):
    feats = []
    labels = []
    for wavs, y in tqdm(support_loader, desc="Support audio embeddings"):
        feats.append(get_embeddings(clap_model, list(wavs), device).detach().cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0).to(device), torch.cat(labels, dim=0).to(device)


def build_class_prototypes(support_feats, support_labels, num_classes):
    """Compute frozen ProtoNet class means without relying on row ordering."""
    class_sum = support_feats.new_zeros(num_classes, support_feats.size(-1))
    class_sum.index_add_(0, support_labels, support_feats)
    counts = torch.bincount(support_labels, minlength=num_classes).to(
        support_feats.dtype
    )
    return F.normalize(class_sum / counts.clamp_min(1).unsqueeze(-1), p=2, dim=-1)


def build_support_frame_tensors(clap_model, support_loader, num_classes, k_shot, device):
    frame_chunks = []
    for wavs, _ in tqdm(support_loader, desc="Support frame embeddings"):
        frame_chunks.append(get_frame_embeddings(clap_model, list(wavs), device).detach().cpu())
        clear_frame_cache(clap_model)

    sample_frames = torch.cat(frame_chunks, dim=0)
    d = sample_frames.shape[-1]
    class_frame_sum = sample_frames.view(num_classes, k_shot, -1, d).sum(dim=1).to(device)
    class_frame_proto = F.normalize(class_frame_sum / float(k_shot), p=2, dim=-1)
    return sample_frames, class_frame_proto, class_frame_sum


def adapt_catclap(
    support_feats,
    support_labels,
    text_init,
    num_classes,
    k_shot,
    use_finetune=True,
    ft_steps=30,
    lr=1e-3,
    adapter_reduction=4,
    adapter_residual_ratio=0.2,
    adapter_arch="mlp",
    train_text_memory=True,
    alpha=0.5,
    beta=5.0,
    distance="l2",
    weight_decay=0.05,
    use_adapter=True,
    eps=1e-8,
):
    """Adapt the CAT-CLAP adapter and text memory on an all-way support set."""
    if adapter_arch not in {"mlp", "ln"}:
        raise ValueError(f"Unknown CAT-CLAP adapter: {adapter_arch}")

    device = support_feats.device
    d = support_feats.shape[-1]
    hidden_dim = max(d // adapter_reduction, 1)
    text_memory = torch.nn.Parameter(text_init.clone())

    if adapter_arch == "ln":
        adapter_down = torch.nn.Linear(d, hidden_dim, bias=False, device=device)
        adapter_norm_down = torch.nn.LayerNorm(hidden_dim, device=device)
        adapter_up = torch.nn.Linear(hidden_dim, d, bias=False, device=device)
        adapter_norm_up = torch.nn.LayerNorm(d, device=device)
        adapter_params = list(adapter_down.parameters())
        adapter_params += list(adapter_norm_down.parameters())
        adapter_params += list(adapter_up.parameters())
        adapter_params += list(adapter_norm_up.parameters())
    else:
        W_down = torch.nn.Parameter(torch.empty(d, hidden_dim, device=device))
        W_up = torch.nn.Parameter(torch.empty(hidden_dim, d, device=device))
        b_down = torch.nn.Parameter(torch.zeros(hidden_dim, device=device))
        b_up = torch.nn.Parameter(torch.zeros(d, device=device))
        torch.nn.init.xavier_uniform_(W_down)
        torch.nn.init.xavier_uniform_(W_up)
        adapter_params = [W_down, W_up, b_down, b_up]

    def adapter(x):
        if not use_adapter or not use_finetune:
            return F.normalize(x, p=2, dim=-1)
        if adapter_arch == "ln":
            z = adapter_norm_down(adapter_down(x))
            z = adapter_norm_up(adapter_up(z))
        else:
            z = F.relu(x @ W_down + b_down, inplace=False)
            z = z @ W_up + b_up
        out = adapter_residual_ratio * z + (1.0 - adapter_residual_ratio) * x
        return F.normalize(out, p=2, dim=-1)

    def support_prototypes():
        features = adapter(support_feats)
        prototypes = features.view(num_classes, k_shot, d).mean(dim=1)
        return F.normalize(prototypes, p=2, dim=-1)

    params = []
    if use_adapter:
        params.extend(adapter_params)
    if train_text_memory:
        params.append(text_memory)

    if use_finetune and params:
        optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, eps=1e-4
        )
        for _ in tqdm(range(ft_steps), desc="CAT-CLAP all-way adaptation"):
            optimizer.zero_grad()
            support_query = adapter(support_feats)
            audio_proto = support_prototypes()
            text_proto = F.normalize(text_memory, p=2, dim=-1)
            audio_logits = make_logits(
                support_query, audio_proto, distance=distance, beta=beta
            )
            text_logits = make_logits(
                support_query, text_proto, distance=distance, beta=beta
            )
            probs = (
                alpha * F.softmax(audio_logits, dim=1)
                + (1.0 - alpha) * F.softmax(text_logits, dim=1)
            )
            loss = F.nll_loss(torch.log(probs.clamp_min(eps)), support_labels)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        audio_proto = support_prototypes()
        text_proto = F.normalize(text_memory, p=2, dim=-1)

    return adapter, audio_proto.detach(), text_proto.detach()


def tcam_audio_logits_from_frames(
    query_frames,
    support_frame_proto,
    adapter,
    class_chunk_size=128,
    tcam_temperature=0.025,
    beta=5.0,
    distance="l2",
    eps=1e-8,
):
    query_frames = F.normalize(query_frames, p=2, dim=-1)
    num_queries = query_frames.size(0)
    num_classes = support_frame_proto.size(0)
    d = support_frame_proto.size(-1)
    logits_out = []

    for start in range(0, num_classes, class_chunk_size):
        support_chunk = support_frame_proto[start:start + class_chunk_size]
        sim = torch.einsum("ntd,qsd->nqts", support_chunk, query_frames)
        support_scores = sim.mean(dim=-1)
        query_scores = sim.mean(dim=-2)
        support_attn = F.softmax(
            support_scores / tcam_temperature, dim=-1
        ) + 1.0
        query_attn = F.softmax(
            query_scores / tcam_temperature, dim=-1
        ) + 1.0

        proto_pair = torch.einsum("ntd,nqt->qnd", support_chunk, support_attn)
        proto_pair = proto_pair / support_attn.sum(dim=-1).transpose(0, 1).unsqueeze(-1).clamp(min=eps)

        query_pair = torch.einsum("qtd,nqt->qnd", query_frames, query_attn)
        query_pair = query_pair / query_attn.permute(1, 0, 2).sum(dim=-1, keepdim=True).clamp(min=eps)

        query_pair = adapter(query_pair.reshape(-1, d)).view(num_queries, -1, d)
        proto_pair = adapter(proto_pair.reshape(-1, d)).view(num_queries, -1, d)
        logits_out.append(make_pairwise_logits(query_pair, proto_pair, distance=distance, beta=beta))

    return torch.cat(logits_out, dim=1)


def tcam_audio_logits_for_batch(
    clap_model,
    query_wavs,
    support_frame_proto,
    adapter,
    device,
    class_chunk_size=128,
    tcam_temperature=0.025,
    beta=5.0,
    distance="l2",
    eps=1e-8,
):
    query_frames = get_frame_embeddings(clap_model, query_wavs, device)
    return tcam_audio_logits_from_frames(
        query_frames=query_frames,
        support_frame_proto=support_frame_proto,
        adapter=adapter,
        class_chunk_size=class_chunk_size,
        tcam_temperature=tcam_temperature,
        beta=beta,
        distance=distance,
        eps=eps,
    )
