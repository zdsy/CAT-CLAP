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


def alignment_loss(audio_proto, text_proto, temperature=1.0):
    audio_proto = F.normalize(audio_proto, p=2, dim=-1)
    text_proto = F.normalize(text_proto, p=2, dim=-1)
    labels = torch.arange(audio_proto.size(0), device=audio_proto.device)
    logits_a2t = audio_proto @ text_proto.T / temperature
    logits_t2a = text_proto @ audio_proto.T / temperature
    return F.cross_entropy(logits_a2t, labels) + F.cross_entropy(logits_t2a, labels)


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


def adapt_protoclap(
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
    adapter_arch="tclap",
    use_leave_one_out=False,
    train_text_memory=True,
    lambda_cls=1.0,
    lambda_align=0.5,
    align_temperature=1.0,
    alpha=0.5,
    beta=5.0,
    distance="l2",
    weight_decay=0.05,
    use_adapter=True,
    learn_audio_memory=True,
    symmetric_audio_memory=True,
    alignment_source="audio_memory",
    tcam_alignment_proto=None,
    eps=1e-8,
):
    device = support_feats.device
    d = support_feats.shape[-1]
    hidden_dim = max(d // adapter_reduction, 1)

    if learn_audio_memory:
        audio_memory = torch.nn.Parameter(support_feats.clone())
    else:
        audio_memory = support_feats.detach()
    text_memory = torch.nn.Parameter(text_init.clone())
    if adapter_arch == "protoclip":
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
        if adapter_arch == "protoclip":
            z = adapter_norm_down(adapter_down(x))
            z = adapter_norm_up(adapter_up(z))
        else:
            z = F.relu(x @ W_down + b_down, inplace=False)
            z = z @ W_up + b_up
        out = adapter_residual_ratio * z + (1.0 - adapter_residual_ratio) * x
        return F.normalize(out, p=2, dim=-1)

    def audio_proto_from_memory():
        if learn_audio_memory:
            proto_source = (
                adapter(audio_memory)
                if symmetric_audio_memory
                else F.normalize(audio_memory, p=2, dim=-1)
            )
        else:
            proto_source = (
                adapter(support_feats)
                if symmetric_audio_memory
                else F.normalize(support_feats, p=2, dim=-1)
            )
        audio_proto = proto_source.view(num_classes, k_shot, d).mean(dim=1)
        return F.normalize(audio_proto, p=2, dim=-1)

    def alignment_audio_proto(audio_proto):
        if alignment_source == "audio_memory":
            return audio_proto
        if alignment_source == "global":
            global_proto = adapter(support_feats).view(num_classes, k_shot, d).mean(dim=1)
            return F.normalize(global_proto, p=2, dim=-1)
        if alignment_source == "tcam":
            if tcam_alignment_proto is None:
                raise ValueError("TCAM alignment requires precomputed TCAM prototypes")
            return adapter(tcam_alignment_proto)
        raise ValueError(f"Unknown alignment source: {alignment_source}")

    support_index_grid = torch.arange(
        num_classes * k_shot, device=device
    ).view(num_classes, k_shot)

    if use_finetune:
        params = []
        if learn_audio_memory:
            params.append(audio_memory)
        if use_adapter:
            params.extend(adapter_params)
        if train_text_memory:
            params.append(text_memory)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, eps=1e-4)

        for ft_step in tqdm(range(ft_steps), desc="T-CLAP all-way adaptation"):
            optimizer.zero_grad()
            text_proto = F.normalize(text_memory, p=2, dim=-1)

            if use_leave_one_out and k_shot > 1:
                heldout_position = ft_step % k_shot
                class_ids = torch.arange(num_classes, device=device)
                pseudo_query_idx = support_index_grid[:, heldout_position]
                pseudo_support_mask = torch.ones(
                    num_classes, k_shot, dtype=torch.bool, device=device
                )
                pseudo_support_mask[:, heldout_position] = False
                pseudo_support_idx = support_index_grid[pseudo_support_mask].view(
                    num_classes, k_shot - 1
                ).reshape(-1)

                support_query = adapter(support_feats[pseudo_query_idx])
                memory_source = (
                    audio_memory[pseudo_support_idx]
                    if learn_audio_memory
                    else support_feats[pseudo_support_idx]
                )
                proto_source = (
                    adapter(memory_source)
                    if symmetric_audio_memory
                    else F.normalize(memory_source, p=2, dim=-1)
                )
                audio_proto = proto_source.view(
                    num_classes, k_shot - 1, d
                ).mean(dim=1)
                audio_proto = F.normalize(audio_proto, p=2, dim=-1)
                cls_labels = class_ids
            else:
                support_query = adapter(support_feats)
                audio_proto = audio_proto_from_memory()
                cls_labels = support_labels

            audio_logits = make_logits(support_query, audio_proto, distance=distance, beta=beta)
            text_logits = make_logits(support_query, text_proto, distance=distance, beta=beta)
            probs = alpha * F.softmax(audio_logits, dim=1) + (1.0 - alpha) * F.softmax(text_logits, dim=1)
            cls_loss = F.nll_loss(torch.log(probs.clamp_min(eps)), cls_labels)
            loss = lambda_cls * cls_loss
            if lambda_align > 0.0:
                align_audio = alignment_audio_proto(audio_proto)
                loss = loss + lambda_align * alignment_loss(
                    align_audio, text_proto, align_temperature
                )
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        audio_proto = audio_proto_from_memory()
        text_proto = F.normalize(text_memory, p=2, dim=-1)

    return adapter, audio_proto.detach(), text_proto.detach()


def can_audio_logits_from_frames(
    query_frames,
    support_frame_proto,
    adapter,
    class_chunk_size=128,
    can_temperature=0.025,
    tcam_score_mode="mean",
    tcam_top_m=8,
    tcam_attn_mode="residual",
    tcam_mix=0.5,
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
        if tcam_score_mode == "mean":
            support_scores = sim.mean(dim=-1)
            query_scores = sim.mean(dim=-2)
        elif tcam_score_mode == "topm":
            support_m = min(max(int(tcam_top_m), 1), sim.size(-1))
            query_m = min(max(int(tcam_top_m), 1), sim.size(-2))
            support_scores = sim.topk(support_m, dim=-1).values.mean(dim=-1)
            query_scores = sim.topk(query_m, dim=-2).values.mean(dim=-2)
        else:
            raise ValueError(f"Unknown TCAM score mode: {tcam_score_mode}")

        if tcam_attn_mode == "residual":
            support_attn = F.softmax(support_scores / can_temperature, dim=-1) + 1.0
            query_attn = F.softmax(query_scores / can_temperature, dim=-1) + 1.0
        elif tcam_attn_mode == "sigmoid_residual":
            support_threshold = support_scores.mean(dim=-1, keepdim=True)
            query_threshold = query_scores.mean(dim=-1, keepdim=True)
            support_attn = 1.0 + torch.sigmoid(
                (support_scores - support_threshold) / can_temperature
            )
            query_attn = 1.0 + torch.sigmoid(
                (query_scores - query_threshold) / can_temperature
            )
        elif tcam_attn_mode == "softmax_mix":
            support_soft = F.softmax(support_scores / can_temperature, dim=-1)
            query_soft = F.softmax(query_scores / can_temperature, dim=-1)
            support_uniform = torch.full_like(support_soft, 1.0 / support_soft.size(-1))
            query_uniform = torch.full_like(query_soft, 1.0 / query_soft.size(-1))
            support_attn = (1.0 - tcam_mix) * support_uniform + tcam_mix * support_soft
            query_attn = (1.0 - tcam_mix) * query_uniform + tcam_mix * query_soft
        elif tcam_attn_mode == "sigmoid":
            support_threshold = support_scores.mean(dim=-1, keepdim=True)
            query_threshold = query_scores.mean(dim=-1, keepdim=True)
            support_attn = torch.sigmoid((support_scores - support_threshold) / can_temperature)
            query_attn = torch.sigmoid((query_scores - query_threshold) / can_temperature)
        else:
            raise ValueError(f"Unknown TCAM attention mode: {tcam_attn_mode}")

        proto_pair = torch.einsum("ntd,nqt->qnd", support_chunk, support_attn)
        proto_pair = proto_pair / support_attn.sum(dim=-1).transpose(0, 1).unsqueeze(-1).clamp(min=eps)

        query_pair = torch.einsum("qtd,nqt->qnd", query_frames, query_attn)
        query_pair = query_pair / query_attn.permute(1, 0, 2).sum(dim=-1, keepdim=True).clamp(min=eps)

        query_pair = adapter(query_pair.reshape(-1, d)).view(num_queries, -1, d)
        proto_pair = adapter(proto_pair.reshape(-1, d)).view(num_queries, -1, d)
        logits_out.append(make_pairwise_logits(query_pair, proto_pair, distance=distance, beta=beta))

    return torch.cat(logits_out, dim=1)


def can_audio_logits_for_batch(
    clap_model,
    query_wavs,
    support_frame_proto,
    adapter,
    device,
    class_chunk_size=128,
    can_temperature=0.025,
    tcam_score_mode="mean",
    tcam_top_m=8,
    tcam_attn_mode="residual",
    tcam_mix=0.5,
    beta=5.0,
    distance="l2",
    eps=1e-8,
):
    query_frames = get_frame_embeddings(clap_model, query_wavs, device)
    return can_audio_logits_from_frames(
        query_frames=query_frames,
        support_frame_proto=support_frame_proto,
        adapter=adapter,
        class_chunk_size=class_chunk_size,
        can_temperature=can_temperature,
        tcam_score_mode=tcam_score_mode,
        tcam_top_m=tcam_top_m,
        tcam_attn_mode=tcam_attn_mode,
        tcam_mix=tcam_mix,
        beta=beta,
        distance=distance,
        eps=eps,
    )
