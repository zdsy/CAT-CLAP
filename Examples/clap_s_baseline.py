"""Faithful MetaAudio translation of the official CLAP-S baselines."""

import math

import torch
import torch.nn.functional as F


class ClapSAdapter(torch.nn.Module):
    """Official two-layer, non-residual CLAP-S+ audio adapter."""

    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden, bias=False),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(hidden, dim, bias=False),
            torch.nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.fc(x)


def clap_s_text_anchors(clap_model, class_names, device, stage="inference"):
    """Use the single prompts in the released CLAP-S training/inference code."""
    labels = []
    for class_name in class_names:
        if hasattr(clap_model, "_label_from_class_name"):
            label = clap_model._label_from_class_name(class_name)
        else:
            label = str(class_name).replace("_", " ").replace("-", " ").strip()
        if stage == "train":
            labels.append(f"this is the sound of {label}")
        elif stage == "inference":
            labels.append(f"this is an audio of {label}")
        else:
            raise ValueError(f"Unknown CLAP-S prompt stage: {stage}")
    with torch.no_grad():
        anchors = clap_model.model.get_text_embeddings(labels)
    anchors = torch.as_tensor(anchors, device=device).float()
    return F.normalize(anchors, p=2, dim=-1)


def clap_logit_scale(clap_model):
    wrapper = getattr(clap_model, "model", clap_model)
    clap = getattr(wrapper, "clap", None)
    scale = getattr(clap, "logit_scale", None)
    if scale is None:
        raise RuntimeError("Could not locate MS-CLAP logit_scale")
    return float(scale.detach().exp().cpu())


def train_clap_s_adapter(
    support,
    labels,
    text_anchors,
    epochs=20,
    batch_size=128,
    seed=1337,
    logit_scale=33.3795,
    validation_features=None,
    validation_labels=None,
):
    """Train CLAP-S+'s adapter using its official text-classification objective."""
    support = F.normalize(support.detach(), p=2, dim=-1)
    text_anchors = F.normalize(text_anchors.detach(), p=2, dim=-1)
    adapter = ClapSAdapter(support.size(-1), reduction=4).to(support.device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=1e-5, weight_decay=1e-2
    )
    steps_per_epoch = max(1, math.ceil(support.size(0) / batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=max(1, epochs * steps_per_epoch),
        pct_start=0.2,
        anneal_strategy="cos",
        final_div_factor=100,
    )
    generator = torch.Generator(device=support.device)
    generator.manual_seed(seed)
    best_accuracy = -1.0
    best_state = None

    if validation_features is not None:
        validation_features = F.normalize(
            validation_features.detach(), p=2, dim=-1
        )
        validation_labels = validation_labels.detach()

    adapter.train()
    for _ in range(epochs):
        order = torch.randperm(
            support.size(0), generator=generator, device=support.device
        )
        for start in range(0, support.size(0), batch_size):
            indices = order[start : start + batch_size]
            adapted = adapter(support[indices])
            logits = float(logit_scale) * (adapted @ text_anchors.T)
            loss = F.cross_entropy(logits, labels[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if validation_features is not None:
            adapter.eval()
            with torch.no_grad():
                validation_adapted = adapter(validation_features)
                validation_logits = float(logit_scale) * (
                    validation_adapted @ text_anchors.T
                )
                validation_accuracy = (
                    validation_logits.argmax(dim=1) == validation_labels
                ).float().mean().item()
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_state = {
                    key: value.detach().clone()
                    for key, value in adapter.state_dict().items()
                }
            adapter.train()

    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter.eval()
    return adapter


def cache_logits_from_affinity(affinity, support_labels, num_classes, beta):
    weights = torch.exp(-float(beta) * (1.0 - affinity))
    logits = weights.new_zeros(weights.size(0), num_classes)
    logits.scatter_add_(
        1,
        support_labels.unsqueeze(0).expand(weights.size(0), -1),
        weights,
    )
    return logits


def clap_s_plus_components(
    query,
    support,
    support_labels,
    text_anchors,
    adapter,
    num_classes,
    logit_scale,
):
    """Return official CLAP-S+ text logits and adapted-query/raw-key affinity."""
    query = F.normalize(query, p=2, dim=-1)
    support = F.normalize(support, p=2, dim=-1)
    adapted_query = adapter(query)
    text_logits = float(logit_scale) * (adapted_query @ text_anchors.T)
    affinity = adapted_query @ support.T
    return text_logits, affinity


def clap_s_plus_logits(
    query,
    support,
    support_labels,
    text_anchors,
    adapter,
    num_classes,
    logit_scale,
    alpha,
    beta,
):
    text_logits, affinity = clap_s_plus_components(
        query,
        support,
        support_labels,
        text_anchors,
        adapter,
        num_classes,
        logit_scale,
    )
    cache_logits = cache_logits_from_affinity(
        affinity, support_labels, num_classes, beta
    )
    return (1.0 - float(alpha)) * text_logits + float(alpha) * cache_logits


def search_clap_s_hparams(records, num_classes):
    """Official CLAP-S+ alpha/beta grid, evaluated only on validation records."""
    beta_values = [
        i * (100.0 - 0.1) / 100.0 + 0.1 for i in range(100)
    ]
    alpha_values = torch.tensor(
        [i * (1.0 - 0.1) / 100.0 + 0.1 for i in range(100)],
        device=records[0][0].device,
    )
    best_acc = -1.0
    best_alpha = 0.1
    best_beta = 0.1

    for beta in beta_values:
        correct = torch.zeros_like(alpha_values)
        total = 0
        for text_logits, affinity, support_labels, targets in records:
            cache_logits = cache_logits_from_affinity(
                affinity, support_labels, num_classes, beta
            )
            fused = (
                (1.0 - alpha_values[:, None, None]) * text_logits[None]
                + alpha_values[:, None, None] * cache_logits[None]
            )
            correct += (fused.argmax(dim=-1) == targets[None]).sum(dim=1)
            total += targets.numel()
        accuracies = correct.float() / max(total, 1)
        value, index = accuracies.max(dim=0)
        if value.item() > best_acc:
            best_acc = value.item()
            best_alpha = alpha_values[index].item()
            best_beta = beta
    return best_alpha, best_beta, best_acc


def _audio_features(clap_model, wavs, device):
    with torch.no_grad():
        features = clap_model.get_audio_features(wavs)
    features = torch.as_tensor(features, device=device).float()
    return F.normalize(features, p=2, dim=-1)


def tune_clap_s_plus_episodic(
    clap_model,
    loader,
    prep_batch,
    device,
    n_way,
    k_shot,
    q_queries,
    max_tasks=200,
):
    """Tune CLAP-S+ fusion only on MetaAudio validation-class episodes."""
    records = []
    seen = 0
    scale = clap_logit_scale(clap_model)
    for batch in loader:
        _, _, _, _, y, batch_wavs, class_names = prep_batch(batch, 1)
        for episode_index, episode_wavs in enumerate(batch_wavs):
            support_num = n_way * k_shot
            query_num = n_way * q_queries
            support = _audio_features(
                clap_model, episode_wavs[:support_num], device
            )
            query = _audio_features(
                clap_model,
                episode_wavs[support_num : support_num + query_num],
                device,
            )
            support_labels = torch.arange(
                n_way, device=device
            ).repeat_interleave(k_shot)
            targets = y[episode_index][support_num:].to(device)
            train_text = clap_s_text_anchors(
                clap_model, class_names[episode_index], device, stage="train"
            )
            inference_text = clap_s_text_anchors(
                clap_model, class_names[episode_index], device, stage="inference"
            )
            adapter = train_clap_s_adapter(
                support,
                support_labels,
                train_text,
                epochs=20,
                batch_size=128,
                seed=seen,
                logit_scale=scale,
            )
            with torch.no_grad():
                text_logits, affinity = clap_s_plus_components(
                    query,
                    support,
                    support_labels,
                    inference_text,
                    adapter,
                    n_way,
                    scale,
                )
            records.append((text_logits, affinity, support_labels, targets))
            seen += 1
            if seen >= max_tasks:
                return search_clap_s_hparams(records, n_way)
    return search_clap_s_hparams(records, n_way)


def eval_step_clap_s_plus_var(
    clap_model,
    batch_wavs,
    q_num,
    y,
    batch_class_names,
    device,
    n_way,
    k_shot,
    q_queries,
    distance,
    train=False,
    alpha=0.1,
    beta=0.1,
    **kwargs,
):
    del q_num, distance, train, kwargs
    all_post = []
    support_num = n_way * k_shot
    query_num = n_way * q_queries
    scale = clap_logit_scale(clap_model)
    for episode_index, episode_wavs in enumerate(batch_wavs):
        support = _audio_features(
            clap_model, episode_wavs[:support_num], device
        )
        query = _audio_features(
            clap_model,
            episode_wavs[support_num : support_num + query_num],
            device,
        )
        support_labels = torch.arange(
            n_way, device=device
        ).repeat_interleave(k_shot)
        targets = y[episode_index][support_num:].to(device)
        train_text = clap_s_text_anchors(
            clap_model,
            batch_class_names[episode_index],
            device,
            stage="train",
        )
        inference_text = clap_s_text_anchors(
            clap_model,
            batch_class_names[episode_index],
            device,
            stage="inference",
        )
        adapter = train_clap_s_adapter(
            support,
            support_labels,
            train_text,
            epochs=20,
            batch_size=128,
            seed=episode_index,
            logit_scale=scale,
        )
        with torch.no_grad():
            logits = clap_s_plus_logits(
                query,
                support,
                support_labels,
                inference_text,
                adapter,
                n_way,
                scale,
                alpha,
                beta,
            )
            acc = (logits.argmax(dim=1) == targets).float().mean().item()
        all_post.append(acc)
    return 0, 0, sum(all_post) / len(all_post), all_post
