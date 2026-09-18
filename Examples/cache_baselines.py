"""TIP-Adapter-F and Treff Adapter baselines for episodic and all-way evaluation."""

import torch
import torch.nn.functional as F

def official_clap_logit_scale(clap_model):
    """Return the pretrained scale used by MS-CLAP compute_similarity()."""
    wrapper = getattr(clap_model, "model", clap_model)
    clap = getattr(wrapper, "clap", None)
    logit_scale = getattr(clap, "logit_scale", None)
    if logit_scale is None:
        raise RuntimeError("Could not locate MS-CLAP\x27s pretrained logit_scale")
    return float(logit_scale.detach().exp().cpu())


def cache_logits(query, cache, labels, num_classes, beta=5.5):
    """Aggregate exponential cache affinities without a dense one-hot matrix."""
    affinity = torch.exp(-beta + beta * (query @ cache.T))
    logits = affinity.new_zeros(affinity.size(0), num_classes)
    logits.scatter_add_(
        1,
        labels.unsqueeze(0).expand(affinity.size(0), -1),
        affinity,
    )
    return logits


def adapt_treff_official_scalable(
    support,
    labels,
    text_anchors,
    num_classes,
    steps=20,
    lr=1e-4,
    batch_size=64,
    alpha=1.0,
    beta=5.5,
    seed=1337,
    text_logit_scale=1.0,
):
    """Treff optimization with official hyperparameters and scalable pseudo-queries.

    The official support-by-support objective is quadratic. For all-way tasks we
    preserve its loss and parameterization but sample support pseudo-queries.
    """
    support = F.normalize(support.detach(), p=2, dim=-1)
    text_anchors = F.normalize(text_anchors.detach(), p=2, dim=-1)
    dim = support.size(-1)
    projection = torch.nn.Parameter(torch.eye(dim, device=support.device))
    learned_alpha = torch.nn.Parameter(
        torch.tensor(float(alpha), device=support.device)
    )
    optimizer = torch.optim.AdamW(
        [projection, learned_alpha], lr=lr, eps=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps)
    )
    generator = torch.Generator(device=support.device)
    generator.manual_seed(seed)

    for _ in range(steps):
        optimizer.zero_grad()
        projected_support = F.linear(support, projection)
        count = min(batch_size, support.size(0))
        indices = torch.randperm(
            support.size(0), generator=generator, device=support.device
        )[:count]
        projected_queries = projected_support[indices]
        retrieval_logits = cache_logits(
            projected_queries,
            projected_support,
            labels,
            num_classes,
            beta,
        )
        # Official Treff retains raw CLIP/CLAP embeddings for the text branch.
        text_logits = text_logit_scale * (support[indices] @ text_anchors.T)
        loss = F.cross_entropy(
            text_logits + learned_alpha * retrieval_logits,
            labels[indices],
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

    return (
        projection.detach(),
        F.linear(support, projection.detach()),
        learned_alpha.detach(),
    )


def adapt_treff_official_full(
    support,
    labels,
    text_anchors,
    num_classes,
    steps=20,
    lr=1e-4,
    batch_size=256,
    alpha=1.0,
    beta=5.5,
    text_logit_scale=1.0,
    seed=None,
):
    """Treff xattention using every support sample as a pseudo-query.

    Query chunks accumulate the exact full-support gradient before each
    optimizer update, avoiding materialization of the dense S-by-S affinity.
    """
    support = F.normalize(support.detach(), p=2, dim=-1)
    text_anchors = F.normalize(text_anchors.detach(), p=2, dim=-1)
    dim = support.size(-1)
    projection = torch.nn.Parameter(torch.eye(dim, device=support.device))
    learned_alpha = torch.nn.Parameter(
        torch.tensor(float(alpha), device=support.device)
    )
    optimizer = torch.optim.AdamW(
        [projection, learned_alpha], lr=lr, eps=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps)
    )
    count = min(batch_size, support.size(0))

    for _ in range(steps):
        optimizer.zero_grad()
        projected = F.linear(support, projection)
        projected_leaf = projected.detach().requires_grad_(True)
        for start in range(0, support.size(0), count):
            indices = slice(start, min(start + count, support.size(0)))
            retrieval_logits = cache_logits(
                projected_leaf[indices], projected_leaf, labels, num_classes, beta
            )
            text_logits = text_logit_scale * (
                support[indices] @ text_anchors.T
            )
            loss = F.cross_entropy(
                text_logits + learned_alpha * retrieval_logits,
                labels[indices],
                reduction="sum",
            ) / support.size(0)
            loss.backward()

        projected.backward(projected_leaf.grad)
        optimizer.step()
        scheduler.step()

    return (
        projection.detach(),
        F.linear(support, projection.detach()),
        learned_alpha.detach(),
    )


def adapt_tip_adapter_f(
    support,
    labels,
    text_anchors,
    num_classes,
    steps=20,
    lr=1e-3,
    batch_size=256,
    alpha=1.17,
    beta=1.0,
    seed=1337,
    text_logit_scale=1.0,
    validation=None,
    validation_labels=None,
    return_history=False,
):
    """Train Tip-Adapter-F cache keys for the requested number of epochs."""
    support = F.normalize(support.detach(), p=2, dim=-1)
    text_anchors = F.normalize(text_anchors.detach(), p=2, dim=-1)
    cache_keys = torch.nn.Parameter(support.clone())
    optimizer = torch.optim.AdamW([cache_keys], lr=lr, eps=1e-4)
    count = support.size(0) if batch_size is None else min(batch_size, support.size(0))
    steps_per_epoch = max(1, (support.size(0) + count - 1) // count)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps * steps_per_epoch)
    )
    generator = torch.Generator(device=support.device)
    generator.manual_seed(seed)
    history = []
    best_state = None
    best_accuracy = -1.0
    best_epoch = steps

    for epoch in range(steps):
        order = torch.randperm(
            support.size(0), generator=generator, device=support.device
        )
        for batch_start in range(0, support.size(0), count):
            indices = order[batch_start : batch_start + count]
            queries = support[indices]
            retrieval_logits = cache_logits(
                queries, cache_keys, labels, num_classes, beta
            )
            text_logits = text_logit_scale * (queries @ text_anchors.T)
            loss = F.cross_entropy(
                text_logits + float(alpha) * retrieval_logits,
                labels[indices],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        state = cache_keys.detach().clone()
        if return_history:
            history.append(state)
        if validation is not None:
            with torch.no_grad():
                val_retrieval = cache_logits(
                    validation, state, labels, num_classes, beta
                )
                val_text = text_logit_scale * (validation @ text_anchors.T)
                accuracy = (
                    (val_text + float(alpha) * val_retrieval).argmax(dim=1)
                    == validation_labels
                ).float().mean().item()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = state
                best_epoch = epoch + 1

    selected = best_state if best_state is not None else cache_keys.detach()
    if return_history:
        return selected, best_epoch, best_accuracy, history
    return selected, best_epoch, best_accuracy


def search_tip_adapter_hparams(records, num_classes):
    """Official Tip-Adapter alpha/beta grid evaluated on validation records."""
    beta_values = [i * (7.0 - 0.1) / 200.0 + 0.1 for i in range(200)]
    alpha_values = torch.tensor(
        [i * (3.0 - 0.1) / 20.0 + 0.1 for i in range(20)],
        device=records[0][0].device,
    )
    best_accuracy = -1.0
    best_alpha = 1.17
    best_beta = 1.0
    for beta in beta_values:
        correct = torch.zeros_like(alpha_values)
        total = 0
        for text_logits, affinity, support_labels, targets in records:
            weights = torch.exp(-float(beta) * (1.0 - affinity))
            cache = weights.new_zeros(weights.size(0), num_classes)
            cache.scatter_add_(
                1,
                support_labels.unsqueeze(0).expand(weights.size(0), -1),
                weights,
            )
            fused = (
                text_logits.unsqueeze(0)
                + alpha_values[:, None, None] * cache.unsqueeze(0)
            )
            correct += (
                fused.argmax(dim=-1) == targets.unsqueeze(0)
            ).sum(dim=1)
            total += targets.numel()
        accuracies = correct.float() / max(total, 1)
        value, index = accuracies.max(dim=0)
        if value.item() > best_accuracy:
            best_accuracy = value.item()
            best_alpha = alpha_values[index].item()
            best_beta = beta
    return best_alpha, best_beta, best_accuracy


def tune_tip_adapter_f_episodic(
    clap_model,
    loader,
    prep_batch,
    device,
    n_way,
    k_shot,
    q_queries,
    epochs=20,
    max_tasks=200,
):
    """Select TIP-F epoch and fusion parameters on validation episodes."""
    epoch_records = [[] for _ in range(epochs)]
    epoch_correct = torch.zeros(epochs, device=device)
    total = 0
    scale = official_clap_logit_scale(clap_model)
    seen = 0
    for batch in loader:
        _, _, _, _, y, batch_wavs, class_names = prep_batch(batch, 1)
        for episode_index, episode_wavs in enumerate(batch_wavs):
            support_num = n_way * k_shot
            query_num = n_way * q_queries
            support = _episode_features(
                clap_model, episode_wavs[:support_num], device
            )
            query = _episode_features(
                clap_model,
                episode_wavs[support_num : support_num + query_num],
                device,
            )
            labels = torch.arange(
                n_way, device=device
            ).repeat_interleave(k_shot)
            targets = y[episode_index][support_num:].to(device)
            with torch.no_grad():
                text_anchors = clap_model.get_text_anchors(
                    class_names[episode_index]
                ).to(device).float()
                text_anchors = F.normalize(text_anchors, p=2, dim=-1)
                text_logits = scale * (query @ text_anchors.T)
            _, _, _, history = adapt_tip_adapter_f(
                support,
                labels,
                text_anchors,
                n_way,
                steps=epochs,
                lr=1e-3,
                batch_size=256,
                alpha=1.17,
                beta=1.0,
                seed=seen,
                text_logit_scale=scale,
                return_history=True,
            )
            with torch.no_grad():
                for epoch, keys in enumerate(history):
                    affinity = query @ keys.T
                    weights = torch.exp(-(1.0 - affinity))
                    cache = weights.new_zeros(weights.size(0), n_way)
                    cache.scatter_add_(
                        1,
                        labels.unsqueeze(0).expand(weights.size(0), -1),
                        weights,
                    )
                    epoch_correct[epoch] += (
                        (text_logits + 1.17 * cache).argmax(dim=1) == targets
                    ).sum()
                    epoch_records[epoch].append(
                        (text_logits, affinity, labels, targets)
                    )
            total += targets.numel()
            seen += 1
            if seen >= max_tasks:
                break
        if seen >= max_tasks:
            break
    best_epoch_index = int(
        (epoch_correct / max(total, 1)).argmax().item()
    )
    alpha, beta, accuracy = search_tip_adapter_hparams(
        epoch_records[best_epoch_index], n_way
    )
    return best_epoch_index + 1, alpha, beta, accuracy


def _episode_features(clap_model, wavs, device):
    features = clap_model.get_audio_features(wavs)
    return F.normalize(
        torch.as_tensor(features, device=device).float(), p=2, dim=-1
    )


def _eval_cache_baseline(
    clap_model,
    batch_wavs,
    y,
    batch_class_names,
    device,
    n_way,
    k_shot,
    q_queries,
    train_treff,
    train_tip_f=False,
    ft_steps=20,
    lr=1e-4,
    alpha=1.0,
    beta=5.5,
):
    post_acc_total = 0.0
    all_post = []
    query_count = n_way * q_queries
    text_logit_scale = (
        official_clap_logit_scale(clap_model)
        if train_treff or train_tip_f
        else 1.0
    )

    for episode_index, episode_wavs in enumerate(batch_wavs):
        class_names = batch_class_names[episode_index]
        support_wavs = episode_wavs[: n_way * k_shot]
        query_wavs = episode_wavs[n_way * k_shot : n_way * k_shot + query_count]
        query_labels = y[episode_index][n_way * k_shot :].to(device)

        with torch.no_grad():
            text_anchors = clap_model.get_text_anchors(class_names).to(device).float()
            text_anchors = F.normalize(text_anchors, p=2, dim=-1)
            support = _episode_features(clap_model, support_wavs, device)
            query = _episode_features(clap_model, query_wavs, device)

        support_labels = torch.arange(n_way, device=device).repeat_interleave(k_shot)
        if train_treff:
            projection, adapted_support, learned_alpha = adapt_treff_official_scalable(
                support,
                support_labels,
                text_anchors,
                n_way,
                steps=ft_steps,
                lr=lr,
                batch_size=support.size(0),
                alpha=alpha,
                beta=beta,
                seed=episode_index,
                text_logit_scale=text_logit_scale,
            )
            adapted_query = F.linear(query, projection)
        elif train_tip_f:
            adapted_support, _, _ = adapt_tip_adapter_f(
                support,
                support_labels,
                text_anchors,
                n_way,
                steps=ft_steps,
                lr=lr,
                batch_size=support.size(0),
                alpha=alpha,
                beta=beta,
                seed=episode_index,
                text_logit_scale=text_logit_scale,
            )
            adapted_query = query
            learned_alpha = query.new_tensor(float(alpha))
        else:
            adapted_support = support
            adapted_query = query
            learned_alpha = query.new_tensor(float(alpha))

        with torch.no_grad():
            retrieval_logits = cache_logits(
                adapted_query,
                adapted_support,
                support_labels,
                n_way,
                beta,
            )
            text_logits = text_logit_scale * (query @ text_anchors.T)
            predictions = (
                text_logits + learned_alpha * retrieval_logits
            ).argmax(dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
        post_acc_total += accuracy
        all_post.append(accuracy)

    return 0, 0, post_acc_total / len(batch_wavs), all_post


def eval_step_treff_official_var(
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
    **kwargs,
):
    del q_num, distance, train
    return _eval_cache_baseline(
        clap_model,
        batch_wavs,
        y,
        batch_class_names,
        device,
        n_way,
        k_shot,
        q_queries,
        train_treff=True,
        ft_steps=int(kwargs.get("ft_steps", 20)),
        lr=float(kwargs.get("lr", 1e-4)),
    )


def eval_step_tip_adapter_f_var(
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
    **kwargs,
):
    del q_num, distance, train
    return _eval_cache_baseline(
        clap_model,
        batch_wavs,
        y,
        batch_class_names,
        device,
        n_way,
        k_shot,
        q_queries,
        train_treff=False,
        train_tip_f=True,
        ft_steps=int(kwargs.get("ft_steps", 20)),
        lr=float(kwargs.get("lr", 1e-3)),
        alpha=float(kwargs.get("tip_alpha", 1.17)),
        beta=float(kwargs.get("tip_beta", 1.0)),
    )
