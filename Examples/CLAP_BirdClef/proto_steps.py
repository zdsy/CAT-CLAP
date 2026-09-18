"""Episodic CLAP methods evaluated in the paper.

TIP-Adapter-F, Treff Adapter, and CLAP-S+ live in shared modules.
"""

import torch
import torch.nn.functional as F

def eval_step_zero_shot_var(clap_model, batch_wavs, q_num, y, batch_class_names,
                            device, n_way, k_shot, q_queries, distance, train=False):
    """
    Frozen CLAP zero-shot evaluation.

    Uses the ZeroShotCLAP wrapper API:
        - get_text_anchors(class_names)
        - get_audio_features(wav_tensors)

    Support audio is ignored; class text anchors are the only prototypes.
    """
    if hasattr(clap_model, "eval"):
        clap_model.eval()

    post_acc_total = 0.0
    all_post = []

    meta_batch_size = len(batch_wavs)
    support_num = n_way * k_shot
    queries_per_ep = n_way * q_queries

    for idx in range(meta_batch_size):
        ep_wavs = batch_wavs[idx]
        ep_classes = batch_class_names[idx]

        wav_task_val = ep_wavs[support_num : support_num + queries_per_ep]
        y_task_val = y[idx][support_num : support_num + len(wav_task_val)].to(device)

        with torch.no_grad():
            t_k = clap_model.get_text_anchors(ep_classes).to(device)
            z_q = clap_model.get_audio_features(wav_task_val).to(device)
            logits = z_q @ t_k.T
            query_pred = torch.argmax(logits, dim=1)

        correct = (query_pred == y_task_val).sum().item()
        post_acc = correct / max(len(y_task_val), 1)

        post_acc_total += post_acc
        all_post.append(post_acc)

    avg_post = post_acc_total / meta_batch_size
    return 0, 0, avg_post, all_post


def eval_step_clap_audio_proto(
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
    alpha=0.5,
    beta=5.0,
):
    """
    Frozen CLAP audio encoder + ProtoNet baseline.

    This baseline uses no text branch, no adapter, no learnable memory, and no
    support-set fine-tuning. Audio prototypes are vanilla ProtoNet class means
    from frozen CLAP audio embeddings.
    """

    if hasattr(clap_model, "eval"):
        clap_model.eval()

    post_acc_total = 0.0
    all_post = []

    meta_batch_size = len(batch_wavs)
    support_num = n_way * k_shot
    queries_per_ep = n_way * q_queries

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    def _get_audio_emb(wav_tensors):
        with torch.no_grad():
            emb = clap_model.get_audio_features(wav_tensors)
        emb = _to_tensor(emb).to(device).float()
        return F.normalize(emb, p=2, dim=-1)

    def _make_audio_logits(query_feats, proto_feats):
        query_feats = F.normalize(query_feats, p=2, dim=-1)
        proto_feats = F.normalize(proto_feats, p=2, dim=-1)
        if isinstance(distance, str) and distance.lower() in ["euclidean", "l2"]:
            dist = torch.cdist(query_feats, proto_feats, p=2).pow(2)
            return -beta * dist
        return beta * (query_feats @ proto_feats.T)

    for idx in range(meta_batch_size):
        ep_wavs = batch_wavs[idx]
        wav_task_train = ep_wavs[:support_num]
        wav_task_val = ep_wavs[support_num : support_num + queries_per_ep]
        y_task_val = y[idx][support_num:].to(device)

        with torch.no_grad():
            F_s = _get_audio_emb(wav_task_train)
            F_q = _get_audio_emb(wav_task_val)
            audio_proto = F_s.view(n_way, k_shot, -1).mean(dim=1)
            audio_proto = F.normalize(audio_proto, p=2, dim=-1)

            logits_audio = _make_audio_logits(F_q, audio_proto)

            query_pred = torch.argmax(logits_audio, dim=1)
            correct = (query_pred == y_task_val).sum().item()
            post_acc = correct / len(y_task_val)

        post_acc_total += post_acc
        all_post.append(post_acc)

    avg_post = post_acc_total / meta_batch_size
    return 0, 0, avg_post, all_post


def eval_step_tmclap_no_audio(
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
    alpha=0.5,
    beta=5.0,
    use_finetune=True,
    ft_steps=30,
    lr=1e-3,
    adapter_reduction=4,
    adapter_residual_ratio=0.2,
    adapter_arch="tclap",
    use_adapter=True,
    use_tcam=True,
    train_text_memory=True,
    lambda_align=0.5,
    align_temperature=1.0,
    weight_decay=0.05,
    can_temperature=0.025,
    tcam_score_mode="mean",
    tcam_top_m=8,
    tcam_attn_mode="residual",
    tcam_weight=0.5,
    text_mem_weight=0.5,
    use_rdft_finetune=False,
    eps=1e-8,
):
    """
    TMCLAP-no-audio-memory: clean TMCLAP variant with no learnable audio memory.

    Learnable episode parameters are only the residual audio adapter and text
    memory. Audio prototypes are always computed from frozen support CLAP
    embeddings after the current adapter, while inference fuses TCAM and text
    branches with probability fusion.
    """

    if hasattr(clap_model, "eval"):
        clap_model.eval()

    post_acc_total = 0.0
    all_post = []

    meta_batch_size = len(batch_wavs)
    support_num = n_way * k_shot
    queries_per_ep = n_way * q_queries

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    def _get_audio_emb(wav_tensors):
        with torch.no_grad():
            emb = clap_model.get_audio_features(wav_tensors)
        emb = _to_tensor(emb).to(device).float()
        return F.normalize(emb, p=2, dim=-1)

    def _get_audio_frames(wav_tensors):
        with torch.no_grad():
            frames = clap_model.get_audio_frame_features(wav_tensors)
        frames = _to_tensor(frames).to(device).float()
        if frames.dim() == 2:
            frames = frames.unsqueeze(1)
        return F.normalize(frames, p=2, dim=-1)

    def _get_text_emb(class_names):
        with torch.no_grad():
            emb = clap_model.get_text_anchors(class_names)
        emb = _to_tensor(emb).to(device).float()
        return F.normalize(emb, p=2, dim=-1)

    def _make_logits(query_feats, proto_feats):
        query_feats = F.normalize(query_feats, p=2, dim=-1)
        proto_feats = F.normalize(proto_feats, p=2, dim=-1)
        if isinstance(distance, str) and distance.lower() in ["euclidean", "l2"]:
            dist = torch.cdist(query_feats, proto_feats, p=2).pow(2)
            return -beta * dist
        return beta * (query_feats @ proto_feats.T)

    def _make_pairwise_logits(query_pair, proto_pair):
        query_pair = F.normalize(query_pair, p=2, dim=-1)
        proto_pair = F.normalize(proto_pair, p=2, dim=-1)
        if isinstance(distance, str) and distance.lower() in ["euclidean", "l2"]:
            dist = (query_pair - proto_pair).pow(2).sum(dim=-1)
            return -beta * dist
        return beta * (query_pair * proto_pair).sum(dim=-1)

    def _proto_clap_logits(query_feats, audio_proto, text_proto):
        audio_logits = _make_logits(query_feats, audio_proto)
        text_logits = _make_logits(query_feats, text_proto)
        return alpha * audio_logits + (1.0 - alpha) * text_logits

    def _alignment_loss(audio_proto, text_proto):
        audio_proto = F.normalize(audio_proto, p=2, dim=-1)
        text_proto = F.normalize(text_proto, p=2, dim=-1)
        logits_a2t = audio_proto @ text_proto.T / align_temperature
        logits_t2a = text_proto @ audio_proto.T / align_temperature
        labels = torch.arange(audio_proto.size(0), device=device)
        return F.cross_entropy(logits_a2t, labels) + F.cross_entropy(logits_t2a, labels)

    def _can_pairwise_features(support_frames, query_frames):
        d = support_frames.shape[-1]
        support_frames = support_frames.view(n_way, k_shot, -1, d).mean(dim=1)
        support_frames = F.normalize(support_frames, p=2, dim=-1)
        query_frames = F.normalize(query_frames, p=2, dim=-1)

        sim = torch.einsum("ntd,qsd->nqts", support_frames, query_frames)
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
        else:
            raise ValueError(f"Unknown TCAM attention mode: {tcam_attn_mode}")

        proto_pair = torch.einsum("ntd,nqt->qnd", support_frames, support_attn)
        proto_pair = proto_pair / support_attn.sum(dim=-1).transpose(0, 1).unsqueeze(-1).clamp(min=eps)

        query_pair = torch.einsum("qtd,nqt->qnd", query_frames, query_attn)
        query_pair = query_pair / query_attn.permute(1, 0, 2).sum(dim=-1, keepdim=True).clamp(min=eps)

        return F.normalize(query_pair, p=2, dim=-1), F.normalize(proto_pair, p=2, dim=-1)

    weight_sum = max(tcam_weight + text_mem_weight, eps)
    tcam_weight = tcam_weight / weight_sum
    text_mem_weight = text_mem_weight / weight_sum

    for idx in range(meta_batch_size):
        ep_wavs = batch_wavs[idx]
        ep_classes = batch_class_names[idx]

        wav_task_train = ep_wavs[:support_num]
        wav_task_val = ep_wavs[support_num : support_num + queries_per_ep]
        y_task_val = y[idx][support_num:].to(device)

        F_s_init = _get_audio_emb(wav_task_train)
        F_q_init = _get_audio_emb(wav_task_val)
        T_init = _get_text_emb(ep_classes)

        d = F_s_init.shape[-1]
        hidden_dim = max(d // adapter_reduction, 1)
        support_labels = torch.arange(n_way, device=device).repeat_interleave(k_shot)
        support_index_grid = torch.arange(support_num, device=device).view(n_way, k_shot)

        text_memory = torch.nn.Parameter(T_init.clone())

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

        def _adapter(x):
            if not use_finetune or not use_adapter:
                return F.normalize(x, p=2, dim=-1)
            if adapter_arch == "protoclip":
                z = adapter_norm_down(adapter_down(x))
                z = adapter_norm_up(adapter_up(z))
            else:
                z = x @ W_down + b_down
                z = F.relu(z, inplace=False)
                z = z @ W_up + b_up
            out = adapter_residual_ratio * z + (1.0 - adapter_residual_ratio) * x
            return F.normalize(out, p=2, dim=-1)

        def _support_audio_proto(support_feats=None, shots_per_class=None):
            if support_feats is None:
                support_feats = _adapter(F_s_init)
            if shots_per_class is None:
                shots_per_class = k_shot
            audio_proto = support_feats.view(n_way, shots_per_class, d).mean(dim=1)
            return F.normalize(audio_proto, p=2, dim=-1)

        def _fused_prob_loss(query_feats, audio_proto, text_proto, labels):
            audio_logits = _make_logits(query_feats, audio_proto)
            text_logits = _make_logits(query_feats, text_proto)
            probs = alpha * F.softmax(audio_logits, dim=-1) + (1.0 - alpha) * F.softmax(text_logits, dim=-1)
            return F.nll_loss(torch.log(probs.clamp(min=eps)), labels)

        if use_finetune:
            params = []
            if use_adapter:
                params.extend(adapter_params)
            if train_text_memory:
                params.append(text_memory)

            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                eps=1e-4,
            )

            for _ in range(ft_steps):
                optimizer.zero_grad()

                text_proto = F.normalize(text_memory, p=2, dim=-1)

                if use_rdft_finetune and k_shot > 1:
                    heldout = torch.randint(0, k_shot, (n_way,), device=device)
                    class_ids = torch.arange(n_way, device=device)
                    pseudo_query_idx = support_index_grid[class_ids, heldout]

                    pseudo_support_mask = torch.ones(n_way, k_shot, dtype=torch.bool, device=device)
                    pseudo_support_mask[class_ids, heldout] = False
                    pseudo_support_idx = support_index_grid[pseudo_support_mask].view(n_way, k_shot - 1).reshape(-1)

                    pseudo_support_feats = _adapter(F_s_init[pseudo_support_idx])
                    audio_proto = _support_audio_proto(
                        pseudo_support_feats,
                        shots_per_class=k_shot - 1,
                    )
                    support_query = _adapter(F_s_init[pseudo_query_idx])
                    cls_loss = _fused_prob_loss(support_query, audio_proto, text_proto, class_ids)
                else:
                    support_query = _adapter(F_s_init)
                    audio_proto = _support_audio_proto(support_query)
                    cls_loss = _fused_prob_loss(support_query, audio_proto, text_proto, support_labels)

                align_loss = _alignment_loss(audio_proto, text_proto)
                loss = cls_loss + lambda_align * align_loss

                loss.backward()
                optimizer.step()

        with torch.no_grad():
            query_feats = _adapter(F_q_init)

            text_proto = F.normalize(text_memory, p=2, dim=-1)
            text_logits = _make_logits(query_feats, text_proto)

            if use_tcam and tcam_weight > 0.0:
                support_frames = _get_audio_frames(wav_task_train)
                query_frames = _get_audio_frames(wav_task_val)
                query_pair, proto_pair = _can_pairwise_features(support_frames, query_frames)
                query_pair = _adapter(query_pair.reshape(-1, d)).view(queries_per_ep, n_way, d)
                proto_pair = _adapter(proto_pair.reshape(-1, d)).view(queries_per_ep, n_way, d)
                audio_logits = _make_pairwise_logits(query_pair, proto_pair)
            elif tcam_weight > 0.0:
                audio_logits = _make_logits(query_feats, _support_audio_proto())
            else:
                audio_logits = torch.zeros_like(text_logits)

            probs = (
                tcam_weight * F.softmax(audio_logits, dim=-1)
                + text_mem_weight * F.softmax(text_logits, dim=-1)
            )
            query_pred = torch.argmax(probs, dim=1)
            correct = (query_pred == y_task_val).sum().item()
            post_acc = correct / len(y_task_val)

        post_acc_total += post_acc
        all_post.append(post_acc)

    avg_post = post_acc_total / meta_batch_size
    return 0, 0, avg_post, all_post
