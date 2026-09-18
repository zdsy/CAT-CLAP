import torch
import torch.nn as nn
import torch.nn.functional as F
from msclap import CLAP


class ZeroShotCLAP(nn.Module):
    def __init__(self, model_id=None, device="cuda", version="2023", sample_rate=44100, register_frame_hook=True):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.sample_rate = sample_rate
        self.frame_features_cache = []
        self._attribute_embedding_cache = None

        use_cuda = self.device.type == "cuda"
        self.model = CLAP(model_id, version=version, use_cuda=use_cuda)
        self.model.read_audio = self._bypass_read_audio
        self._set_eval()
        if register_frame_hook:
            self._register_frame_hook()

    def _set_eval(self):
        for attr in ("model", "clap", "audio_encoder", "caption_encoder"):
            module = getattr(self.model, attr, None)
            if isinstance(module, nn.Module):
                module.eval()

    def _register_frame_hook(self):
        """
        Capture real HTSAT frame features from msclap before temporal pooling.

        msclap's public API only returns clip embeddings. Internally, HTSAT builds
        a tensor shaped [B, 768, F, T] immediately before tscam_conv. Averaging the
        frequency axis gives frame embeddings [B, T, 768], which can then pass
        through msclap's audio projection to enter the shared CLAP space.
        """
        def hook_fn(module, inputs, output):
            if not inputs:
                return

            frame_map = inputs[0]
            if not torch.is_tensor(frame_map):
                return

            if frame_map.dim() == 4:
                frames = frame_map.mean(dim=2).transpose(1, 2).contiguous()
            elif frame_map.dim() == 3:
                frames = frame_map.transpose(1, 2).contiguous()
            else:
                return

            self.frame_features_cache.append(frames.detach())

        try:
            target_layer = self.model.clap.audio_encoder.base.htsat.tscam_conv
            target_layer.register_forward_hook(hook_fn)
        except AttributeError:
            print("Warning: Could not find msclap HTSAT tscam_conv for frame-feature hook.")

    def _bypass_read_audio(self, audio_tensor, resample=False):
        if not torch.is_tensor(audio_tensor):
            audio_tensor = torch.as_tensor(audio_tensor, dtype=torch.float32)
        else:
            audio_tensor = audio_tensor.float()

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.dim() > 2:
            audio_tensor = audio_tensor.squeeze()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

        return audio_tensor.detach().cpu(), self.sample_rate

    def _to_waveform_list(self, audio_waveforms):
        if isinstance(audio_waveforms, list):
            return [
                torch.as_tensor(w, dtype=torch.float32).detach().cpu()
                for w in audio_waveforms
            ]

        if torch.is_tensor(audio_waveforms):
            audio_waveforms = audio_waveforms.detach().cpu().float()
            if audio_waveforms.dim() == 1:
                return [audio_waveforms]
            return [w for w in audio_waveforms]

        raise ValueError("audio_waveforms must be a list of tensors or a tensor")

    def _get_audio_embeddings_from_tensors(self, audio_tensors, batch_size=32):
        all_embeds = []
        for i in range(0, len(audio_tensors), batch_size):
            chunk = audio_tensors[i : i + batch_size]
            chunk_embeds = self.model.get_audio_embeddings(chunk)
            if not torch.is_tensor(chunk_embeds):
                chunk_embeds = torch.as_tensor(chunk_embeds)
            all_embeds.append(chunk_embeds)
        return torch.cat(all_embeds, dim=0)

    def _project_frame_features(self, frames):
        projection = self.model.clap.audio_encoder.projection
        frames = frames.to(self.device)
        bsz, time_steps, dim = frames.shape
        frames = projection(frames.reshape(bsz * time_steps, dim))
        return frames.reshape(bsz, time_steps, -1)

    def _speaker_name_from_class_name(self, class_name):
        class_name = str(class_name).strip()
        return class_name.replace("_", " ").strip()

    def _text_prompts_for_class(self, class_name):
        speaker_name = self._speaker_name_from_class_name(class_name)
        return [
            speaker_name,
            f"a recording of speaker {speaker_name}",
            f"the voice of speaker {speaker_name}",
            f"a person speaking with speaker identity {speaker_name}",
        ]

    def _universal_attribute_prompts(self):
        """
        Domain-agnostic acoustic attributes used for support-conditioned text anchors.

        These prompts describe morphology, spectrum, texture, and temporal structure
        rather than dataset classes such as speakers, birds, or sound events.
        """
        return [
            "a short impulsive sound",
            "a long sustained sound",
            "a repeating sound",
            "a rhythmic sound",
            "an irregular sound",
            "a sparse sound with silence between events",
            "a dense continuous sound texture",
            "a sound with a sharp attack",
            "a sound with a gradual onset",
            "a sound with abrupt changes",
            "a sound with smooth changes",
            "a tonal sound",
            "a harmonic sound",
            "a noisy sound",
            "a rough textured sound",
            "a smooth textured sound",
            "a bright high frequency sound",
            "a dark low frequency sound",
            "a muffled sound",
            "a clear sound",
            "a metallic sound",
            "a percussive sound",
            "a resonant sound",
            "a dry non reverberant sound",
            "a reverberant sound",
            "a foreground sound with background noise",
            "a sound with strong background ambience",
            "a sound with a stable pitch",
            "a sound with changing pitch",
            "a sound with a wide frequency range",
            "a sound with a narrow frequency range",
            "a low energy sound",
            "a high energy sound",
            "a sound with fluctuating energy",
            "a steady even sound",
            "a sound made of many small events",
            "a sound dominated by one salient event",
            "a natural acoustic sound",
            "a mechanical acoustic sound",
            "a human produced acoustic sound",
        ]

    @torch.no_grad()
    def get_universal_attribute_embeddings(self):
        if self._attribute_embedding_cache is not None:
            return self._attribute_embedding_cache.to(self.device)

        prompts = self._universal_attribute_prompts()
        embeds = self.model.get_text_embeddings(prompts)
        if not torch.is_tensor(embeds):
            embeds = torch.as_tensor(embeds)
        embeds = F.normalize(embeds.to(self.device), p=2, dim=-1)
        self._attribute_embedding_cache = embeds.detach().cpu()
        return embeds

    @torch.no_grad()
    def get_attribute_enhanced_text_anchors(
        self,
        class_names,
        support_audio_features=None,
        audio_waveforms=None,
        k_shot=None,
        attr_weight=1.0,
        attr_temperature=20.0,
        top_k=None,
    ):
        """
        Build support-conditioned text anchors from a universal acoustic attribute bank.

        The base class anchor keeps the normal class-name prompts. A class-level
        support audio prototype retrieves or softly mixes domain-agnostic acoustic
        attribute prompts in CLAP space, then the retrieved attribute vector is
        fused back into the text anchor.
        """
        base_text = self.get_text_anchors(class_names)

        if support_audio_features is None:
            if audio_waveforms is None:
                return base_text
            support_audio_features = self.get_audio_features(audio_waveforms)
        elif not torch.is_tensor(support_audio_features):
            support_audio_features = torch.as_tensor(support_audio_features)

        support_audio_features = F.normalize(
            support_audio_features.to(self.device).float(), p=2, dim=-1
        )

        num_classes = len(class_names)
        if support_audio_features.size(0) == num_classes:
            support_proto = support_audio_features
        else:
            if k_shot is None:
                if support_audio_features.size(0) % num_classes != 0:
                    raise ValueError(
                        "Cannot infer k_shot from support_audio_features. "
                        "Pass k_shot explicitly."
                    )
                k_shot = support_audio_features.size(0) // num_classes
            support_proto = support_audio_features.view(num_classes, k_shot, -1).mean(dim=1)
            support_proto = F.normalize(support_proto, p=2, dim=-1)

        attr_bank = self.get_universal_attribute_embeddings()
        scores = support_proto @ attr_bank.T

        if top_k is not None and 0 < top_k < attr_bank.size(0):
            top_vals, top_idx = torch.topk(scores, k=top_k, dim=-1)
            weights = F.softmax(top_vals * attr_temperature, dim=-1)
            attr_selected = attr_bank[top_idx]
            attr_proto = torch.einsum("nk,nkd->nd", weights, attr_selected)
        else:
            weights = F.softmax(scores * attr_temperature, dim=-1)
            attr_proto = weights @ attr_bank

        attr_proto = F.normalize(attr_proto, p=2, dim=-1)
        enhanced = F.normalize(base_text + attr_weight * attr_proto, p=2, dim=-1)
        return enhanced

    @torch.no_grad()
    def get_text_anchors(self, class_names, audio_waveforms=None):
        """Return class-name text anchors.

        The optional audio_waveforms argument is accepted for backward
        compatibility, but support-conditioned text augmentation now lives in
        get_attribute_enhanced_text_anchors().
        """
        text_anchors = []

        for class_name in class_names:
            prompts = self._text_prompts_for_class(class_name)
            text_embeds = self.model.get_text_embeddings(prompts)
            if not torch.is_tensor(text_embeds):
                text_embeds = torch.as_tensor(text_embeds)
            text_embeds = text_embeds.to(self.device)
            text_embed = text_embeds.mean(dim=0, keepdim=True)
            text_anchors.append(F.normalize(text_embed, p=2, dim=-1))

        return torch.cat(text_anchors, dim=0)

    @torch.no_grad()
    def get_audio_features(self, audio_waveforms):
        # get_audio_embeddings also triggers the optional frame hook. Most callers
        # only need clip-level features, so clear the cache to avoid accumulating
        # frame tensors across thousands of episodes.
        self.frame_features_cache = []
        audio_tensors = self._to_waveform_list(audio_waveforms)
        audio_embeds = self._get_audio_embeddings_from_tensors(audio_tensors)
        self.frame_features_cache = []
        audio_embeds = audio_embeds.to(self.device)
        return F.normalize(audio_embeds, p=2, dim=-1)

    @torch.no_grad()
    def get_audio_frame_features(self, audio_waveforms, batch_size=32):
        """
        Return real msclap/HTSAT frame-level embeddings in shared CLAP space.

        Shape: [Batch, Time, Dim], where Dim matches get_audio_features() and
        get_text_anchors() after msclap's audio projection.
        """
        audio_tensors = self._to_waveform_list(audio_waveforms)
        all_frames = []

        for i in range(0, len(audio_tensors), batch_size):
            chunk = audio_tensors[i : i + batch_size]
            self.frame_features_cache = []

            _ = self.model.get_audio_embeddings(chunk)

            if not self.frame_features_cache:
                raise RuntimeError(
                    "Hook failed to capture msclap frame features. "
                    "Check that the audio encoder has base.htsat.tscam_conv."
                )

            if len(self.frame_features_cache) == 1:
                frames = self.frame_features_cache[0]
            else:
                same_batch = all(f.shape[0] == len(chunk) for f in self.frame_features_cache)
                same_shape = len({tuple(f.shape) for f in self.frame_features_cache}) == 1
                if same_batch and same_shape:
                    frames = torch.stack(self.frame_features_cache, dim=0).mean(dim=0)
                else:
                    frames = torch.cat(self.frame_features_cache, dim=0)

            frames = self._project_frame_features(frames)
            all_frames.append(frames)

        self.frame_features_cache = []
        frames = torch.cat(all_frames, dim=0)
        return F.normalize(frames, p=2, dim=-1)
