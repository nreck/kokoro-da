"""Discriminator models for adversarial training."""

import torch
import torch.nn as nn
from danish_tts.models.Modules.discriminators import (
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
    WavLMDiscriminator,
)


class MultiDiscriminator(nn.Module):
    """Multi-scale discriminator combining MPD, MSD, and WavLM."""

    def __init__(self, use_wavlm=True, wavlm_hidden=768, wavlm_nlayers=13, wavlm_initial_channel=64):
        super().__init__()

        # Multi-Period Discriminator
        self.mpd = MultiPeriodDiscriminator()

        # Multi-Resolution Spectrogram Discriminator
        self.msd = MultiResSpecDiscriminator()

        # WavLM-based discriminator (optional)
        self.use_wavlm = use_wavlm
        if use_wavlm:
            self.wd = WavLMDiscriminator(
                slm_hidden=wavlm_hidden,
                slm_layers=wavlm_nlayers,
                initial_channel=wavlm_initial_channel,
            )

    def forward(self, audio):
        """Run all discriminators on audio.

        Args:
            audio: [batch, audio_samples] or [batch, 1, audio_samples]

        Returns:
            List of logits from each discriminator group (MPD, MSD, WavLM)
            Each element is a list of logits from sub-discriminators
        """
        # Ensure audio has channel dimension for discriminators
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [batch, 1, audio_samples]

        discriminator_outputs = []

        # MPD - Multi-Period Discriminator (5 periods)
        # Returns (y_d_rs, y_d_gs, fmap_rs, fmap_gs)
        mpd_logits, _, _, _ = self.mpd(audio, audio)
        discriminator_outputs.append(mpd_logits)

        # MSD - Multi-Resolution Spectrogram Discriminator (3 resolutions)
        msd_logits, _, _, _ = self.msd(audio, audio)
        discriminator_outputs.append(msd_logits)

        # WavLM - WavLM-based discriminator
        if self.use_wavlm:
            # WavLM expects concatenated features from all layers
            # For now, we'll create dummy features since we don't have WavLM model here
            # In actual training, you'd extract WavLM features first
            # This is a placeholder - actual implementation would need WavLM features
            batch_size = audio.shape[0]
            seq_len = audio.shape[2] // 320  # Approximate feature length
            dummy_features = torch.randn(
                batch_size,
                self.wd.pre.in_channels,  # wavlm_hidden * wavlm_nlayers
                seq_len,
                device=audio.device
            )
            wd_logit = self.wd(dummy_features)
            # Wrap in list to maintain consistent structure
            discriminator_outputs.append([wd_logit])

        return discriminator_outputs
