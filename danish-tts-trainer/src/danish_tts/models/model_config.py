"""Model configuration and initialization for StyleTTS2."""

import torch
import torch.nn as nn
from typing import Dict
from danish_tts.models.models import TextEncoder, StyleEncoder, ProsodyPredictor
from danish_tts.models.Modules.istftnet import Decoder as iSTFTNetDecoder


def build_model(config: Dict) -> nn.Module:
    """Build StyleTTS2 model from config.

    Args:
        config: Configuration dictionary from YAML

    Returns:
        StyleTTS2 model instance
    """
    model_config = config["model"]

    # Build model using actual StyleTTS2 components
    model = StyleTTS2Model(
        n_symbols=model_config["n_symbols"],
        n_speakers=model_config.get("n_speakers", 1),
        text_encoder_config=model_config.get("text_encoder", {}),
        style_encoder_config=model_config.get("style_encoder", {}),
        decoder_config=model_config.get("decoder", {}),
        vocoder_config=model_config.get("vocoder", {}),
    )

    return model


class StyleTTS2Model(nn.Module):
    """StyleTTS2 model with real components."""

    def __init__(
        self,
        n_symbols: int,
        n_speakers: int,
        text_encoder_config: Dict,
        style_encoder_config: Dict,
        decoder_config: Dict,
        vocoder_config: Dict,
    ):
        super().__init__()

        # Real text encoder from StyleTTS2
        channels = text_encoder_config.get("channels", 256)
        kernel_size = text_encoder_config.get("kernel_size", 5)
        depth = text_encoder_config.get("depth", 3)

        self.text_encoder = TextEncoder(
            channels=channels,
            kernel_size=kernel_size,
            depth=depth,
            n_symbols=n_symbols,
        )

        # Store attributes for testing
        self.text_encoder.channels = channels
        self.text_encoder.kernel_size = kernel_size
        self.text_encoder.depth = depth

        # Real style encoder from StyleTTS2
        self.style_encoder = StyleEncoder(
            dim_in=80,  # Mel spectrogram dimension
            style_dim=style_encoder_config.get("dim", 256),
            max_conv_dim=text_encoder_config.get("channels", 256),
        )

        # Prosody predictor (duration + pitch)
        self.predictor = ProsodyPredictor(
            style_dim=style_encoder_config.get("dim", 256),
            d_hid=text_encoder_config.get("channels", 256),
            nlayers=text_encoder_config.get("depth", 3),
            max_dur=50,
            dropout=text_encoder_config.get("dropout", 0.1),
        )

        # Real iSTFTNet decoder (produces audio directly)
        # Note: iSTFTNet decoder expects specific channel sizes
        # F0_channel should match the text encoder output channels for asr_res layer
        decoder_cfg = decoder_config
        text_channels = text_encoder_config.get("channels", 256)
        self.decoder = iSTFTNetDecoder(
            dim_in=text_channels,
            F0_channel=text_channels,  # Should match dim_in for asr_res layer
            style_dim=style_encoder_config.get("dim", 256),
            dim_out=80,  # Not used for iSTFTNet, goes straight to audio
            resblock_kernel_sizes=decoder_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
            upsample_rates=decoder_cfg.get("upsample_rates", [10, 5, 3, 2]),
            upsample_initial_channel=decoder_cfg.get("upsample_initial_channel", 512),
            resblock_dilation_sizes=decoder_cfg.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
            upsample_kernel_sizes=decoder_cfg.get("upsample_kernel_sizes", [20, 10, 6, 4]),
            gen_istft_n_fft=decoder_cfg.get("gen_istft_n_fft", 2048),
            gen_istft_hop_size=decoder_cfg.get("gen_istft_hop_size", 300),
        )

        # Remove vocoder - iSTFTNet goes directly to audio
        self.vocoder = None

        self.n_speakers = n_speakers
        if n_speakers > 1:
            self.speaker_embedding = nn.Embedding(n_speakers, 256)

    def forward(self, phoneme_ids, speaker_ids=None, ref_audio=None, phoneme_lengths=None):
        """Training forward pass.

        Args:
            phoneme_ids: [batch, seq_len]
            speaker_ids: [batch]
            ref_audio: [batch, audio_len] - reference audio for style
            phoneme_lengths: [batch] - actual lengths

        Returns:
            Dictionary with model outputs
        """
        batch_size, seq_len = phoneme_ids.shape

        # Create input lengths and mask for TextEncoder
        if phoneme_lengths is None:
            phoneme_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=phoneme_ids.device)

        # Create mask (True where padded)
        mask = torch.arange(seq_len, device=phoneme_ids.device)[None, :] >= phoneme_lengths[:, None]

        # Text encoding
        text_enc = self.text_encoder(phoneme_ids, phoneme_lengths, mask)  # [batch, channels, seq_len]

        # Extract style from reference audio (placeholder mel extraction)
        if ref_audio is not None:
            # TODO: Extract mel spectrogram from audio
            # For now, create dummy mel
            mel = torch.zeros(batch_size, 1, 80, seq_len, device=phoneme_ids.device)
            style = self.style_encoder(mel)  # [batch, style_dim]
        else:
            style = torch.zeros(batch_size, 256, device=phoneme_ids.device)

        # Add speaker embedding if multi-speaker
        if self.n_speakers > 1 and speaker_ids is not None:
            speaker_emb = self.speaker_embedding(speaker_ids)  # [batch, 256]
            style = style + speaker_emb

        # Create dummy alignment for predictor (will be replaced with real alignment)
        alignment = torch.eye(seq_len, device=phoneme_ids.device).unsqueeze(0).expand(batch_size, -1, -1)

        # Duration prediction
        durations, pitch = self.predictor(text_enc, style, phoneme_lengths, alignment, mask)

        # Prepare inputs for iSTFTNet decoder
        # The decoder's F0_conv and N_conv have stride=2, which downsamples by 2
        # So we need to:
        # - F0_curve: [batch, seq_len * 2] -> after stride=2 conv -> [batch, seq_len]
        # - N: [batch, seq_len * 2] -> after stride=2 conv -> [batch, seq_len]
        # - asr: [batch, channels, seq_len] - must match the downsampled F0/N

        # Downsample text_enc by 2 to match F0/N after their stride=2 conv
        # Using average pooling with kernel_size=2, stride=2
        import torch.nn.functional as F
        text_enc_downsampled = F.avg_pool1d(text_enc, kernel_size=2, stride=2)  # [batch, channels, seq_len//2]
        downsampled_seq_len = text_enc_downsampled.shape[2]

        # For now, use dummy F0 and N at 2x the downsampled length (will be downsampled by decoder)
        F0_curve = torch.zeros(batch_size, downsampled_seq_len * 2, device=phoneme_ids.device)
        N = torch.ones(batch_size, downsampled_seq_len * 2, device=phoneme_ids.device) * 0.003  # Small noise

        # Decoder produces audio waveform
        # Note: decoder expects style as [batch, style_dim], not [batch, style_dim, 1]
        predicted_audio = self.decoder(text_enc_downsampled, F0_curve, N, style)  # [batch, audio_samples]

        return {
            "predicted_audio": predicted_audio.squeeze(1) if predicted_audio.ndim == 3 else predicted_audio,
            "predicted_mel": torch.zeros(batch_size, 80, seq_len, device=phoneme_ids.device),  # Placeholder for compatibility
            "target_mel": torch.zeros(batch_size, 80, seq_len, device=phoneme_ids.device),  # Placeholder
            "style_mean": torch.zeros(batch_size, 256, device=phoneme_ids.device),
            "style_log_var": torch.zeros(batch_size, 256, device=phoneme_ids.device),
            "durations": durations,
            "pitch": pitch,
        }

    def inference(self, phoneme_ids, speaker_ids=None, temperature=0.667, length_scale=1.0):
        """Inference forward pass.

        Args:
            phoneme_ids: [batch, seq_len]
            speaker_ids: [batch]
            temperature: Sampling temperature for style (default: 0.667)
            length_scale: Duration scaling factor (>1 = slower, <1 = faster)

        Returns:
            Dictionary with audio
        """
        batch_size, seq_len = phoneme_ids.shape

        # Create input lengths and mask
        phoneme_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=phoneme_ids.device)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=phoneme_ids.device)

        # Text encoding
        text_enc = self.text_encoder(phoneme_ids, phoneme_lengths, mask)

        # Sample style from prior with temperature (not reference audio)
        style = torch.randn(batch_size, 256, device=phoneme_ids.device) * temperature

        # Add speaker embedding
        if self.n_speakers > 1 and speaker_ids is not None:
            speaker_emb = self.speaker_embedding(speaker_ids)
            style = style + speaker_emb

        # Create dummy alignment
        alignment = torch.eye(seq_len, device=phoneme_ids.device).unsqueeze(0).expand(batch_size, -1, -1)

        # Duration prediction
        durations, pitch = self.predictor(text_enc, style, phoneme_lengths, alignment, mask)

        # Apply length_scale to durations
        durations = durations * length_scale

        # Prepare inputs for iSTFTNet decoder (same as forward())
        import torch.nn.functional as F
        text_enc_downsampled = F.avg_pool1d(text_enc, kernel_size=2, stride=2)
        downsampled_seq_len = text_enc_downsampled.shape[2]

        # For inference, use dummy F0 and N (will be replaced with real values)
        F0_curve = torch.zeros(batch_size, downsampled_seq_len * 2, device=phoneme_ids.device)
        N = torch.ones(batch_size, downsampled_seq_len * 2, device=phoneme_ids.device) * 0.003

        # Decoder produces audio waveform
        audio = self.decoder(text_enc_downsampled, F0_curve, N, style)
        audio = audio.squeeze(1) if audio.ndim == 3 else audio

        return {"audio": audio}

    def optimize_for_inference(self):
        """Optimize model for faster inference.

        Returns:
            Self (for chaining)
        """
        self.eval()

        # Set all submodules to eval mode recursively
        for module in self.modules():
            module.eval()

        # Disable gradient computation for all parameters
        for param in self.parameters():
            param.requires_grad = False

        return self
