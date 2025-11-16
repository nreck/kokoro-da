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
            ref_audio: [batch, audio_len] - reference audio for style and reconstruction target
            phoneme_lengths: [batch] - actual lengths

        Returns:
            Dictionary with model outputs (predicted/target mels will be cropped to match)
        """
        batch_size, seq_len = phoneme_ids.shape

        # Create input lengths and mask for TextEncoder
        if phoneme_lengths is None:
            phoneme_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=phoneme_ids.device)

        # Create mask (True where padded)
        mask = torch.arange(seq_len, device=phoneme_ids.device)[None, :] >= phoneme_lengths[:, None]

        # Text encoding
        text_enc = self.text_encoder(phoneme_ids, phoneme_lengths, mask)  # [batch, channels, seq_len]

        # Extract style from reference audio
        if ref_audio is not None:
            # Convert audio to mel spectrogram
            import torchaudio.transforms as T
            mel_transform = T.MelSpectrogram(
                sample_rate=24000,
                n_fft=2048,
                hop_length=300,
                n_mels=80,
            ).to(ref_audio.device)

            # ref_audio: [batch, audio_samples]
            if ref_audio.ndim == 1:
                ref_audio = ref_audio.unsqueeze(0)

            # Ensure minimum audio length for STFT
            # For 80+ mel frames: (n_samples - n_fft) // hop_length + 1 >= 80
            # (n_samples - 2048) // 300 + 1 >= 80
            # n_samples >= 79 * 300 + 2048 = 25748 samples
            min_len = 25748  # Exact minimum for 80 mel frames (saves memory)
            original_len = ref_audio.shape[-1]
            if ref_audio.shape[-1] < min_len:
                pad_len = min_len - ref_audio.shape[-1]
                ref_audio = torch.nn.functional.pad(ref_audio, (0, pad_len), mode='constant', value=0)

            mel = mel_transform(ref_audio)  # [batch, n_mels, time]

            # Debug: check mel dimensions before unsqueeze
            if mel.shape[2] < 80:
                print(f"WARNING: Mel has {mel.shape[2]} frames, need 80+ for StyleEncoder!")
                print(f"  Original audio length: {original_len}")
                print(f"  Padded audio length: {ref_audio.shape[-1]}")
                print(f"  Mel shape before unsqueeze: {mel.shape}")
                print(f"  Expected frames: {(ref_audio.shape[-1] - 2048) // 300 + 1}")

            mel = mel.unsqueeze(1)  # [batch, 1, n_mels, time] for style encoder

            # Ensure minimum size for style encoder (kernel_size=5 requires at least 5x5)
            # mel.shape = [batch, 1, n_mels=80, time]
            # StyleEncoder has 4x ResBlk(downsample='half') that divide BOTH dims by 2
            # After 4 downsamplings: [1, 80, T] -> [dim, 5, T/16]
            # Final Conv2d(5, 1, 0) needs at least 5x5, so T/16 >= 5 -> T >= 80
            min_time = 80  # Minimum required for StyleEncoder architecture
            _, _, n_mels, time_frames = mel.shape
            if time_frames < min_time:
                pad_w = min_time - time_frames
                # Pad: (left, right, top, bottom)
                mel = torch.nn.functional.pad(mel, (0, pad_w, 0, 0), mode='constant', value=0)
                print(f"  Padded mel from {time_frames} to {mel.shape[3]} time frames")

            style = self.style_encoder(mel)  # [batch, style_dim]
        else:
            # Random style if no reference
            style = torch.randn(batch_size, 256, device=phoneme_ids.device)

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

        # Ensure minimum length before downsampling (need at least 2 frames after downsampling)
        min_text_len = 4  # Will become 2 after downsampling
        if text_enc.shape[2] < min_text_len:
            pad_len = min_text_len - text_enc.shape[2]
            text_enc = F.pad(text_enc, (0, pad_len), mode='constant', value=0)

        text_enc_downsampled = F.avg_pool1d(text_enc, kernel_size=2, stride=2)  # [batch, channels, seq_len//2]
        downsampled_seq_len = text_enc_downsampled.shape[2]

        # For now, use dummy F0 and N at 2x the downsampled length (will be downsampled by decoder)
        F0_curve = torch.zeros(batch_size, downsampled_seq_len * 2, device=phoneme_ids.device)
        N = torch.ones(batch_size, downsampled_seq_len * 2, device=phoneme_ids.device) * 0.003  # Small noise

        # Decoder produces audio waveform
        # Note: decoder expects style as [batch, style_dim], not [batch, style_dim, 1]
        predicted_audio = self.decoder(text_enc_downsampled, F0_curve, N, style)  # [batch, audio_samples]

        # Convert predicted audio to mel
        import torchaudio.transforms as T
        mel_transform = T.MelSpectrogram(
            sample_rate=24000,
            n_fft=2048,
            hop_length=300,
            n_mels=80,
        ).to(predicted_audio.device)

        # Compute mels from audio
        if predicted_audio.ndim == 3:
            predicted_audio = predicted_audio.squeeze(1)

        # Ensure minimum audio length for STFT (match ref_audio padding)
        # For 80+ mel frames: (n_samples - n_fft) // hop_length + 1 >= 80
        # n_samples >= 79 * 300 + 2048 = 25748 samples
        min_len = 25748  # Exact minimum for 80 mel frames (saves memory)
        if predicted_audio.shape[-1] < min_len:
            pad_len = min_len - predicted_audio.shape[-1]
            predicted_audio = torch.nn.functional.pad(predicted_audio, (0, pad_len), mode='constant', value=0)

        predicted_mel = mel_transform(predicted_audio)  # [batch, n_mels, time]

        # Ensure minimum mel size for any subsequent processing
        # Match the minimum from style encoder (80 frames for T/16 >= 5)
        min_mel_time = 80
        if predicted_mel.shape[2] < min_mel_time:
            pad_w = min_mel_time - predicted_mel.shape[2]
            predicted_mel = torch.nn.functional.pad(predicted_mel, (0, pad_w, 0, 0), mode='constant', value=0)

        # Target mel from reference audio
        if ref_audio is not None:
            target_mel = mel_transform(ref_audio)

            # Ensure minimum mel size for target as well
            if target_mel.shape[2] < min_mel_time:
                pad_w = min_mel_time - target_mel.shape[2]
                target_mel = torch.nn.functional.pad(target_mel, (0, pad_w, 0, 0), mode='constant', value=0)

            # Match dimensions - crop or pad to match lengths
            pred_time = predicted_mel.shape[2]
            target_time = target_mel.shape[2]

            if pred_time > target_time:
                # Crop predicted to match target
                predicted_mel = predicted_mel[:, :, :target_time]
                predicted_audio = predicted_audio[:, :target_time * 300]  # 300 = hop_length
            elif pred_time < target_time:
                # Crop target to match predicted
                target_mel = target_mel[:, :, :pred_time]
        else:
            target_mel = predicted_mel.detach()

        # For StyleTTS2, style is sampled from N(style_mean, style_log_var)
        # For now, approximate: style_mean ≈ style, style_log_var ≈ 0
        style_mean = style
        style_log_var = torch.zeros_like(style)

        return {
            "predicted_audio": predicted_audio,
            "predicted_mel": predicted_mel,
            "target_mel": target_mel,
            "style_mean": style_mean,
            "style_log_var": style_log_var,
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

        # Ensure minimum length before downsampling (need at least 2 frames after downsampling)
        min_text_len = 4  # Will become 2 after downsampling
        if text_enc.shape[2] < min_text_len:
            pad_len = min_text_len - text_enc.shape[2]
            text_enc = F.pad(text_enc, (0, pad_len), mode='constant', value=0)

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
