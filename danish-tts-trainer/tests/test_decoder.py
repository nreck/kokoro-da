"""Tests for iSTFTNet decoder integration."""

import pytest
import torch
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from danish_tts.models.model_config import build_model


def test_decoder_is_istftnet():
    """Test decoder uses iSTFTNet from StyleTTS2."""
    config_path = Path("configs/coral_danish.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = build_model(config)

    # Check decoder is from Modules.istftnet
    assert hasattr(model, 'decoder')
    assert model.decoder.__class__.__name__ == 'Decoder'


def test_decoder_output_is_audio():
    """Test decoder produces audio waveform output through full model forward."""
    config_path = Path("configs/coral_danish.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = build_model(config)
    model.eval()

    # Test through the full model forward pass instead of decoder directly
    batch_size = 2
    # Use seq_len that works well with upsample rates after downsampling
    # seq_len=20 -> after avg_pool(2) -> 10 -> upsample[10,5,3,2] -> 300 samples
    seq_len = 20
    phoneme_ids = torch.randint(0, 42, (batch_size, seq_len))
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)

    # Forward pass through full model
    with torch.no_grad():
        outputs = model(phoneme_ids, speaker_ids=speaker_ids, phoneme_lengths=None)

    # Check that model produces audio output (not just mel)
    assert "predicted_audio" in outputs
    audio = outputs["predicted_audio"]

    # iSTFTNet produces audio directly, not mel
    assert audio.ndim in [2, 3]  # [batch, audio_samples] or [batch, 1, audio_samples]
    if audio.ndim == 3:
        assert audio.shape[1] == 1  # Single channel
        audio = audio.squeeze(1)
    assert audio.shape[0] == batch_size
    assert audio.shape[1] > 0  # Has audio samples
    assert audio.dtype == torch.float32
