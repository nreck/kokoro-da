"""Tests for real StyleTTS2 text encoder."""

import pytest
import torch
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from danish_tts.models.model_config import build_model


def test_text_encoder_architecture():
    """Test text encoder has correct StyleTTS2 architecture."""
    config_path = Path("configs/coral_danish.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = build_model(config)

    # Check text encoder is the real TextEncoder class
    from danish_tts.models.models import TextEncoder
    assert isinstance(model.text_encoder, TextEncoder)

    # Check architecture params
    assert model.text_encoder.channels == 256
    assert model.text_encoder.kernel_size == 5
    assert model.text_encoder.depth == 3


def test_text_encoder_forward():
    """Test text encoder forward pass with correct output shape."""
    config_path = Path("configs/coral_danish.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = build_model(config)

    # Create input
    batch_size = 2
    seq_len = 10
    phoneme_ids = torch.randint(0, 42, (batch_size, seq_len))
    phoneme_lengths = torch.LongTensor([seq_len, seq_len])
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    # Forward pass
    with torch.no_grad():
        output = model.text_encoder(phoneme_ids, phoneme_lengths, mask)

    # Check output shape: [batch, channels, seq_len]
    # Note: TextEncoder returns [batch, channels, seq_len] not [batch, seq_len, channels]
    assert output.shape == (batch_size, 256, seq_len)
    assert output.dtype == torch.float32
