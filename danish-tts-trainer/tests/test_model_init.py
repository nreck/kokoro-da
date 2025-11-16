"""Tests for StyleTTS2 model initialization."""

import pytest
import torch
from pathlib import Path
import yaml
from danish_tts.models.model_config import build_model


def test_build_model_from_config():
    """Test building StyleTTS2 model from config."""
    config_path = Path("configs/coral_danish.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = build_model(config)

    # Check model has required components
    assert hasattr(model, 'text_encoder')
    assert hasattr(model, 'style_encoder')
    assert hasattr(model, 'decoder')
    assert hasattr(model, 'vocoder')


def test_model_forward_pass():
    """Test model can do forward pass."""
    config_path = Path("configs/coral_danish.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = build_model(config)
    model.eval()

    # Create dummy inputs
    batch_size = 2
    seq_len = 10

    phoneme_ids = torch.randint(0, 42, (batch_size, seq_len))
    speaker_ids = torch.randint(0, 2, (batch_size,))

    # Forward pass (just text encoder for now)
    with torch.no_grad():
        text_enc = model.text_encoder(phoneme_ids)

    assert text_enc.shape[0] == batch_size
    assert text_enc.shape[1] == seq_len
