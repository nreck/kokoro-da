"""Tests for TTS loss functions."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from danish_tts.losses import ReconstructionLoss, KLDivergenceLoss


def test_reconstruction_loss():
    """Test reconstruction loss computes correctly."""
    loss_fn = ReconstructionLoss()

    predicted = torch.randn(2, 80, 100)  # [batch, n_mels, time]
    target = torch.randn(2, 80, 100)

    loss = loss_fn(predicted, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert loss > 0


def test_kl_divergence_loss():
    """Test KL divergence loss."""
    loss_fn = KLDivergenceLoss()

    mean = torch.randn(2, 256)
    log_var = torch.randn(2, 256)

    loss = loss_fn(mean, log_var)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
