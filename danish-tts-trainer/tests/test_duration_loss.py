"""Tests for duration prediction loss."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from danish_tts.losses import DurationLoss


def test_duration_loss():
    """Test duration loss computes correctly."""
    loss_fn = DurationLoss()

    # Predicted and target durations
    predicted = torch.randn(2, 10)  # [batch, seq_len] (log scale)
    target = torch.randint(1, 20, (2, 10)).float()  # [batch, seq_len]
    lengths = torch.LongTensor([8, 10])  # Actual sequence lengths

    loss = loss_fn(predicted, target, lengths)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert loss > 0
