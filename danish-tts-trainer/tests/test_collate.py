"""Tests for batch collation."""

import pytest
import torch
from danish_tts.data.collate import collate_fn


def test_collate_pads_sequences():
    """Test collate_fn pads variable length sequences."""
    # Create batch with different sequence lengths
    batch = [
        {
            "phoneme_ids": torch.LongTensor([1, 2, 3]),
            "audio": torch.randn(1000),
            "text": "test 1",
            "speaker_id": 0,
        },
        {
            "phoneme_ids": torch.LongTensor([4, 5]),
            "audio": torch.randn(800),
            "text": "test 2",
            "speaker_id": 1,
        },
    ]

    collated = collate_fn(batch)

    # Check shapes
    assert collated["phoneme_ids"].shape == (2, 3)  # Padded to max length
    assert collated["audio"].shape == (2, 1000)  # Padded to max audio length
    assert collated["speaker_ids"].shape == (2,)

    # Check padding worked
    assert collated["phoneme_ids"][1, 2] == 0  # Padding token
    assert collated["phoneme_lengths"][0] == 3
    assert collated["phoneme_lengths"][1] == 2


def test_collate_returns_lengths():
    """Test collate_fn returns sequence lengths."""
    batch = [
        {
            "phoneme_ids": torch.LongTensor([1, 2, 3, 4]),
            "audio": torch.randn(1200),
            "text": "test",
            "speaker_id": 0,
        },
    ]

    collated = collate_fn(batch)

    assert "phoneme_lengths" in collated
    assert "audio_lengths" in collated
    assert collated["phoneme_lengths"][0] == 4
    assert collated["audio_lengths"][0] == 1200
