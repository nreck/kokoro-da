"""Tests for TTS PyTorch Dataset."""

import pytest
import torch
from pathlib import Path
from danish_tts.data.tts_dataset import TTSDataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "misaki"))
from misaki.da import G2P


def test_tts_dataset_initialization():
    """Test TTSDataset initializes correctly."""
    g2p = G2P()
    # Path from danish-tts-trainer/tests/test_tts_dataset.py -> kokoro-research-training/coral-tts/data
    coral_data_dir = Path(__file__).parent.parent.parent / "coral-tts" / "data"
    dataset = TTSDataset(
        data_dir=coral_data_dir,
        g2p=g2p,
        sample_rate=24000,
    )
    assert len(dataset) > 0


def test_tts_dataset_getitem_returns_correct_format():
    """Test __getitem__ returns tensors in correct format."""
    g2p = G2P()
    coral_data_dir = Path(__file__).parent.parent.parent / "coral-tts" / "data"
    dataset = TTSDataset(
        data_dir=coral_data_dir,
        g2p=g2p,
        sample_rate=24000,
    )

    item = dataset[0]

    # Check keys
    assert "phoneme_ids" in item
    assert "audio" in item
    assert "text" in item
    assert "speaker_id" in item

    # Check types
    assert isinstance(item["phoneme_ids"], torch.Tensor)
    assert isinstance(item["audio"], torch.Tensor)
    assert isinstance(item["text"], str)
    assert isinstance(item["speaker_id"], int)

    # Check shapes
    assert item["phoneme_ids"].ndim == 1  # [seq_len]
    assert item["audio"].ndim == 1  # [num_samples]
    assert item["phoneme_ids"].dtype == torch.long
    assert item["audio"].dtype == torch.float32


def test_tts_dataset_speaker_mapping():
    """Test speaker IDs are mapped to integers."""
    g2p = G2P()
    coral_data_dir = Path(__file__).parent.parent.parent / "coral-tts" / "data"
    dataset = TTSDataset(data_dir=coral_data_dir, g2p=g2p)

    # Should have speaker mapping
    assert hasattr(dataset, "speaker_to_id")
    assert isinstance(dataset.speaker_to_id, dict)

    # Get item and check speaker_id is int
    item = dataset[0]
    assert isinstance(item["speaker_id"], int)
    assert item["speaker_id"] >= 0
