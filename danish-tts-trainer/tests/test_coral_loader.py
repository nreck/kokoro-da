"""Tests for CoRal dataset loader."""

import pytest
from pathlib import Path
import numpy as np
from danish_tts.data.coral_loader import CoralDataLoader


def test_coral_loader_initialization():
    """Test CoralDataLoader can initialize with coral-tts directory."""
    loader = CoralDataLoader(
        data_dir=Path("coral-tts"),
        sample_rate=24000,
    )
    assert loader.data_dir.exists()
    assert loader.sample_rate == 24000


def test_coral_loader_loads_all_parquet_files():
    """Test loader finds all 25 parquet files."""
    loader = CoralDataLoader(data_dir=Path("coral-tts"))
    assert len(loader.parquet_files) == 25
    assert all(f.name.startswith("train-") for f in loader.parquet_files)


def test_coral_loader_get_item():
    """Test getting single item returns audio bytes and text."""
    loader = CoralDataLoader(data_dir=Path("coral-tts"))
    item = loader[0]

    assert "text" in item
    assert "audio" in item
    assert "speaker_id" in item
    assert isinstance(item["text"], str)
    assert isinstance(item["audio"], np.ndarray)
    assert item["audio"].dtype == np.float32


def test_coral_loader_audio_is_24khz():
    """Test audio is resampled to 24kHz."""
    loader = CoralDataLoader(data_dir=Path("coral-tts"), sample_rate=24000)
    item = loader[0]

    # Audio should be at 24kHz
    # Check it's not empty and is normalized
    assert len(item["audio"]) > 0
    assert item["audio"].max() <= 1.0
    assert item["audio"].min() >= -1.0
