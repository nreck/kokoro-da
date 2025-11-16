import pytest
import numpy as np
import torch
from danish_tts.audio_utils import (
    load_and_resample_audio,
    normalize_audio,
    trim_silence,
    TARGET_SAMPLE_RATE,
)


def test_target_sample_rate():
    """Test that target sample rate is 24kHz."""
    assert TARGET_SAMPLE_RATE == 24000


def test_normalize_audio():
    """Test audio normalization to -1 dBFS peak."""
    # Create test signal with known peak
    audio = np.array([0.0, 0.5, -0.5, 0.25, -0.25])
    normalized = normalize_audio(audio, target_db=-1.0)

    # Peak should be close to 10^(-1/20) ≈ 0.891
    expected_peak = 10 ** (-1.0 / 20)
    actual_peak = np.abs(normalized).max()
    assert abs(actual_peak - expected_peak) < 0.01


def test_normalize_audio_zero():
    """Test that zero audio stays zero."""
    audio = np.zeros(100)
    normalized = normalize_audio(audio)
    assert np.allclose(normalized, 0)
