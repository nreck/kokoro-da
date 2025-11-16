"""Audio preprocessing utilities for Danish TTS."""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


TARGET_SAMPLE_RATE = 24000  # 24 kHz for Kokoro compatibility


def normalize_audio(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """
    Normalize audio to target peak dBFS.

    Args:
        audio: Input audio array
        target_db: Target peak level in dBFS (default: -1.0)

    Returns:
        Normalized audio array
    """
    # Handle zero audio
    peak = np.abs(audio).max()
    if peak == 0:
        return audio

    # Calculate target peak amplitude
    target_peak = 10 ** (target_db / 20)

    # Normalize
    scale = target_peak / peak
    return audio * scale


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: int = 40,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        top_db: Threshold in dB below peak to consider as silence
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation

    Returns:
        Trimmed audio array
    """
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return trimmed


def load_and_resample_audio(
    filepath: Path | str,
    target_sr: int = TARGET_SAMPLE_RATE,
    normalize: bool = True,
    trim: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        filepath: Path to audio file
        target_sr: Target sample rate (default: 24000)
        normalize: Whether to normalize audio (default: True)
        trim: Whether to trim silence (default: True)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Load audio (librosa automatically converts to mono)
    audio, sr = librosa.load(filepath, sr=target_sr, mono=True)

    # Trim silence if requested
    if trim:
        audio = trim_silence(audio, sr)

    # Normalize if requested
    if normalize:
        audio = normalize_audio(audio, target_db=-1.0)

    return audio, sr


def save_audio(
    audio: np.ndarray,
    filepath: Path | str,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> None:
    """
    Save audio to file.

    Args:
        audio: Audio array to save
        filepath: Output file path
        sample_rate: Sample rate of audio
    """
    sf.write(filepath, audio, sample_rate, subtype='PCM_16')
