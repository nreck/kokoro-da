"""Preprocess CoRal TTS dataset for StyleTTS2 training."""

import json
import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import librosa

from danish_tts.audio_utils import load_and_resample_audio, save_audio


def create_manifest_entry(
    audio_path: str,
    text: str,
    phonemes: str,
    speaker_id: int,
    duration: float,
    lang_id: int = 0,
) -> dict:
    """
    Create a manifest entry for training.

    Args:
        audio_path: Path to preprocessed audio file
        text: Original text transcript
        phonemes: Phoneme sequence (space-separated)
        speaker_id: Speaker ID (0 or 1 for CoRal)
        duration: Audio duration in seconds
        lang_id: Language ID (0 for Danish)

    Returns:
        Dictionary with manifest entry
    """
    return {
        "audio_path": audio_path,
        "text": text,
        "phonemes": phonemes,
        "speaker_id": speaker_id,
        "lang_id": lang_id,
        "duration": duration,
    }


def parse_coral_metadata(metadata_file: Path) -> list[dict]:
    """
    Parse CoRal dataset metadata.

    Args:
        metadata_file: Path to metadata CSV/JSON file

    Returns:
        List of metadata entries
    """
    # TODO: Implement based on actual CoRal metadata format
    # This is a placeholder
    raise NotImplementedError("Implement based on CoRal metadata format")


def preprocess_utterance(
    audio_path: Path,
    text: str,
    output_path: Path,
    g2p,
) -> Optional[dict]:
    """
    Preprocess single utterance.

    Args:
        audio_path: Path to input audio
        text: Text transcript
        output_path: Path to save preprocessed audio
        g2p: G2P instance for phonemization

    Returns:
        Manifest entry or None if preprocessing failed
    """
    try:
        # Load and preprocess audio
        audio, sr = load_and_resample_audio(
            audio_path,
            normalize=True,
            trim=True,
        )

        # Get duration
        duration = len(audio) / sr

        # Skip very short or very long utterances
        if duration < 0.5 or duration > 15.0:
            return None

        # Save preprocessed audio
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_audio(audio, output_path, sr)

        # Get phonemes
        phonemes, _ = g2p(text)

        return {
            "audio": audio,
            "duration": duration,
            "phonemes": phonemes,
        }

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def main():
    """Main preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocess CoRal TTS dataset"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to raw CoRal dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to save preprocessed data",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata file",
    )
    args = parser.parse_args()

    print("Preprocessing CoRal TTS dataset...")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")

    # TODO: Implement full preprocessing pipeline
    # This is a placeholder
    raise NotImplementedError("Implement full preprocessing")


if __name__ == "__main__":
    main()
