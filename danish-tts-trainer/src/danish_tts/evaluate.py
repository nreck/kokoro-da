"""Evaluation script for Danish TTS model."""

import argparse
from pathlib import Path
from typing import List
import torch
import soundfile as sf
from tqdm import tqdm

from danish_tts.inference import load_model, synthesize
from danish_tts.g2p_da import G2P


def load_test_sentences(filepath: Path) -> List[str]:
    """
    Load test sentences from file.

    Args:
        filepath: Path to test sentences file

    Returns:
        List of test sentences (excluding comments and empty lines)
    """
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                sentences.append(line)
    return sentences


def evaluate_model(
    model,
    g2p: G2P,
    test_sentences: List[str],
    output_dir: Path,
    speaker_id: int = 0,
    device: str = "cpu",
) -> None:
    """
    Evaluate model on test sentences.

    Args:
        model: Trained TTS model
        g2p: Danish G2P instance
        test_sentences: List of test sentences
        output_dir: Directory to save generated audio
        speaker_id: Speaker ID
        device: Device to run on
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating on {len(test_sentences)} test sentences...")

    for idx, sentence in enumerate(tqdm(test_sentences)):
        try:
            # Synthesize
            audio = synthesize(
                model,
                sentence,
                g2p,
                speaker_id=speaker_id,
                device=device,
            )

            # Save audio
            output_path = output_dir / f"test_{idx:03d}.wav"
            sf.write(
                output_path,
                audio.cpu().numpy().squeeze(),
                24000,
            )

            # Save transcript
            transcript_path = output_dir / f"test_{idx:03d}.txt"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(sentence)

        except Exception as e:
            print(f"Error synthesizing '{sentence}': {e}")


def main():
    """CLI for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Danish TTS Model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test_sentences",
        type=Path,
        default=Path("test_sentences.txt"),
        help="Path to test sentences file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("evaluation_outputs"),
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=0,
        choices=[0, 1],
        help="Speaker ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    args = parser.parse_args()

    # Load test sentences
    print(f"Loading test sentences from {args.test_sentences}...")
    sentences = load_test_sentences(args.test_sentences)
    print(f"Loaded {len(sentences)} sentences")

    # Initialize G2P
    print("Loading Danish G2P...")
    g2p = G2P()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)

    # Evaluate
    evaluate_model(
        model,
        g2p,
        sentences,
        args.output_dir,
        speaker_id=args.speaker,
        device=args.device,
    )

    print(f"Evaluation complete! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
