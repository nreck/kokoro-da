"""Danish TTS inference script."""

import argparse
from pathlib import Path
import torch
import soundfile as sf
import sys

from danish_tts.g2p_da import G2P


@torch.no_grad()
def synthesize_optimized(
    model,
    text: str,
    g2p,
    speaker_id: int = 0,
    device: str = "cpu",
    temperature: float = 0.667,
    length_scale: float = 1.0,
) -> torch.Tensor:
    """Optimized synthesis with temperature and length control.

    Args:
        model: Trained TTS model
        text: Input Danish text
        g2p: Danish G2P instance
        speaker_id: Speaker ID (0 or 1)
        device: Device to run inference on
        temperature: Sampling temperature for style
        length_scale: Duration scaling factor (>1 = slower, <1 = faster)

    Returns:
        Audio waveform tensor [1, n_samples]
    """
    model.eval()

    # Get phonemes
    phonemes, tokens = g2p(text)

    # Convert to tensors
    phoneme_ids = torch.LongTensor(tokens).unsqueeze(0).to(device)
    speaker_ids = torch.LongTensor([speaker_id]).to(device) if model.n_speakers > 1 else None

    # Inference with optimizations
    outputs = model.inference(
        phoneme_ids=phoneme_ids,
        speaker_ids=speaker_ids,
        temperature=temperature,
        length_scale=length_scale,
    )

    return outputs["audio"]


@torch.no_grad()
def synthesize_batch(
    model,
    texts: list,
    g2p,
    speaker_id: int = 0,
    device: str = "cpu",
    batch_size: int = 8,
) -> list:
    """Batch synthesis for multiple texts.

    Args:
        model: Trained TTS model
        texts: List of Danish texts
        g2p: Danish G2P instance
        speaker_id: Speaker ID
        device: Device
        batch_size: Batch size for processing

    Returns:
        List of audio waveform tensors
    """
    model.eval()
    all_audio = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Convert all texts to phonemes
        batch_phonemes = []
        batch_lengths = []
        for text in batch_texts:
            phonemes, tokens = g2p(text)
            batch_phonemes.append(torch.LongTensor(tokens))
            batch_lengths.append(len(tokens))

        # Pad to max length in batch
        max_len = max(batch_lengths)
        padded = torch.zeros(len(batch_texts), max_len, dtype=torch.long)
        for j, phonemes in enumerate(batch_phonemes):
            padded[j, :len(phonemes)] = phonemes

        # Batch inference
        padded = padded.to(device)
        speaker_ids_batch = torch.LongTensor([speaker_id] * len(batch_texts)).to(device) if model.n_speakers > 1 else None

        outputs = model.inference(
            phoneme_ids=padded,
            speaker_ids=speaker_ids_batch,
        )

        # Split batch back into individual audio
        audio_batch = outputs["audio"]
        for audio in audio_batch:
            all_audio.append(audio)

    return all_audio


def load_model(checkpoint_path: Path, device: str = "cpu"):
    """Load trained Danish TTS model.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    import yaml
    from danish_tts.models.model_config import build_model

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint["config"]

    # Build model
    model = build_model(config)

    # Load state dict
    model.load_state_dict(checkpoint["model"])

    # Set to eval mode
    model.eval()
    model = model.to(device)

    print(f"Loaded model from step {checkpoint['step']}")

    return model


def synthesize(
    model,
    text: str,
    g2p: G2P,
    speaker_id: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """Synthesize speech from text.

    Args:
        model: Trained TTS model
        text: Input Danish text
        g2p: Danish G2P instance
        speaker_id: Speaker ID (0 or 1)
        device: Device to run inference on

    Returns:
        Audio waveform tensor (shape: [1, n_samples])
    """
    model.eval()

    # Get phonemes
    phonemes, tokens = g2p(text)
    print(f"Text: {text}")
    print(f"Phonemes: {phonemes}")

    # Convert to tensors
    phoneme_ids = torch.LongTensor(tokens).unsqueeze(0).to(device)  # [1, seq_len]
    speaker_ids = torch.LongTensor([speaker_id]).to(device)  # [1]

    # Inference
    with torch.no_grad():
        outputs = model.inference(
            phoneme_ids=phoneme_ids,
            speaker_ids=speaker_ids,
        )

        # Extract audio
        audio = outputs["audio"]  # [1, n_samples]

    return audio


def main():
    """CLI for Danish TTS inference."""
    parser = argparse.ArgumentParser(description="Danish TTS Inference")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Danish text to synthesize",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=0,
        choices=[0, 1],
        help="Speaker ID (0=female, 1=male)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output audio file path (.wav)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    args = parser.parse_args()

    # Initialize G2P
    print("Loading Danish G2P...")
    g2p = G2P()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)

    # Synthesize
    print(f"Synthesizing: {args.text}")
    audio = synthesize(
        model,
        args.text,
        g2p,
        speaker_id=args.speaker,
        device=args.device,
    )

    # Save audio
    print(f"Saving to {args.output}")
    sf.write(args.output, audio.cpu().numpy().squeeze(), 24000)
    print("Done!")


if __name__ == "__main__":
    main()
