"""PyTorch Dataset for Danish TTS training."""

from pathlib import Path
from typing import Dict, Optional
import torch
from torch.utils.data import Dataset
import numpy as np

from danish_tts.data.coral_loader import CoralDataLoader


class TTSDataset(Dataset):
    """PyTorch Dataset for TTS training.

    Wraps CoralDataLoader and adds:
    - G2P conversion (text -> phoneme IDs)
    - Speaker ID mapping (string -> int)
    - PyTorch tensor conversion
    """

    def __init__(
        self,
        data_dir: Path,
        g2p,
        sample_rate: int = 24000,
        normalize: bool = True,
    ):
        """Initialize TTS Dataset.

        Args:
            data_dir: Directory containing CoRal parquet files
            g2p: Danish G2P instance (misaki.da.G2P)
            sample_rate: Target sample rate for audio
            normalize: Whether to normalize audio
        """
        self.data_dir = Path(data_dir)
        self.g2p = g2p
        self.sample_rate = sample_rate

        # Initialize CoRal loader
        self.loader = CoralDataLoader(
            data_dir=data_dir,
            sample_rate=sample_rate,
            normalize=normalize,
        )

        # Build speaker mapping
        self._build_speaker_mapping()

    def _build_speaker_mapping(self):
        """Build mapping from speaker_id string to integer."""
        speakers = set()
        # Query parquet tables directly instead of loading all samples
        for table in self.loader.tables:
            speakers.update(table.column('speaker_id').to_pylist())

        # Create mapping
        self.speaker_to_id = {
            speaker: idx for idx, speaker in enumerate(sorted(speakers))
        }
        self.id_to_speaker = {
            idx: speaker for speaker, idx in self.speaker_to_id.items()
        }

        print(f"Found {len(speakers)} unique speakers: {sorted(speakers)}")

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.loader)

    def __getitem__(self, idx: int) -> Dict:
        """Get single training sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - phoneme_ids: torch.Tensor [seq_len] (long)
                - audio: torch.Tensor [num_samples] (float32)
                - text: str (original text)
                - speaker_id: int
        """
        # Load from CoRal
        item = self.loader[idx]

        # Convert text to phoneme IDs
        text = item["text"]
        phonemes, phoneme_ids = self.g2p(text)

        # Convert speaker to int
        speaker_str = item["speaker_id"]
        speaker_id = self.speaker_to_id[speaker_str]

        # Convert to tensors
        phoneme_ids_tensor = torch.LongTensor(phoneme_ids)
        audio_tensor = torch.FloatTensor(item["audio"])

        return {
            "phoneme_ids": phoneme_ids_tensor,
            "audio": audio_tensor,
            "text": text,
            "speaker_id": speaker_id,
            "phonemes": phonemes,  # For debugging
        }
