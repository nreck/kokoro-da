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
        use_phoneme_cache: bool = False,
    ):
        """Initialize TTS Dataset.

        Args:
            data_dir: Directory containing CoRal parquet files
            g2p: Danish G2P instance (misaki.da.G2P)
            sample_rate: Target sample rate for audio
            normalize: Whether to normalize audio
            use_phoneme_cache: Cache all phonemes at init (for validation)
        """
        self.data_dir = Path(data_dir)
        self.g2p = g2p
        self.sample_rate = sample_rate
        self.use_phoneme_cache = use_phoneme_cache

        # Initialize CoRal loader
        self.loader = CoralDataLoader(
            data_dir=data_dir,
            sample_rate=sample_rate,
            normalize=normalize,
        )

        # Build speaker mapping
        self._build_speaker_mapping()

        # Cache phonemes if requested (for thread safety during validation)
        self.phoneme_cache = {}
        if use_phoneme_cache:
            self._cache_all_phonemes()

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

    def _cache_all_phonemes(self):
        """Pre-compute all phonemes to avoid espeak-ng threading issues."""
        import time
        print("Caching phonemes for validation (espeak-ng thread safety)...")

        # Process in smaller batches with delays to avoid espeak-ng crashes
        batch_size = 100
        for i in range(0, len(self), batch_size):
            end_idx = min(i + batch_size, len(self))

            for idx in range(i, end_idx):
                item = self.loader[idx]
                text = item["text"]
                try:
                    phonemes, phoneme_ids = self.g2p(text)
                    self.phoneme_cache[idx] = (phonemes, phoneme_ids)
                except Exception as e:
                    print(f"Warning: Failed to cache phonemes for idx {idx}: {e}")
                    # Use dummy phonemes as fallback
                    self.phoneme_cache[idx] = ("", [0])

            # Small delay between batches to let espeak-ng reset
            if i + batch_size < len(self):
                time.sleep(0.1)
                print(f"  Cached {min(i + batch_size, len(self))}/{len(self)} samples...")

        print(f"Cached {len(self.phoneme_cache)} phoneme conversions")

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

        # Convert text to phoneme IDs (use cache if available)
        if self.use_phoneme_cache and idx in self.phoneme_cache:
            phonemes, phoneme_ids = self.phoneme_cache[idx]
        else:
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
