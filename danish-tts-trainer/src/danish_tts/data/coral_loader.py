"""CoRal dataset loader for Danish TTS training."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pyarrow.parquet as pq
import io
import soundfile as sf
import librosa


class CoralDataLoader:
    """Loader for CoRal TTS dataset in Parquet format.

    Dataset schema:
        speaker_id: string
        transcription_id: int64
        text: string
        audio: struct<bytes: binary, path: string>
    """

    def __init__(
        self,
        data_dir: Path,
        sample_rate: int = 24000,
        normalize: bool = True,
    ):
        """Initialize CoRal dataset loader.

        Args:
            data_dir: Directory containing CoRal parquet files
            sample_rate: Target sample rate for audio
            normalize: Whether to normalize audio to [-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.normalize = normalize

        # Find all parquet files - they are in the data/ subdirectory
        data_subdir = self.data_dir / "data"
        if data_subdir.exists():
            self.parquet_files = sorted(data_subdir.glob("train-*.parquet"))
        else:
            # Fallback: look in the root directory
            self.parquet_files = sorted(self.data_dir.glob("train-*.parquet"))

        if len(self.parquet_files) == 0:
            raise ValueError(f"No parquet files found in {data_dir}")

        # Build index mapping global index to (file_idx, row_idx)
        self._build_index()

    def _build_index(self):
        """Build index mapping global row index to (file_idx, local_row_idx)."""
        self.index = []
        self.total_rows = 0
        self.tables = []  # Cache Parquet tables in memory

        for file_idx, parquet_file in enumerate(self.parquet_files):
            table = pq.read_table(parquet_file)
            num_rows = len(table)

            for row_idx in range(num_rows):
                self.index.append((file_idx, row_idx))

            self.total_rows += num_rows
            self.tables.append(table)  # Store table in memory

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.total_rows

    def __getitem__(self, idx: int) -> Dict:
        """Get single dataset item.

        Args:
            idx: Global dataset index

        Returns:
            Dictionary with keys:
                - text: str
                - audio: np.ndarray (float32, normalized)
                - speaker_id: str
                - transcription_id: int
        """
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(f"Index {idx} out of range [0, {self.total_rows})")

        # Get file and row indices
        file_idx, row_idx = self.index[idx]

        # Use cached table instead of reading from disk
        table = self.tables[file_idx]
        row = table.slice(row_idx, 1).to_pydict()

        # Extract fields
        text = row["text"][0]
        speaker_id = row["speaker_id"][0]
        transcription_id = row["transcription_id"][0]
        audio_bytes = row["audio"][0]["bytes"]

        # Decode audio from bytes
        audio, sr = sf.read(io.BytesIO(audio_bytes))

        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
            )

        # Normalize to [-1, 1]
        if self.normalize:
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak

        # Convert to float32
        audio = audio.astype(np.float32)

        return {
            "text": text,
            "audio": audio,
            "speaker_id": speaker_id,
            "transcription_id": transcription_id,
        }
