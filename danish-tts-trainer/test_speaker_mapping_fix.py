#!/usr/bin/env python3
"""Test script to verify the speaker mapping performance fix."""

import time
import tempfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import soundfile as sf
import io

from src.danish_tts.data.coral_loader import CoralDataLoader


class MockG2P:
    """Mock G2P for testing without espeak dependency."""
    def __call__(self, text):
        # Simple mock: return character-level IDs
        phonemes = list(text)
        phoneme_ids = [ord(c) % 100 for c in text]
        return phonemes, phoneme_ids


def create_dummy_audio_bytes():
    """Create dummy audio bytes for testing."""
    # Generate 0.1 second of silence at 24kHz
    audio = np.zeros(2400, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format='WAV')
    buffer.seek(0)
    return buffer.read()


def create_test_parquet_files(data_dir: Path, num_files: int = 2, rows_per_file: int = 100):
    """Create test parquet files with dummy data."""
    data_dir = Path(data_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    audio_bytes = create_dummy_audio_bytes()

    for file_idx in range(num_files):
        # Create dummy data with multiple speakers
        data = {
            "speaker_id": [f"speaker_{i % 10}" for i in range(rows_per_file)],
            "transcription_id": list(range(rows_per_file)),
            "text": [f"Test text {i}" for i in range(rows_per_file)],
            "audio": [{"bytes": audio_bytes, "path": f"test_{i}.wav"} for i in range(rows_per_file)],
        }

        # Create schema
        schema = pa.schema([
            ("speaker_id", pa.string()),
            ("transcription_id", pa.int64()),
            ("text", pa.string()),
            ("audio", pa.struct([
                ("bytes", pa.binary()),
                ("path", pa.string()),
            ])),
        ])

        # Create table and write to parquet
        table = pa.table(data, schema=schema)
        output_file = data_dir / f"train-{file_idx:05d}.parquet"
        pq.write_table(table, output_file)
        print(f"Created {output_file}")


def test_old_speaker_mapping(loader):
    """Test the old (slow) speaker mapping approach."""
    speakers = set()
    for idx in range(len(loader)):
        item = loader[idx]
        speakers.add(item["speaker_id"])
    return speakers


def test_new_speaker_mapping(loader):
    """Test the new (fast) speaker mapping approach."""
    speakers = set()
    for table in loader.tables:
        speakers.update(table.column('speaker_id').to_pylist())
    return speakers


def test_performance():
    """Test the performance improvement of the speaker mapping fix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Creating test dataset...")
        # Create a dataset with enough samples to show the difference
        create_test_parquet_files(tmpdir, num_files=2, rows_per_file=1000)

        print("\nInitializing loader...")
        loader = CoralDataLoader(data_dir=Path(tmpdir), sample_rate=24000)
        print(f"Total samples: {len(loader)}")

        # Test OLD approach (iterate all samples)
        print("\n=== Testing OLD approach (iterating samples) ===")
        start = time.perf_counter()
        speakers_old = test_old_speaker_mapping(loader)
        old_time = time.perf_counter() - start
        print(f"Time: {old_time:.3f}s")
        print(f"Found {len(speakers_old)} speakers: {sorted(speakers_old)}")

        # Test NEW approach (query parquet directly)
        print("\n=== Testing NEW approach (parquet query) ===")
        start = time.perf_counter()
        speakers_new = test_new_speaker_mapping(loader)
        new_time = time.perf_counter() - start
        print(f"Time: {new_time:.3f}s")
        print(f"Found {len(speakers_new)} speakers: {sorted(speakers_new)}")

        # Verify they return the same results
        assert speakers_old == speakers_new, "Results don't match!"
        print("\n✓ Both methods return identical results")

        # Calculate speedup
        if new_time > 0:
            speedup = old_time / new_time
            time_saved = old_time - new_time
            print(f"\n=== PERFORMANCE IMPROVEMENT ===")
            print(f"Old approach: {old_time:.3f}s")
            print(f"New approach: {new_time:.3f}s")
            print(f"Time saved: {time_saved:.3f}s")
            print(f"Speedup: {speedup:.1f}x faster")

            # Extrapolate to real dataset size (18,863 samples)
            real_dataset_size = 18863
            test_dataset_size = len(loader)
            estimated_old_time = old_time * (real_dataset_size / test_dataset_size)
            estimated_new_time = new_time * (real_dataset_size / test_dataset_size)
            estimated_savings = estimated_old_time - estimated_new_time

            print(f"\n=== ESTIMATED IMPROVEMENT ON FULL DATASET ===")
            print(f"Dataset size: {real_dataset_size} samples")
            print(f"Estimated old time: {estimated_old_time:.2f}s ({estimated_old_time/60:.2f} minutes)")
            print(f"Estimated new time: {estimated_new_time:.3f}s")
            print(f"Estimated time saved: {estimated_savings:.2f}s ({estimated_savings/60:.2f} minutes)")

        # Now test with TTSDataset
        print("\n=== Testing TTSDataset initialization ===")
        from src.danish_tts.data.tts_dataset import TTSDataset

        g2p = MockG2P()
        start = time.perf_counter()
        dataset = TTSDataset(data_dir=Path(tmpdir), g2p=g2p, sample_rate=24000)
        init_time = time.perf_counter() - start

        print(f"TTSDataset initialization: {init_time:.3f}s")
        print(f"Speaker mapping created: {len(dataset.speaker_to_id)} speakers")
        print(f"Speakers: {sorted(dataset.speaker_to_id.keys())}")

        # Verify functionality
        item = dataset[0]
        assert isinstance(item["speaker_id"], int)
        assert item["speaker_id"] >= 0
        print(f"\n✓ TTSDataset working correctly")
        print(f"  Sample 0 speaker_id: {item['speaker_id']} (mapped from string)")

        print("\n✓ All tests passed!")
        return speedup


if __name__ == "__main__":
    speedup = test_performance()
