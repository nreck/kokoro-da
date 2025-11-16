#!/usr/bin/env python3
"""Test script to verify the performance fix for coral_loader.py."""

import time
import tempfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import soundfile as sf
import io

from src.danish_tts.data.coral_loader import CoralDataLoader


def create_dummy_audio_bytes():
    """Create dummy audio bytes for testing."""
    # Generate 1 second of silence at 24kHz
    audio = np.zeros(24000, dtype=np.float32)
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
        # Create dummy data
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


def test_performance():
    """Test the performance of the loader with caching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Creating test dataset...")
        create_test_parquet_files(tmpdir, num_files=2, rows_per_file=100)

        print("\nInitializing loader...")
        start = time.perf_counter()
        loader = CoralDataLoader(data_dir=Path(tmpdir), sample_rate=24000)
        init_time = time.perf_counter() - start
        print(f"Initialization took: {init_time:.3f}s")
        print(f"Total samples: {len(loader)}")

        # Verify tables are cached
        assert hasattr(loader, 'tables'), "ERROR: tables attribute not found!"
        assert len(loader.tables) == 2, f"ERROR: Expected 2 cached tables, found {len(loader.tables)}"
        print(f"✓ Tables cached in memory: {len(loader.tables)}")

        # Test access performance
        print("\nTesting sample access performance...")
        num_samples = 50

        start = time.perf_counter()
        for i in range(num_samples):
            item = loader[i]
            assert "text" in item
            assert "audio" in item
            assert item["audio"].dtype == np.float32
        end = time.perf_counter()

        total_time = end - start
        avg_time_ms = (total_time / num_samples) * 1000

        print(f"\nPerformance Results:")
        print(f"  Samples accessed: {num_samples}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average per sample: {avg_time_ms:.2f}ms")

        # Verify performance improvement
        # With caching, each access should be < 10ms (vs ~320ms without caching)
        if avg_time_ms < 10:
            print(f"\n✓ PERFORMANCE EXCELLENT: {avg_time_ms:.2f}ms per sample")
            print(f"  Expected improvement: ~{320/avg_time_ms:.0f}x faster than uncached version")
        elif avg_time_ms < 50:
            print(f"\n✓ PERFORMANCE GOOD: {avg_time_ms:.2f}ms per sample")
            print(f"  Expected improvement: ~{320/avg_time_ms:.0f}x faster than uncached version")
        else:
            print(f"\n⚠ WARNING: Performance may not be optimal: {avg_time_ms:.2f}ms per sample")

        print("\n✓ All tests passed!")
        return avg_time_ms


if __name__ == "__main__":
    test_performance()
