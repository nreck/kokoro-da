import pytest
from pathlib import Path
import json
from danish_tts.preprocess_coral import (
    parse_coral_metadata,
    create_manifest_entry,
)


def test_create_manifest_entry():
    """Test creation of manifest entry."""
    entry = create_manifest_entry(
        audio_path="/path/to/audio.wav",
        text="Dette er en test.",
        phonemes="d ɛ t ə _ ɛ r _ ɛ n _ t ɛ s t",
        speaker_id=0,
        duration=2.5,
    )

    assert entry["audio_path"] == "/path/to/audio.wav"
    assert entry["text"] == "Dette er en test."
    assert entry["phonemes"] == "d ɛ t ə _ ɛ r _ ɛ n _ t ɛ s t"
    assert entry["speaker_id"] == 0
    assert entry["lang_id"] == 0  # Danish
    assert entry["duration"] == 2.5


def test_manifest_entry_is_json_serializable():
    """Test that manifest entries can be serialized to JSON."""
    entry = create_manifest_entry(
        audio_path="/path/to/audio.wav",
        text="Test",
        phonemes="t ɛ s t",
        speaker_id=0,
        duration=1.0,
    )

    # Should not raise
    json_str = json.dumps(entry)
    assert isinstance(json_str, str)
