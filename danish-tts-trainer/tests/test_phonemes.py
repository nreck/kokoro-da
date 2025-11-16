import pytest
from danish_tts.phonemes import PHONEME_TO_ID, ID_TO_PHONEME, get_num_phonemes


def test_phoneme_to_id_mapping():
    """Test that all phonemes have unique IDs."""
    assert len(PHONEME_TO_ID) > 40  # At least 40 phonemes
    assert 0 in PHONEME_TO_ID.values()  # Padding exists
    assert 1 in PHONEME_TO_ID.values()  # Unknown exists


def test_id_to_phoneme_inverse():
    """Test that ID_TO_PHONEME is inverse of PHONEME_TO_ID."""
    for phone, idx in PHONEME_TO_ID.items():
        assert ID_TO_PHONEME[idx] == phone


def test_get_num_phonemes():
    """Test that num_phonemes returns correct count."""
    num = get_num_phonemes()
    assert num == len(PHONEME_TO_ID)
    assert num > 40


def test_specific_phonemes_exist():
    """Test that key Danish phonemes exist."""
    assert "a" in PHONEME_TO_ID
    assert "ø" in PHONEME_TO_ID
    assert "ð" in PHONEME_TO_ID
    assert "ˀ" in PHONEME_TO_ID  # stød
    assert "_" in PHONEME_TO_ID  # silence
