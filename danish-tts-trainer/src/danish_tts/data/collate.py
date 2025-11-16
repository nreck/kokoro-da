"""Batch collation for TTS training."""

from typing import List, Dict
import torch
import torch.nn.functional as F


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate batch of TTS samples with padding.

    Args:
        batch: List of dictionaries from TTSDataset

    Returns:
        Dictionary with padded tensors:
            - phoneme_ids: [batch, max_seq_len] (padded with 0)
            - audio: [batch, max_audio_len] (padded with 0)
            - speaker_ids: [batch]
            - phoneme_lengths: [batch]
            - audio_lengths: [batch]
            - texts: List[str]
    """
    # Extract lists
    phoneme_ids_list = [item["phoneme_ids"] for item in batch]
    audio_list = [item["audio"] for item in batch]
    speaker_ids = [item["speaker_id"] for item in batch]
    texts = [item["text"] for item in batch]

    # Get lengths
    phoneme_lengths = torch.LongTensor([len(p) for p in phoneme_ids_list])
    audio_lengths = torch.LongTensor([len(a) for a in audio_list])

    # Pad phoneme sequences
    max_phoneme_len = phoneme_lengths.max().item()
    phoneme_ids_padded = torch.zeros(len(batch), max_phoneme_len, dtype=torch.long)

    for i, phoneme_ids in enumerate(phoneme_ids_list):
        length = len(phoneme_ids)
        phoneme_ids_padded[i, :length] = phoneme_ids

    # Pad audio
    max_audio_len = audio_lengths.max().item()
    audio_padded = torch.zeros(len(batch), max_audio_len, dtype=torch.float32)

    for i, audio in enumerate(audio_list):
        length = len(audio)
        audio_padded[i, :length] = audio

    # Stack speaker IDs
    speaker_ids_tensor = torch.LongTensor(speaker_ids)

    return {
        "phoneme_ids": phoneme_ids_padded,
        "audio": audio_padded,
        "speaker_ids": speaker_ids_tensor,
        "phoneme_lengths": phoneme_lengths,
        "audio_lengths": audio_lengths,
        "texts": texts,
    }
