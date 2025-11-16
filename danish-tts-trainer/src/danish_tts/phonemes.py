"""Danish phoneme inventory and mappings."""

# Phoneme symbols based on DA_PHONES.md
PHONEMES = [
    "_",  # silence/word boundary
    "<unk>",  # unknown
    # Vowels (short)
    "a", "æ", "e", "ø", "i", "o", "u", "y",
    # Vowels (long)
    "ɑː", "ɛː", "eː", "øː", "iː", "oː", "uː", "yː",
    # Diphthongs
    "ɑi", "ɔi", "ʌu",
    # Consonants
    "p", "b", "t", "d", "k", "g",
    "f", "v", "s", "h",
    "j", "l", "m", "n", "ŋ", "r",
    "ð", "D",
    # Stød
    "ˀ",
    # Stress/prosody
    "ˈ", "ˌ", ".",
]

# Create bidirectional mappings
PHONEME_TO_ID = {phone: idx for idx, phone in enumerate(PHONEMES)}
ID_TO_PHONEME = {idx: phone for idx, phone in enumerate(PHONEMES)}

# Constants
PAD_ID = PHONEME_TO_ID["_"]
UNK_ID = PHONEME_TO_ID["<unk>"]


def get_num_phonemes() -> int:
    """Return total number of phonemes."""
    return len(PHONEMES)


def phoneme_to_id(phoneme: str) -> int:
    """Convert phoneme symbol to ID."""
    return PHONEME_TO_ID.get(phoneme, UNK_ID)


def id_to_phoneme(idx: int) -> str:
    """Convert ID to phoneme symbol."""
    return ID_TO_PHONEME.get(idx, "<unk>")


def phonemes_to_ids(phonemes: list[str]) -> list[int]:
    """Convert list of phoneme symbols to IDs."""
    return [phoneme_to_id(p) for p in phonemes]


def ids_to_phonemes(ids: list[int]) -> list[str]:
    """Convert list of IDs to phoneme symbols."""
    return [id_to_phoneme(i) for i in ids]
