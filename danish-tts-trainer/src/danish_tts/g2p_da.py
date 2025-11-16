"""Danish G2P for Misaki."""

import re
from typing import Optional
from num2words import num2words


# Danish abbreviation expansions
ABBREVIATIONS = {
    "dvs.": "det vil sige",
    "dvs": "det vil sige",
    "bl.a.": "blandt andet",
    "bl.a": "blandt andet",
    "osv.": "og så videre",
    "osv": "og så videre",
    "mht.": "med hensyn til",
    "mht": "med hensyn til",
    "ca.": "cirka",
    "ca": "cirka",
    "f.eks.": "for eksempel",
    "f.eks": "for eksempel",
    "mr.": "herr",
    "mrs.": "fru",
    "dr.": "doktor",
}


def normalize_text(text: str) -> str:
    """
    Normalize Danish text for TTS.

    - Remove extra whitespace
    - Expand numbers to words
    - Expand abbreviations
    - Normalize punctuation
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Expand abbreviations (case-insensitive)
    for abbr, expansion in ABBREVIATIONS.items():
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        text = pattern.sub(expansion, text)

    # Expand numbers to Danish words
    def replace_number(match):
        num_str = match.group(0)
        try:
            num = int(num_str)
            # Use num2words with Danish locale
            return num2words(num, lang='da')
        except (ValueError, NotImplementedError):
            return num_str

    text = re.sub(r'\b\d+\b', replace_number, text)

    # Normalize punctuation (keep . , ? ! but remove others for now)
    text = re.sub(r'[^\w\s.,?!æøåÆØÅ-]', '', text)

    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text.strip())

    return text


from phonemizer import phonemize
from phonemizer.backend import EspeakBackend


# IPA to Danish phoneme symbol mapping
# Maps eSpeak-ng IPA output to our DA_PHONES symbols
IPA_TO_DA_PHONE = {
    # Vowels
    "a": "a",
    "ɑ": "ɑː",
    "ɛ": "æ",
    "e": "e",
    "ə": "e",  # schwa -> e
    "ø": "ø",
    "i": "i",
    "ɔ": "o",
    "o": "oː",
    "u": "u",
    "y": "y",
    # Length marker
    "ː": "",  # Handled by vowel variants
    # Consonants
    "p": "p",
    "b": "b",
    "t": "t",
    "d": "d",
    "k": "k",
    "ɡ": "g",
    "g": "g",
    "f": "f",
    "v": "v",
    "s": "s",
    "h": "h",
    "j": "j",
    "l": "l",
    "m": "m",
    "n": "n",
    "ŋ": "ŋ",
    "ʁ": "r",
    "r": "r",
    "ð": "ð",
    # Stress
    "ˈ": "ˈ",
    "ˌ": "ˌ",
    # Glottal stop (stød approximation)
    "ʔ": "ˀ",
    # Word boundary
    " ": "_",
}


class G2P:
    """Danish Grapheme-to-Phoneme converter."""

    def __init__(self, fallback: Optional[str] = None):
        """
        Initialize Danish G2P.

        Args:
            fallback: Optional fallback G2P (not used for Danish currently)
        """
        self.fallback = fallback
        self.backend = EspeakBackend('da', language_switch='remove-flags')

        # Import phoneme mappings from danish_tts if available
        try:
            from danish_tts.phonemes import phoneme_to_id
            self.phoneme_to_id = phoneme_to_id
        except ImportError:
            # Create minimal mapping if danish_tts not installed
            self.phoneme_to_id = lambda x: 0

    def _ipa_to_da_phones(self, ipa: str) -> str:
        """Convert IPA string to Danish phoneme symbols."""
        result = []
        i = 0
        while i < len(ipa):
            # Try 2-char symbols first (e.g., ɑː)
            if i + 1 < len(ipa):
                two_char = ipa[i:i+2]
                if two_char in IPA_TO_DA_PHONE:
                    mapped = IPA_TO_DA_PHONE[two_char]
                    if mapped:  # Skip empty mappings
                        result.append(mapped)
                    i += 2
                    continue

            # Try 1-char symbol
            one_char = ipa[i]
            if one_char in IPA_TO_DA_PHONE:
                mapped = IPA_TO_DA_PHONE[one_char]
                if mapped:
                    result.append(mapped)
            elif one_char.isalpha():
                # Unknown phoneme, keep as-is or map to unknown
                result.append(one_char)

            i += 1

        return " ".join(result)

    def __call__(self, text: str) -> tuple[str, list[int]]:
        """
        Convert Danish text to phonemes.

        Args:
            text: Input Danish text

        Returns:
            Tuple of (phoneme_string, phoneme_ids)
        """
        # Normalize text
        normalized = normalize_text(text)

        # Get IPA from espeak-ng
        ipa = phonemize(
            normalized,
            language='da',
            backend='espeak',
            strip=True,
            preserve_punctuation=False,
            with_stress=True,
        )

        # Convert IPA to our phoneme symbols
        phonemes = self._ipa_to_da_phones(ipa)

        # Convert to token IDs
        phone_list = phonemes.split()
        try:
            from danish_tts.phonemes import phonemes_to_ids
            tokens = phonemes_to_ids(phone_list)
        except ImportError:
            # Fallback if danish_tts not available
            tokens = [self.phoneme_to_id(p) for p in phone_list]

        return phonemes, tokens
