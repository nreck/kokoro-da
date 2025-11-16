# Danish Kokoro TTS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build single-speaker or dual-speaker Danish Kokoro-style TTS voices based on CoRal TTS dataset and StyleTTS2 architecture.

**Architecture:** Fork Misaki for Danish G2P (grapheme-to-phoneme), prepare CoRal TTS data with 24kHz audio + phoneme transcription, train StyleTTS2 model on Danish phonemes (~48h data, 2 speakers), optionally integrate into Kokoro's KPipeline interface.

**Tech Stack:** Python 3.10+, Misaki (G2P), phonemizer/espeak-ng, StyleTTS2, PyTorch, HuggingFace datasets, torchaudio, num2words

---

## Phase 0: Repository Setup & Environment

### Task 0.1: Clone Required Repositories

**Files:**
- None (repository setup only)

**Step 1: Clone Misaki fork**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training
git clone https://github.com/hexgrad/misaki.git
cd misaki
git checkout -b feature/danish-g2p
```

**Step 2: Clone Kokoro (reference)**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training
git clone https://github.com/hexgrad/kokoro.git
```

**Step 3: Clone StyleTTS2 training code**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
git checkout -b feature/danish-training
```

**Step 4: Verify clones**

Run:
```bash
ls -la /Users/nicolajreck/Documents/kokoro-research-training/
```

Expected: Directories `misaki/`, `kokoro/`, `StyleTTS2/` visible

**Step 5: Commit setup notes**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training
git add .
git commit -m "docs: document repo structure for Danish TTS"
```

---

### Task 0.2: Create Training Project Structure

**Files:**
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/README.md`
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/requirements.txt`
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/setup.py`

**Step 1: Create project directory**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training
mkdir -p danish-tts-trainer/src/danish_tts
mkdir -p danish-tts-trainer/tests
mkdir -p danish-tts-trainer/data
mkdir -p danish-tts-trainer/configs
mkdir -p danish-tts-trainer/checkpoints
cd danish-tts-trainer
git init
```

**Step 2: Write README**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/README.md`

```markdown
# Danish TTS Trainer

Training pipeline for Danish Kokoro-style TTS using CoRal dataset and StyleTTS2.

## Components

- Danish phoneme inventory (DA_PHONES.md)
- Data preprocessing (CoRal TTS → StyleTTS2 format)
- Training configs for StyleTTS2
- Inference scripts

## Setup

```bash
pip install -e ".[all]"
```

## Data

Download CoRal TTS subset from: https://huggingface.co/datasets/yl4579/CoRal
```

**Step 3: Write requirements.txt**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/requirements.txt`

```txt
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
librosa>=0.10.0
soundfile>=0.12.0
phonemizer>=3.2.1
num2words>=0.5.12
espeak-ng-python>=0.1.0
huggingface-hub>=0.16.0
datasets>=2.14.0
pyyaml>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0
tensorboard>=2.13.0
```

**Step 4: Write setup.py**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="danish-tts-trainer",
    version="0.1.0",
    description="Danish Kokoro-style TTS training pipeline",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "phonemizer>=3.2.1",
        "num2words>=0.5.12",
        "huggingface-hub>=0.16.0",
        "datasets>=2.14.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "black>=23.7.0", "isort>=5.12.0"],
        "all": ["pytest>=7.4.0", "black>=23.7.0", "isort>=5.12.0"],
    },
)
```

**Step 5: Verify structure**

Run:
```bash
tree -L 2 /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
```

Expected: Directory structure with src/, tests/, data/, configs/, checkpoints/

**Step 6: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
git add .
git commit -m "feat: initialize danish-tts-trainer project structure"
```

---

## Phase 1: Design Danish Phonemes & G2P

### Task 1.1: Define Danish Phoneme Inventory

**Files:**
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/DA_PHONES.md`

**Step 1: Research existing phoneme sets**

Manual task: Read through:
- `/Users/nicolajreck/Documents/kokoro-research-training/misaki/EN_PHONES.md`
- eSpeak-ng Danish phonemes: https://github.com/espeak-ng/espeak-ng/blob/master/phsource/ph_danish
- CoRal dataset phoneme annotations (if available)

**Step 2: Write DA_PHONES.md**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/DA_PHONES.md`

```markdown
# Danish Phoneme Inventory

Phoneme set for Danish Kokoro TTS, compatible with StyleTTS2 and Misaki G2P.

## Design Principles

- ASCII-friendly symbols (like Kokoro English)
- Capture Danish-specific features: vowel length, stød
- Consistent with eSpeak-ng Danish output

## Vowels (Short)

| Symbol | IPA | Example | Word | Notes |
|--------|-----|---------|------|-------|
| a | a | **a**nd | and | short /a/ |
| æ | ɛ | **æ**ble | æble (apple) | short /ɛ/ |
| e | e̝ | **e**n | en (one) | short /e/ |
| ø | ø | **ø**l | øl (beer) | short /ø/ |
| i | i | **i**s | is (ice) | short /i/ |
| o | ɔ | **o**rd | ord (word) | short /ɔ/ |
| u | u | **u**ng | ung (young) | short /u/ |
| y | y | **y**nder | ynder (favors) | short /y/ |

## Vowels (Long)

| Symbol | IPA | Example | Word | Notes |
|--------|-----|---------|------|-------|
| ɑː | ɑː | b**a**d | bad (bath) | long /ɑː/ |
| ɛː | ɛː | s**æ**d | sæd (seed) | long /ɛː/ |
| eː | eː | v**e**d | ved (wood) | long /eː/ |
| øː | øː | **ø**re | øre (ear) | long /øː/ |
| iː | iː | v**i**n | vin (wine) | long /iː/ |
| oː | oː | b**o**d | bod (stall) | long /oː/ |
| uː | uː | h**u**s | hus (house) | long /uː/ |
| yː | yː | h**y**tte | hytte (cabin) | long /yː/ |

## Diphthongs

| Symbol | IPA | Example | Word | Notes |
|--------|-----|---------|------|-------|
| ɑi | ɑi | t**aj** | taj | /ɑi/ |
| ɔi | ɔi | t**oj** | toj (clothes) | /ɔi/ |
| ʌu | ʌu | h**av** | hav (sea) | /ʌu/ |

## Consonants

| Symbol | IPA | Example | Word | Notes |
|--------|-----|---------|------|-------|
| p | p | **p**and | pand | voiceless bilabial |
| b | b | **b**ord | bord | voiced bilabial |
| t | t | **t**ak | tak | voiceless alveolar |
| d | d | **d**ag | dag | voiced alveolar |
| k | k | **k**at | kat | voiceless velar |
| g | ɡ | **g**ammel | gammel | voiced velar |
| f | f | **f**ar | far | voiceless labiodental |
| v | v | **v**and | vand | voiced labiodental |
| s | s | **s**ol | sol | voiceless alveolar |
| h | h | **h**us | hus | voiceless glottal |
| j | j | **j**a | ja | palatal approximant |
| l | l | **l**and | land | alveolar lateral |
| m | m | **m**and | mand | bilabial nasal |
| n | n | **n**avn | navn | alveolar nasal |
| ŋ | ŋ | la**ng** | lang | velar nasal |
| r | ʁ | **r**ød | rød | uvular approximant |
| ð | ð | ma**d** | mad | voiced dental fricative |
| D | ð̞ | me**d** | med | approximant d (soft d) |

## Stød

| Symbol | IPA | Example | Word | Notes |
|--------|-----|---------|------|-------|
| ˀ | ˀ | hun**d**ˀ | hund | glottal reinforcement |

Stød is marked with ˀ after the vowel nucleus.

## Special Symbols

| Symbol | Meaning |
|--------|---------|
| _ | word boundary / silence |
| ˈ | primary stress (before syllable) |
| ˌ | secondary stress (before syllable) |
| . | syllable boundary |

## Total Symbol Count

- Vowels: 16 (8 short + 8 long)
- Diphthongs: 3
- Consonants: 18
- Stød: 1
- Special: 4

**Total: 42 symbols**

## Encoding for StyleTTS2

Each symbol maps to a unique integer ID (0-41).
Reserved IDs:
- 0: padding
- 1: unknown/OOV

Actual phoneme IDs start at 2.
```

**Step 3: Verify phoneme count**

Manual check: Count unique symbols in DA_PHONES.md = 42 phonemes

**Step 4: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
git add DA_PHONES.md
git commit -m "feat: define Danish phoneme inventory (42 symbols)"
```

---

### Task 1.2: Create Phoneme Mapping Module

**Files:**
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/phonemes.py`
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/tests/test_phonemes.py`

**Step 1: Write test for phoneme mapping**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/tests/test_phonemes.py`

```python
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
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
pytest tests/test_phonemes.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'danish_tts.phonemes'"

**Step 3: Write minimal implementation**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/__init__.py`

```python
"""Danish TTS training pipeline."""
```

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/phonemes.py`

```python
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
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
pip install -e .
pytest tests/test_phonemes.py -v
```

Expected: PASS (all tests green)

**Step 5: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
git add src/danish_tts/phonemes.py tests/test_phonemes.py src/danish_tts/__init__.py
git commit -m "feat: add Danish phoneme mapping (42 symbols)"
```

---

### Task 1.3: Implement Misaki Danish G2P - Text Normalizer

**Files:**
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/misaki/misaki/da.py`
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/misaki/tests/test_da.py`

**Step 1: Write test for text normalization**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/misaki/tests/test_da.py`

```python
import pytest
from misaki.da import normalize_text


def test_normalize_whitespace():
    """Test that extra whitespace is normalized."""
    text = "Dette  er   en    test."
    expected = "Dette er en test."
    assert normalize_text(text) == expected


def test_normalize_numbers():
    """Test that numbers are expanded to Danish words."""
    assert normalize_text("Jeg har 1 kat.") == "Jeg har en kat."
    assert normalize_text("Jeg har 2 katte.") == "Jeg har to katte."
    assert normalize_text("Tallet er 42.") == "Tallet er toogfyrre."


def test_normalize_abbreviations():
    """Test Danish abbreviations are expanded."""
    assert "det vil sige" in normalize_text("dvs. dette")
    assert "blandt andet" in normalize_text("bl.a. dette")


def test_normalize_punctuation():
    """Test punctuation handling."""
    text = "Hej! Hvordan går det?"
    result = normalize_text(text)
    assert "!" not in result or result.endswith(".")
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/misaki
pytest tests/test_da.py::test_normalize_whitespace -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'misaki.da'"

**Step 3: Write minimal implementation**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/misaki/misaki/da.py`

```python
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
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/misaki
pip install num2words
pytest tests/test_da.py -v
```

Expected: PASS (may need to adjust num2words Danish output format)

**Step 5: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/misaki
git add misaki/da.py tests/test_da.py
git commit -m "feat: add Danish text normalization for G2P"
```

---

### Task 1.4: Implement Misaki Danish G2P - Phonemizer Integration

**Files:**
- Modify: `/Users/nicolajreck/Documents/kokoro-research-training/misaki/misaki/da.py`
- Modify: `/Users/nicolajreck/Documents/kokoro-research-training/misaki/tests/test_da.py`

**Step 1: Write test for phonemizer integration**

Modify: `/Users/nicolajreck/Documents/kokoro-research-training/misaki/tests/test_da.py`

Add at end:

```python
from misaki.da import G2P


def test_g2p_initialization():
    """Test that G2P class initializes."""
    g2p = G2P()
    assert g2p is not None


def test_g2p_simple_word():
    """Test G2P on simple Danish word."""
    g2p = G2P()
    phonemes, tokens = g2p("hund")

    assert isinstance(phonemes, str)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert "h" in phonemes or "ɔ" in phonemes  # Contains Danish sounds


def test_g2p_sentence():
    """Test G2P on Danish sentence."""
    g2p = G2P()
    phonemes, tokens = g2p("Dette er en dansk testsætning.")

    assert isinstance(phonemes, str)
    assert len(tokens) > 10  # Reasonable length
    assert "_" in phonemes  # Word boundaries
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/misaki
pytest tests/test_da.py::test_g2p_initialization -v
```

Expected: FAIL with "ImportError: cannot import name 'G2P'"

**Step 3: Write minimal implementation**

Modify: `/Users/nicolajreck/Documents/kokoro-research-training/misaki/misaki/da.py`

Add at end:

```python
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
```

**Step 4: Install dependencies**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/misaki
pip install phonemizer espeak-ng
```

**Step 5: Run test to verify it passes**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/misaki
pytest tests/test_da.py::test_g2p_simple_word -v
```

Expected: PASS (eSpeak-ng must be installed on system)

**Step 6: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/misaki
git add misaki/da.py tests/test_da.py
git commit -m "feat: add Danish G2P with eSpeak-ng backend"
```

---

### Task 1.5: Update Misaki Package Configuration

**Files:**
- Modify: `/Users/nicolajreck/Documents/kokoro-research-training/misaki/pyproject.toml`

**Step 1: Add Danish optional dependencies**

Modify: `/Users/nicolajreck/Documents/kokoro-research-training/misaki/pyproject.toml`

Find the `[project.optional-dependencies]` section and add:

```toml
[project.optional-dependencies]
da = [
    "phonemizer>=3.2.1",
    "num2words>=0.5.12",
]
```

**Step 2: Test installation**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/misaki
pip install -e ".[da]"
```

Expected: No errors, phonemizer and num2words installed

**Step 3: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/misaki
git add pyproject.toml
git commit -m "feat: add Danish G2P installation extras"
```

---

## Phase 2: Prepare CoRal TTS Data

### Task 2.1: Download CoRal Dataset

**Files:**
- None (data download only)

**Step 1: Install HuggingFace CLI**

```bash
pip install huggingface-hub[cli]
```

**Step 2: Download CoRal TTS subset**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/data
huggingface-cli download yl4579/CoRal --repo-type dataset --local-dir coral_raw
```

Expected: Downloads ~50GB of data (both TTS and ASR subsets)

**Step 3: Verify download**

Run:
```bash
ls -lh /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/data/coral_raw/
```

Expected: See audio files and metadata

**Step 4: Document download**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/data/README.md`

```markdown
# Data Directory

## CoRal TTS Dataset

Downloaded from: https://huggingface.co/datasets/yl4579/CoRal

### Structure
- `coral_raw/`: Original downloaded data
- `coral_processed/`: Preprocessed 24kHz audio + phoneme transcripts
- `manifests/`: Train/val/test splits in JSONL format

### Speakers
- Speaker 0: Female, ~24h
- Speaker 1: Male, ~24h

Total: ~48h of Danish TTS data
```

**Step 5: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
git add data/README.md
git commit -m "docs: document CoRal dataset structure"
```

---

### Task 2.2: Implement Audio Preprocessing

**Files:**
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/audio_utils.py`
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/tests/test_audio_utils.py`

**Step 1: Write test for audio resampling**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/tests/test_audio_utils.py`

```python
import pytest
import numpy as np
import torch
from danish_tts.audio_utils import (
    load_and_resample_audio,
    normalize_audio,
    trim_silence,
    TARGET_SAMPLE_RATE,
)


def test_target_sample_rate():
    """Test that target sample rate is 24kHz."""
    assert TARGET_SAMPLE_RATE == 24000


def test_normalize_audio():
    """Test audio normalization to -1 dBFS peak."""
    # Create test signal with known peak
    audio = np.array([0.0, 0.5, -0.5, 0.25, -0.25])
    normalized = normalize_audio(audio, target_db=-1.0)

    # Peak should be close to 10^(-1/20) ≈ 0.891
    expected_peak = 10 ** (-1.0 / 20)
    actual_peak = np.abs(normalized).max()
    assert abs(actual_peak - expected_peak) < 0.01


def test_normalize_audio_zero():
    """Test that zero audio stays zero."""
    audio = np.zeros(100)
    normalized = normalize_audio(audio)
    assert np.allclose(normalized, 0)
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
pytest tests/test_audio_utils.py::test_target_sample_rate -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'danish_tts.audio_utils'"

**Step 3: Write minimal implementation**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/audio_utils.py`

```python
"""Audio preprocessing utilities for Danish TTS."""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


TARGET_SAMPLE_RATE = 24000  # 24 kHz for Kokoro compatibility


def normalize_audio(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """
    Normalize audio to target peak dBFS.

    Args:
        audio: Input audio array
        target_db: Target peak level in dBFS (default: -1.0)

    Returns:
        Normalized audio array
    """
    # Handle zero audio
    peak = np.abs(audio).max()
    if peak == 0:
        return audio

    # Calculate target peak amplitude
    target_peak = 10 ** (target_db / 20)

    # Normalize
    scale = target_peak / peak
    return audio * scale


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: int = 40,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        top_db: Threshold in dB below peak to consider as silence
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation

    Returns:
        Trimmed audio array
    """
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return trimmed


def load_and_resample_audio(
    filepath: Path | str,
    target_sr: int = TARGET_SAMPLE_RATE,
    normalize: bool = True,
    trim: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        filepath: Path to audio file
        target_sr: Target sample rate (default: 24000)
        normalize: Whether to normalize audio (default: True)
        trim: Whether to trim silence (default: True)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Load audio (librosa automatically converts to mono)
    audio, sr = librosa.load(filepath, sr=target_sr, mono=True)

    # Trim silence if requested
    if trim:
        audio = trim_silence(audio, sr)

    # Normalize if requested
    if normalize:
        audio = normalize_audio(audio, target_db=-1.0)

    return audio, sr


def save_audio(
    audio: np.ndarray,
    filepath: Path | str,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> None:
    """
    Save audio to file.

    Args:
        audio: Audio array to save
        filepath: Output file path
        sample_rate: Sample rate of audio
    """
    sf.write(filepath, audio, sample_rate, subtype='PCM_16')
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
pytest tests/test_audio_utils.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
git add src/danish_tts/audio_utils.py tests/test_audio_utils.py
git commit -m "feat: add audio preprocessing utilities (24kHz resample, normalize, trim)"
```

---

### Task 2.3: Implement Dataset Preprocessing Script

**Files:**
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/preprocess_coral.py`
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/tests/test_preprocess_coral.py`

**Step 1: Write test for metadata parsing**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/tests/test_preprocess_coral.py`

```python
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
```

**Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
pytest tests/test_preprocess_coral.py::test_create_manifest_entry -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'danish_tts.preprocess_coral'"

**Step 3: Write minimal implementation**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/preprocess_coral.py`

```python
"""Preprocess CoRal TTS dataset for StyleTTS2 training."""

import json
import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import librosa

from danish_tts.audio_utils import load_and_resample_audio, save_audio


def create_manifest_entry(
    audio_path: str,
    text: str,
    phonemes: str,
    speaker_id: int,
    duration: float,
    lang_id: int = 0,
) -> dict:
    """
    Create a manifest entry for training.

    Args:
        audio_path: Path to preprocessed audio file
        text: Original text transcript
        phonemes: Phoneme sequence (space-separated)
        speaker_id: Speaker ID (0 or 1 for CoRal)
        duration: Audio duration in seconds
        lang_id: Language ID (0 for Danish)

    Returns:
        Dictionary with manifest entry
    """
    return {
        "audio_path": audio_path,
        "text": text,
        "phonemes": phonemes,
        "speaker_id": speaker_id,
        "lang_id": lang_id,
        "duration": duration,
    }


def parse_coral_metadata(metadata_file: Path) -> list[dict]:
    """
    Parse CoRal dataset metadata.

    Args:
        metadata_file: Path to metadata CSV/JSON file

    Returns:
        List of metadata entries
    """
    # TODO: Implement based on actual CoRal metadata format
    # This is a placeholder
    raise NotImplementedError("Implement based on CoRal metadata format")


def preprocess_utterance(
    audio_path: Path,
    text: str,
    output_path: Path,
    g2p,
) -> Optional[dict]:
    """
    Preprocess single utterance.

    Args:
        audio_path: Path to input audio
        text: Text transcript
        output_path: Path to save preprocessed audio
        g2p: G2P instance for phonemization

    Returns:
        Manifest entry or None if preprocessing failed
    """
    try:
        # Load and preprocess audio
        audio, sr = load_and_resample_audio(
            audio_path,
            normalize=True,
            trim=True,
        )

        # Get duration
        duration = len(audio) / sr

        # Skip very short or very long utterances
        if duration < 0.5 or duration > 15.0:
            return None

        # Save preprocessed audio
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_audio(audio, output_path, sr)

        # Get phonemes
        phonemes, _ = g2p(text)

        return {
            "audio": audio,
            "duration": duration,
            "phonemes": phonemes,
        }

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def main():
    """Main preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocess CoRal TTS dataset"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to raw CoRal dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to save preprocessed data",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata file",
    )
    args = parser.parse_args()

    print("Preprocessing CoRal TTS dataset...")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")

    # TODO: Implement full preprocessing pipeline
    # This is a placeholder
    raise NotImplementedError("Implement full preprocessing")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
pytest tests/test_preprocess_coral.py -v
```

Expected: PASS (for implemented tests)

**Step 5: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
git add src/danish_tts/preprocess_coral.py tests/test_preprocess_coral.py
git commit -m "feat: add CoRal preprocessing scaffold"
```

---

## Phase 3: StyleTTS2 Training Configuration

### Task 3.1: Create Danish Training Config

**Files:**
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/configs/coral_danish.yaml`

**Step 1: Write training configuration**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/configs/coral_danish.yaml`

```yaml
# Danish StyleTTS2 Training Configuration
# Based on CoRal TTS dataset

# Model architecture
model:
  n_symbols: 42  # Number of Danish phonemes
  n_speakers: 2  # CoRal has 2 speakers
  n_languages: 1  # Single language (Danish)

  # Text encoder
  text_encoder:
    channels: 256
    kernel_size: 5
    depth: 3
    dropout: 0.1

  # Style encoder (from reference audio)
  style_encoder:
    dim: 256
    n_layers: 3
    kernel_size: 5

  # Decoder (converts text + style -> mel/audio)
  decoder:
    channels: 256
    kernel_size: 5
    dilation_base: 2
    n_layers: 8
    dropout: 0.1

  # Vocoder (mel -> waveform)
  vocoder:
    type: "istftnet"  # Integrated STFT vocoder
    n_fft: 2048
    hop_length: 300  # 12.5ms at 24kHz
    win_length: 1200  # 50ms at 24kHz

# Training data
data:
  train_manifest: "data/manifests/train.jsonl"
  val_manifest: "data/manifests/val.jsonl"
  sample_rate: 24000

  # Data augmentation
  augmentation:
    pitch_shift: 0.0  # No pitch shift for now
    time_stretch: 0.0  # No time stretch for now
    add_noise: false

# Training hyperparameters
training:
  batch_size: 16  # Adjust based on GPU memory
  num_workers: 4
  max_steps: 600000  # ~600k steps for 48h data

  # Learning rate
  learning_rate: 2.0e-4
  warmup_steps: 8000
  lr_schedule: "warmup_cosine"

  # Gradient clipping
  grad_clip_norm: 5.0

  # Mixed precision
  use_amp: true

  # Checkpointing
  checkpoint_interval: 10000  # Every 10k steps
  keep_n_checkpoints: 5

  # Validation
  val_interval: 5000  # Every 5k steps
  val_samples: 10  # Generate 10 samples during validation

# Loss weights
loss:
  reconstruction: 1.0  # Mel/STFT reconstruction
  adversarial: 1.0  # GAN discriminator loss
  style_kl: 0.1  # KL divergence on style latent
  duration: 1.0  # Duration prediction loss

# Discriminator (for adversarial training)
discriminator:
  type: "wavlm"  # WavLM-based discriminator
  layers: [2, 3, 4]  # Which WavLM layers to use
  learning_rate: 2.0e-4

# Logging
logging:
  log_interval: 100  # Log every 100 steps
  tensorboard_dir: "logs/tensorboard"
  sample_dir: "logs/samples"

# Paths
paths:
  checkpoint_dir: "checkpoints"
  resume_from: null  # Set to checkpoint path to resume
```

**Step 2: Verify YAML syntax**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
python -c "import yaml; yaml.safe_load(open('configs/coral_danish.yaml'))"
```

Expected: No errors

**Step 3: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
git add configs/coral_danish.yaml
git commit -m "feat: add Danish StyleTTS2 training config"
```

---

### Task 3.2: Create Training Script Scaffold

**Files:**
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/train.py`

**Step 1: Write training script scaffold**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/train.py`

```python
"""Training script for Danish StyleTTS2."""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: dict) -> nn.Module:
    """
    Initialize StyleTTS2 model.

    Args:
        config: Model configuration

    Returns:
        Model instance
    """
    # TODO: Integrate actual StyleTTS2 model
    # For now, placeholder
    raise NotImplementedError("Integrate StyleTTS2 model from yl4579/StyleTTS2")


def setup_dataloader(config: dict, split: str = "train") -> DataLoader:
    """
    Create data loader.

    Args:
        config: Data configuration
        split: 'train' or 'val'

    Returns:
        DataLoader instance
    """
    # TODO: Implement dataset class
    raise NotImplementedError("Implement TTS dataset class")


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> dict:
    """
    Single training step.

    Args:
        model: StyleTTS2 model
        batch: Training batch
        optimizer: Optimizer
        config: Training configuration

    Returns:
        Dictionary with loss values
    """
    # TODO: Implement training step
    raise NotImplementedError("Implement training step")


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    config: dict,
    step: int,
    writer: SummaryWriter,
) -> float:
    """
    Validation loop.

    Args:
        model: StyleTTS2 model
        val_loader: Validation data loader
        config: Configuration
        step: Current training step
        writer: TensorBoard writer

    Returns:
        Average validation loss
    """
    # TODO: Implement validation
    raise NotImplementedError("Implement validation")


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train Danish StyleTTS2")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Training for {config['training']['max_steps']} steps")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: Implement full training loop
    # 1. Setup model
    # 2. Setup dataloaders
    # 3. Setup optimizers
    # 4. Training loop with validation
    # 5. Checkpointing
    # 6. TensorBoard logging

    print("Training not yet implemented - integrate StyleTTS2")


if __name__ == "__main__":
    main()
```

**Step 2: Test script runs**

Run:
```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
python src/danish_tts/train.py --config configs/coral_danish.yaml
```

Expected: Prints config info and "Training not yet implemented" message

**Step 3: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
git add src/danish_tts/train.py
git commit -m "feat: add training script scaffold"
```

---

## Phase 4: Kokoro Integration (Optional)

### Task 4.1: Create Danish Inference Script (Simple Option)

**Files:**
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/inference.py`

**Step 1: Write inference script**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/inference.py`

```python
"""Danish TTS inference script."""

import argparse
from pathlib import Path
import torch
import soundfile as sf
import sys

# Add misaki to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "misaki"))

from misaki.da import G2P


def load_model(checkpoint_path: Path, device: str = "cpu"):
    """
    Load trained Danish TTS model.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    # TODO: Integrate StyleTTS2 model loading
    raise NotImplementedError("Integrate StyleTTS2 inference")


def synthesize(
    model,
    text: str,
    g2p: G2P,
    speaker_id: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Synthesize speech from text.

    Args:
        model: Trained TTS model
        text: Input Danish text
        g2p: Danish G2P instance
        speaker_id: Speaker ID (0 or 1)
        device: Device to run inference on

    Returns:
        Audio waveform tensor (shape: [1, n_samples])
    """
    # Get phonemes
    phonemes, tokens = g2p(text)
    print(f"Phonemes: {phonemes}")

    # TODO: Run model inference
    # tokens_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    # speaker_tensor = torch.LongTensor([speaker_id]).to(device)
    # with torch.no_grad():
    #     audio = model.inference(tokens_tensor, speaker_tensor)

    raise NotImplementedError("Integrate StyleTTS2 inference")


def main():
    """CLI for Danish TTS inference."""
    parser = argparse.ArgumentParser(description="Danish TTS Inference")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Danish text to synthesize",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=0,
        choices=[0, 1],
        help="Speaker ID (0=female, 1=male)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output audio file path (.wav)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    args = parser.parse_args()

    # Initialize G2P
    print("Loading Danish G2P...")
    g2p = G2P()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)

    # Synthesize
    print(f"Synthesizing: {args.text}")
    audio = synthesize(
        model,
        args.text,
        g2p,
        speaker_id=args.speaker,
        device=args.device,
    )

    # Save audio
    print(f"Saving to {args.output}")
    sf.write(args.output, audio.cpu().numpy().squeeze(), 24000)
    print("Done!")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
git add src/danish_tts/inference.py
git commit -m "feat: add Danish TTS inference script scaffold"
```

---

## Phase 5: Testing & Validation

### Task 5.1: Create Test Sentences

**Files:**
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/test_sentences.txt`

**Step 1: Write diverse test sentences**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/test_sentences.txt`

```
# Danish TTS Test Sentences
# Cover diverse phonetic phenomena

# Basic sentences
Dette er en dansk testsætning.
Jeg hedder Claude og jeg kan tale dansk.
Hvordan går det med dig i dag?

# Numbers
Jeg har 1 kat og 2 hunde.
Tallet er 42.
Det koster 100 kroner.

# Dates and times
I dag er det den 16. november 2025.
Klokken er 15 minutter over 3.

# Stød examples
En hund (with stød)
Mange hunde (without stød)
Et hus (with stød)
Mange huse (without stød)

# Vowel length contrasts
Pil (short i) vs. pille (long i)
Tak (short a) vs. tag (long a)

# Questions vs statements
Dette er en test.
Er dette en test?

# Foreign names
Claude kommer fra San Francisco.
København er hovedstaden i Danmark.

# Abbreviations
Dvs. det vil sige.
Bl.a. dette og hint.

# Complex sentences
Selvom vejret var dårligt, gik vi en tur i skoven og så mange forskellige fugle.
Den danske sprogets mange vokaler og stød gør det udfordrende for udenlandske talere.
```

**Step 2: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
git add test_sentences.txt
git commit -m "feat: add Danish TTS test sentences"
```

---

### Task 5.2: Create Evaluation Script

**Files:**
- Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/evaluate.py`

**Step 1: Write evaluation script**

Create: `/Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer/src/danish_tts/evaluate.py`

```python
"""Evaluation script for Danish TTS model."""

import argparse
from pathlib import Path
from typing import List
import torch
import soundfile as sf
from tqdm import tqdm

from danish_tts.inference import load_model, synthesize
from misaki.da import G2P


def load_test_sentences(filepath: Path) -> List[str]:
    """
    Load test sentences from file.

    Args:
        filepath: Path to test sentences file

    Returns:
        List of test sentences (excluding comments and empty lines)
    """
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                sentences.append(line)
    return sentences


def evaluate_model(
    model,
    g2p: G2P,
    test_sentences: List[str],
    output_dir: Path,
    speaker_id: int = 0,
    device: str = "cpu",
) -> None:
    """
    Evaluate model on test sentences.

    Args:
        model: Trained TTS model
        g2p: Danish G2P instance
        test_sentences: List of test sentences
        output_dir: Directory to save generated audio
        speaker_id: Speaker ID
        device: Device to run on
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating on {len(test_sentences)} test sentences...")

    for idx, sentence in enumerate(tqdm(test_sentences)):
        try:
            # Synthesize
            audio = synthesize(
                model,
                sentence,
                g2p,
                speaker_id=speaker_id,
                device=device,
            )

            # Save audio
            output_path = output_dir / f"test_{idx:03d}.wav"
            sf.write(
                output_path,
                audio.cpu().numpy().squeeze(),
                24000,
            )

            # Save transcript
            transcript_path = output_dir / f"test_{idx:03d}.txt"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(sentence)

        except Exception as e:
            print(f"Error synthesizing '{sentence}': {e}")


def main():
    """CLI for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Danish TTS Model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test_sentences",
        type=Path,
        default=Path("test_sentences.txt"),
        help="Path to test sentences file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("evaluation_outputs"),
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=0,
        choices=[0, 1],
        help="Speaker ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    args = parser.parse_args()

    # Load test sentences
    print(f"Loading test sentences from {args.test_sentences}...")
    sentences = load_test_sentences(args.test_sentences)
    print(f"Loaded {len(sentences)} sentences")

    # Initialize G2P
    print("Loading Danish G2P...")
    g2p = G2P()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)

    # Evaluate
    evaluate_model(
        model,
        g2p,
        sentences,
        args.output_dir,
        speaker_id=args.speaker,
        device=args.device,
    )

    print(f"Evaluation complete! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
cd /Users/nicolajreck/Documents/kokoro-research-training/danish-tts-trainer
git add src/danish_tts/evaluate.py
git commit -m "feat: add model evaluation script"
```

---

## Next Steps & Integration Points

### StyleTTS2 Integration Tasks (Not Detailed Here)

The following tasks require deep integration with the StyleTTS2 codebase:

1. **Dataset Class**: Implement `TorchDataset` that loads JSONL manifests and yields batches of (phoneme_ids, audio, speaker_ids)

2. **Model Integration**:
   - Copy/adapt StyleTTS2 model architecture from `yl4579/StyleTTS2`
   - Modify text encoder to use Danish phonemes (n_symbols=42)
   - Configure for 2-speaker setup

3. **Training Loop**:
   - Implement forward pass: text_encoder → style_encoder → decoder → vocoder
   - Implement losses: reconstruction + adversarial + KL divergence
   - Add EMA (exponential moving average) for stable checkpoints
   - Add gradient accumulation for large effective batch sizes

4. **Discriminator**:
   - Integrate WavLM-based discriminator for adversarial training
   - Alternating optimizer updates (generator vs discriminator)

5. **Inference**:
   - Load checkpoint
   - Convert text → phonemes → model forward → waveform
   - Save as 24kHz WAV

### Recommended Workflow

**@superpowers:test-driven-development** - Use TDD for each component

**@superpowers:verification-before-completion** - Always verify outputs before moving forward

**Execution:**
1. Start with Phase 0-1 (G2P)
2. Verify G2P works standalone on test sentences
3. Proceed to Phase 2 (data preprocessing)
4. Verify preprocessed data quality (listen to samples)
5. Phase 3 (training integration - most complex)
6. Start training, monitor losses
7. Phase 4-5 (inference & evaluation)

**Frequent commits** after each working component!

---

## Summary

This plan provides:
- ✅ Exact file paths for every component
- ✅ Complete code for scaffolds and utilities
- ✅ Test-first approach with pytest
- ✅ Verification commands with expected outputs
- ✅ Git commits at each milestone

**Missing pieces** (intentionally left as TODOs):
- CoRal metadata parsing (depends on actual dataset format)
- StyleTTS2 model integration (requires cloning and adapting their code)
- Full training loop (needs StyleTTS2 components)
- Actual model inference (needs trained checkpoint)

These TODOs represent integration work that can't be fully specified without exploring the StyleTTS2 codebase structure and CoRal dataset format.
