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
