# Phase 0 – Scope & Repos

**Goal:** Single-speaker or dual-speaker Danish Kokoro-style voices, based on CoRal TTS.

You'll need:

- [hexgrad/kokoro](https://github.com/hexgrad/kokoro) (for inference interface + KPipeline)
- [hexgrad/misaki](https://github.com/hexgrad/misaki) (for G2P integration)
- [CoRal Danish TTS data](https://huggingface.co/datasets/yl4579/CoRal) – the read-aloud TTS subset (~2×24h speakers)
- [yl4579/StyleTTS2](https://huggingface.co/yl4579/StyleTTS2) (training code; Kokoro's base model is a finetune of StyleTTS2-LJSpeech)

Work in three places:

- **misaki fork:** add Danish G2P
- **Training repo:** StyleTTS2-based trainer for Danish (new project or fork of StyleTTS2)
- **kokoro fork:** optionally expose your Danish model via a Kokoro-style KPipeline

## Phase 1 – Design Danish Phonemes & G2P (Misaki)

### 1.1 Define a Danish Phoneme Inventory

You need a symbol set that will be:

- Expressive enough for Danish (vowel length, stød, etc.)
- Stable for a neural model (no crazy diacritics; think simple ASCII-ish symbols like Kokoro's EN phones)

**Plan:**

- Inspect [EN_PHONES.md](https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md) in Misaki for how symbols are defined
- From [CoRal](https://huggingface.co/datasets/yl4579/CoRal) / [eSpeak-ng's Danish voice](https://sprogteknologi.dk/), list the IPA/phoneme symbols actually used
- Map each IPA symbol to a Kokoro-style ASCII symbol, e.g.:
  - a / ɑ → ɑ
  - ø → 2
  - y → y
  - ð → D
  - stød → dedicated symbol like ˀ or encoded via accent mark or separate token

Create `DA_PHONES.md` (in your trainer repo) mirroring `EN_PHONES.md` layout, documenting:

- Phoneme symbol
- Example word
- Features (vowel length, etc.)

This doesn't have to be perfect linguistics; it just has to be consistent across G2P and training.

### 1.2 Implement misaki.da G2P

Inside your misaki fork:

In `misaki/`, create `da.py` similar in structure to `en.py` but simpler:

Use phonemizer-fork with espeak-ng Danish as backend, or another Danish G2P.

Implement:

```python
# misaki/da.py
class G2P:
    def __init__(self, fallback=None):
        self.fallback = fallback
        # load phonemizer / lexicon / text normalizer here

    def __call__(self, text: str):
        # 1) normalize text (numbers, abbrevs, etc.)
        # 2) run Danish phonemizer/espeak
        # 3) map raw phones -> your DA_PHONES symbols
        # 4) return ("ph1 ph2 ...", [token_ids])
```

Add Danish normalisation:

- Expand numbers to Danish words (either via `num2words` with `lang='da'` if available, or your own mapping)
- Normalise punctuation, spacing, common abbreviations (dvs., bl.a., etc.)

Wire it into Misaki's installation extras:

In `pyproject.toml`, add something like:

```toml
[project.optional-dependencies]
da = ["phonemizer-fork", "espeak-ng", "num2words-da (if you have it)"]
```

Test standalone:

```python
from misaki import da

g2p = da.G2P()
phonemes, tokens = g2p("Dette er en dansk testsætning.")
print(phonemes)
print(tokens)
```

At this point you have: Danish text → clean phoneme string + token IDs, which is exactly what Kokoro needs on the front-end.

Phase 1 – Design Danish phonemes & G2P (Misaki)
1.1 Define a Danish phoneme inventory

You need a symbol set that will be:

Expressive enough for Danish (vowel length, stød, etc.).

Stable for a neural model (no crazy diacritics; think simple ASCII-ish symbols like Kokoro’s EN phones). 
GitHub
+1

Plan:

Inspect EN_PHONES.md in Misaki for how symbols are defined. 
GitHub

From CoRal / eSpeak-ng’s Danish voice, list the IPA/phoneme symbols actually used. 
sprogteknologi.dk
+1

Map each IPA symbol to a Kokoro-style ASCII symbol, e.g.:

a / ɑ → ɑ

ø → 2

y → y

ð → D

stød → dedicated symbol like ˀ or encoded via accent mark or separate token.

Create DA_PHONES.md (in your trainer repo) mirroring EN_PHONES.md layout, documenting:

Phoneme symbol

Example word

Features (vowel length, etc.)

This doesn’t have to be perfect linguistics; it just has to be consistent across G2P and training.

1.2 Implement misaki.da G2P

Inside your misaki fork:

In misaki/, create da.py similar in structure to en.py but simpler:

Use phonemizer-fork with espeak-ng Danish as backend, or another Danish G2P.

Implement:

# misaki/da.py
class G2P:
    def __init__(self, fallback=None):
        self.fallback = fallback
        # load phonemizer / lexicon / text normalizer here

    def __call__(self, text: str):
        # 1) normalize text (numbers, abbrevs, etc.)
        # 2) run Danish phonemizer/espeak
        # 3) map raw phones -> your DA_PHONES symbols
        # 4) return ("ph1 ph2 ...", [token_ids])


Add Danish normalisation:

Expand numbers to Danish words (either via num2words with lang='da' if available, or your own mapping).

Normalise punctuation, spacing, common abbreviations (dvs., bl.a., etc.).

Wire it into Misaki’s installation extras:

In pyproject.toml, add something like:

```toml
[project.optional-dependencies]
da = ["phonemizer-fork", "espeak-ng", "num2words-da (if you have it)"]
```

Test standalone:

```python
from misaki import da

g2p = da.G2P()
phonemes, tokens = g2p("Dette er en dansk testsætning.")
print(phonemes)
print(tokens)
```

At this point you have: Danish text → clean phoneme string + token IDs, which is exactly what Kokoro needs on the front-end.

## Phase 2 – Prepare CoRal TTS Data for StyleTTS2

### 2.1 Download & Inspect Dataset

Use the TTS subset: 2 professional speakers ~24h each.

From [Hugging Face](https://huggingface.co/datasets/yl4579/CoRal) you'll typically get:

- `audio/` with .wav or .flac files
- metadata (CSV/JSON) with text per utterance

### 2.2 Audio Preprocessing

[Kokoro uses 24 kHz mono audio](https://github.com/hexgrad/kokoro).

For each file:

- Resample to 24,000 Hz, mono
- Normalize peak to, say, -1 dBFS
- Optionally trim leading/trailing silence (small amount of silence at start/end is fine)

Store in e.g.:

```
data/coral_tts_danish/{speaker_id}/{utt_id}.wav
```

### 2.3 Text Normalization & G2P

For each utterance:

- Clean text:
  - Strip extra spaces
  - Normalize quotes, dashes, etc.
  - Expand numbers (42 → toogfyrre) and dates (simple regex rules)

- Run your Danish Misaki:

```python
phonemes, tokens = da_g2p(text_clean)
```

Save a training manifest (e.g. `train.jsonl`):

```json
{
  "audio_path": "data/coral_tts_danish/female_01/000123.wav",
  "text": "Dette er en dansk testsætning.",
  "phonemes": "dˈetə ɑɐ en dˈansk tˈɛtˌsɛtnɪŋ",
  "speaker_id": 0,
  "lang_id": 0,
  "duration": 3.21
}
```

Split into train/val/test (e.g. 95/3/2 or by fixed set of speakers/segments).

You now have a StyleTTS2-friendly dataset: phoneme strings + audio.

## Phase 3 – Train a Danish StyleTTS2 Model

[Kokoro's base is StyleTTS2-LJSpeech](https://huggingface.co/hexgrad/Kokoro-82M) (single-speaker English) finetuned.

Since Kokoro's own training code isn't public, the pragmatic route is:

- Use StyleTTS2 training code
- Use your Danish phoneme set + CoRal TTS
- Train a Danish StyleTTS2 (single-language) model

Later, if you want "true" Kokoro integration, you can adapt its architecture/checkpoint format.

### 3.1 Set up StyleTTS2 Config

In a fork/clone of [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2):

Create a new config file, e.g. `configs/coral_danish.yaml`, derived from their LJSpeech config:

- Change dataset paths to your manifest
- Replace text frontend with phoneme-based inputs (not graphemes)
- Set `n_symbols = len(DA_PHONES)`
- Specify:
  - Sample rate = 24000
  - Speaker embedding: 1 or 2 speakers (you can either:
    - Train two separate models (one per CoRal speaker), or
    - Treat it as a 2-speaker model with `speaker_id ∈ {0,1}`)
  - Language embedding: single language, so `lang_id = 0` or removed

Adjust training hyperparameters:

- Batch size: sized for your GPU (you know this drill)
- Learning rate around 1e-4 to 2e-4 is typical for TTS; warmup steps ~4-8k
- Total steps: with ~48h data, aim at 400k–800k steps depending on batch size

### 3.2 Training Loop (Conceptual)

Using StyleTTS2's training script, each iteration:

- Sample batch: `[phoneme_ids, text_lengths, audio_waveforms, speaker_ids]`
- Forward pass:
  - StyleTTS2 encoder (or text embedding)
  - Style latent sampled
  - Decoder predicts mel / waveform representation
  - ISTFTNet vocoder (or integrated vocoder) predicts audio
- Loss:
  - Reconstruction loss (L1/L2 on STFT or wave)
  - Adversarial loss from speech discriminator (e.g. [WavLM-based discriminator](https://github.com/microsoft/DNS-Challenge))
  - Style regularization losses
- Backprop, optimizer step, EMA, log metrics

Run until:

- Validation loss plateaus
- Subjective listening looks good (no weird vowels or rhythm)
- Checkpoint best model weights

## Phase 4 – Make it Kokoro-style

There are two levels you can aim for:

### Option A – Use StyleTTS2 Interface for Danish (Simpler, Fastest)

Keep inference in StyleTTS2 codebase.

Create a simple `danish_tts.py` that:

```python
from misaki import da
from styletts2 import StyleTTS2

g2p = da.G2P()
model = StyleTTS2.load_from_checkpoint("checkpoints/coral_danish_best.pth")

def synthesize(text: str, speaker_id: int = 0):
    phonemes, tokens = g2p(text)
    audio = model.inference(phoneme_ids=tokens, speaker_id=speaker_id)
    return audio  # numpy array 24kHz
```

You already get "Kokoro-like Danish" in quality and architecture, just not via the kokoro pip package.

### Option B – Deeper Integration: a Danish "Kokoro-style" KPipeline

If you want to plug into KPipeline so your code uses the same interface as Kokoro:

- Create a new model checkpoint repo (HF or local):
  - Package your StyleTTS2 Danish checkpoint and vocoder weights in a format similar to [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) (you can follow structure under hexgrad/Kokoro-82M files)

- Fork [hexgrad/kokoro](https://github.com/hexgrad/kokoro) and add a Danish pipeline:

In `kokoro/__init__.py` or wherever KPipeline lives, add:

```python
if lang_code == "d":  # Danish
    from misaki import da
    self.g2p = da.G2P()
    # load your Danish weights here instead of Kokoro-82M
```

- Provide a mapping from voice names (e.g. "da_coral_female") to specific speaker embedding or checkpoint
- Adapt model loading code:
  - Where Kokoro currently loads Kokoro-82M weights, insert a branch for Danish that loads your checkpoint/model structure
  - You might need a thin wrapper class to match Kokoro's expected forward signature: `(phones, style_latent, speaker, ...) -> waveform`

- Update docs/VOICES:
  - In `VOICES.md`, add a section:
    - lang_code: 'd'
    - Voice names like `da_coral_female`, `da_coral_male`
    - Small sample sentences

This route is more engineering work because Kokoro's training internals aren't public, but the model is still fundamentally StyleTTS2, so it's mostly about matching input/output interfaces, not inventing new ML.

## Phase 5 – Evaluation & Iteration

### Objective Checks:

- Verify alignment between phonemes and output audio (no truncated sentences)
- Check for consistent sampling rate and loudness

### Subjective Listening:

- Collect diverse Danish sentences (with numbers, dates, foreign names)
- Listen for:
  - Stød correctness
  - Vowel length (e.g. pil vs pille)
  - Prosody on questions vs statements

### Iterate:

- If pronunciation is off for certain phonemes → adjust your DA_PHONES mapping and retrain or at least fine-tune
- If rhythm/prosody is flat → extend training, tweak style prior or loss weights

## Reality Check / Gotchas

There is no official Kokoro training repo, so you are effectively:

- Using StyleTTS2 as the training backbone (which Kokoro is built on), and
- Plugging it into the Kokoro-style inference interface yourself

The critical dependency is your Danish G2P (Misaki extension). Once that is clean and stable, the rest is "just" standard TTS training work.