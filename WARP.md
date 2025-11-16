# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a clone/fork of **Kokoro-82M**, an open-weight Text-to-Speech (TTS) model with 82 million parameters. The model provides high-quality speech synthesis while being lightweight and fast.

The repository contains:
- **kokoro/**: Python inference library (main TTS implementation)
- **kokoro.js/**: JavaScript/TypeScript implementation for browser/Node.js

## Commands

### Python Package Development

```bash
# Install the package in development mode (from kokoro/ directory)
cd kokoro
pip install -e .

# Install with specific language support
pip install -e ".[ja]"  # Japanese support
pip install -e ".[zh]"  # Chinese support

# Using uv (if available)
cd kokoro
uv pip install -e .
```

### Running Tests

```bash
# Run tests with pytest (from kokoro/ directory)
cd kokoro
pytest tests/

# Run specific test file
pytest tests/test_custom_stft.py

# Run with verbose output
pytest -v tests/
```

### Using the CLI

```bash
# Basic usage - generate from text
python -m kokoro --text "Hello world" -o output.wav

# Generate from file input
echo "Your text here" > input.txt
python -m kokoro -i input.txt -l a --voice af_heart -o output.wav

# With custom voice and speed
python -m kokoro --text "Hello world" --voice af_bella --speed 1.2 -o output.wav

# Debug mode
python -m kokoro --text "Test" -o output.wav --debug

# Language codes: a (US English), b (UK English), e (Spanish), 
# f (French), h (Hindi), i (Italian), p (Portuguese), j (Japanese), z (Chinese)
```

### Running Examples

```bash
# Device selection example
cd kokoro
python examples/device_examples.py

# Phoneme generation example
python examples/phoneme_example.py

# Export model (ONNX)
python examples/export.py
```

## Architecture

### Core Components

1. **KModel** (`kokoro/model.py`)
   - Main neural network module (torch.nn.Module)
   - Responsibilities: Initialize weights, download from HuggingFace, perform TTS inference
   - Language-agnostic - handles only phoneme → audio conversion
   - Can be shared across multiple KPipeline instances to save memory
   - Components:
     - **CustomAlbert**: BERT-based text encoder for phoneme embeddings
     - **ProsodyPredictor**: Predicts duration, F0 (pitch), and energy
     - **TextEncoder**: Processes phoneme sequences with CNN+LSTM
     - **Decoder** (iSTFTNet): Generates raw audio from features

2. **KPipeline** (`kokoro/pipeline.py`)
   - Language-aware wrapper and orchestration layer
   - Responsibilities:
     - Text → phoneme conversion (G2P) for specific languages
     - Text chunking (handles long inputs by splitting into ≤510 phoneme chunks)
     - Voice management (lazy-loading from HuggingFace)
     - Coordinate KModel for audio generation
   - One pipeline per language, but can share the same KModel
   - Supports multiple voices via weighted averaging

3. **G2P (Grapheme-to-Phoneme)** (via `misaki` library)
   - English (a/b): Uses `misaki.en.G2P` with espeak fallback for OOD words
   - Other languages: Uses `espeak.EspeakG2P` or language-specific implementations
   - Japanese/Chinese require extra dependencies: `pip install misaki[ja]` or `misaki[zh]`

4. **iSTFTNet Decoder** (`kokoro/istftnet.py`)
   - Neural vocoder based on StyleTTS2 architecture
   - Uses AdaIN (Adaptive Instance Normalization) for style conditioning
   - Custom STFT implementation for complex-valued operations (`kokoro/custom_stft.py`)
   - Can optionally use simplified TorchSTFT for compatibility

### Key Design Patterns

- **Generator Pattern**: All audio generation uses Python generators, yielding `KPipeline.Result` objects containing graphemes, phonemes, and audio
- **Lazy Loading**: Models and voices are downloaded from HuggingFace on first use and cached locally
- **Device Flexibility**: Supports CUDA, MPS (Apple Silicon), and CPU with auto-detection
- **Chunking Strategy**: Text is automatically split into processable chunks using a "waterfall" strategy (prefer sentence boundaries like `.!?`, then `:;`, then `,—`)

### Data Flow

```
Text → KPipeline.g2p() → Tokens (MToken objects) → 
→ Chunked by en_tokenize() (if >510 phonemes) → 
→ KModel.forward() → Audio samples
```

For non-English:
```
Text → Chunked by sentence boundaries (≈400 chars) → 
→ G2P for each chunk → Phonemes → 
→ KModel.forward() → Audio samples
```

### Important Constraints

- **Maximum phoneme length**: 510 characters per inference pass (hard limit from ALBERT model's context length)
- **Sample rate**: 24,000 Hz (fixed)
- **Audio format**: PyTorch FloatTensor, range approximately [-1, 1]
- **Voice format**: `.pt` files containing torch tensors with style embeddings

## Device Management

```python
# Auto-detect (default): CUDA > MPS > CPU
pipeline = KPipeline(lang_code='a')

# Explicit device
pipeline = KPipeline(lang_code='a', device='cuda')  # Raises error if unavailable
pipeline = KPipeline(lang_code='a', device='cpu')

# MPS (Apple Silicon) - requires environment variable
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
pipeline = KPipeline(lang_code='a', device='mps')
```

## Working with Voices

- **Predefined voices**: Download automatically from HuggingFace (`hexgrad/Kokoro-82M`)
- **Custom voices**: Load from `.pt` files using `voice='path/to/voice.pt'`
- **Voice blending**: Combine voices with comma delimiter: `voice='af_bella,af_jessica'`
- **Voice naming convention**: First letter indicates language (e.g., `af_*` = American English Female)

## Model Versions

- **Default**: `hexgrad/Kokoro-82M` (v1.0)
- **Chinese v1.1**: `hexgrad/Kokoro-82M-v1.1-zh` (use `repo_id='hexgrad/Kokoro-82M-v1.1-zh'`)

## Dependencies

- **Core**: torch, transformers, huggingface_hub, loguru, numpy, misaki
- **System**: espeak-ng (for English OOD words and some languages)
- **Language-specific**: 
  - Japanese: `pip install misaki[ja]`
  - Chinese: `pip install misaki[zh]`

## Repository Structure Note

This repository is a clone/fork, so:
- The actual Kokoro implementation is in the `kokoro/` subdirectory
- There's also a JavaScript implementation in `kokoro.js/`
- This WARP.md file is at the repository root
- Always work from the `kokoro/` subdirectory for Python development
