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
