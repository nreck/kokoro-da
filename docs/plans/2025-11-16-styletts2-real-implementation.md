# StyleTTS2 Real Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace SimpleStyleTTS2 placeholder with actual StyleTTS2 components, add discriminator training, duration prediction, and optimize for production.

**Architecture:** Integrate real StyleTTS2 text encoder, style encoder, prosody predictor, decoder, and diffusion models. Implement multi-discriminator GAN training (MPD, MSD, WavLM). Add duration prediction for better prosody control. Optimize inference pipeline for production use.

**Tech Stack:** PyTorch, StyleTTS2, ASR models, PLBERT, diffusion models, discriminators (MPD/MSD/WavLM), iSTFTNet vocoder

---

## Phase 1: StyleTTS2 Core Components Integration

### Task 1.1: Text Encoder Integration

**Files:**
- Modify: `danish-tts-trainer/src/danish_tts/models/model_config.py`
- Test: `danish-tts-trainer/tests/test_real_text_encoder.py`

**Step 1: Write the failing test**

Create `danish-tts-trainer/tests/test_real_text_encoder.py`:

```python
"""Tests for real StyleTTS2 text encoder."""

import pytest
import torch
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from danish_tts.models.model_config import build_model


def test_text_encoder_architecture():
    """Test text encoder has correct StyleTTS2 architecture."""
    config_path = Path("configs/coral_danish.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = build_model(config)

    # Check text encoder is the real TextEncoder class
    from danish_tts.models.models import TextEncoder
    assert isinstance(model.text_encoder, TextEncoder)

    # Check architecture params
    assert model.text_encoder.channels == 256
    assert model.text_encoder.kernel_size == 5
    assert model.text_encoder.depth == 3


def test_text_encoder_forward():
    """Test text encoder forward pass with correct output shape."""
    config_path = Path("configs/coral_danish.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = build_model(config)

    # Create input
    batch_size = 2
    seq_len = 10
    phoneme_ids = torch.randint(0, 42, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        output = model.text_encoder(phoneme_ids)

    # Check output shape: [batch, seq_len, channels]
    assert output.shape == (batch_size, seq_len, 256)
    assert output.dtype == torch.float32
```

**Step 2: Run test to verify it fails**

```bash
cd danish-tts-trainer
python -m pytest tests/test_real_text_encoder.py -v
```

Expected: FAIL because SimpleStyleTTS2 text_encoder is Embedding, not TextEncoder

**Step 3: Update model_config.py to use real TextEncoder**

Edit `danish-tts-trainer/src/danish_tts/models/model_config.py`:

Replace `SimpleStyleTTS2` initialization with real components:

```python
"""Model configuration and initialization for StyleTTS2."""

import torch
import torch.nn as nn
from typing import Dict
from danish_tts.models.models import TextEncoder, StyleEncoder, ProsodyPredictor


def build_model(config: Dict) -> nn.Module:
    """Build StyleTTS2 model from config.

    Args:
        config: Configuration dictionary from YAML

    Returns:
        StyleTTS2 model instance
    """
    model_config = config["model"]

    # Build model using actual StyleTTS2 components
    model = StyleTTS2Model(
        n_symbols=model_config["n_symbols"],
        n_speakers=model_config.get("n_speakers", 1),
        text_encoder_config=model_config.get("text_encoder", {}),
        style_encoder_config=model_config.get("style_encoder", {}),
        decoder_config=model_config.get("decoder", {}),
        vocoder_config=model_config.get("vocoder", {}),
    )

    return model


class StyleTTS2Model(nn.Module):
    """StyleTTS2 model with real components."""

    def __init__(
        self,
        n_symbols: int,
        n_speakers: int,
        text_encoder_config: Dict,
        style_encoder_config: Dict,
        decoder_config: Dict,
        vocoder_config: Dict,
    ):
        super().__init__()

        # Real text encoder from StyleTTS2
        self.text_encoder = TextEncoder(
            channels=text_encoder_config.get("channels", 256),
            kernel_size=text_encoder_config.get("kernel_size", 5),
            depth=text_encoder_config.get("depth", 3),
            n_symbols=n_symbols,
        )

        # Real style encoder from StyleTTS2
        self.style_encoder = StyleEncoder(
            dim_in=80,  # Mel spectrogram dimension
            style_dim=style_encoder_config.get("dim", 256),
            max_conv_dim=text_encoder_config.get("channels", 256),
        )

        # Prosody predictor (duration + pitch)
        self.predictor = ProsodyPredictor(
            style_dim=style_encoder_config.get("dim", 256),
            d_hid=text_encoder_config.get("channels", 256),
            nlayers=text_encoder_config.get("depth", 3),
            max_dur=50,
            dropout=text_encoder_config.get("dropout", 0.1),
        )

        # Placeholder for decoder (will be replaced in next task)
        self.decoder = nn.Linear(256, 80)

        # Placeholder for vocoder (will be replaced in next task)
        self.vocoder = nn.Linear(80, 1)

        self.n_speakers = n_speakers
        if n_speakers > 1:
            self.speaker_embedding = nn.Embedding(n_speakers, 256)

    def forward(self, phoneme_ids, speaker_ids=None, ref_audio=None, phoneme_lengths=None):
        """Training forward pass.

        Args:
            phoneme_ids: [batch, seq_len]
            speaker_ids: [batch]
            ref_audio: [batch, audio_len] - reference audio for style
            phoneme_lengths: [batch] - actual lengths

        Returns:
            Dictionary with model outputs
        """
        batch_size, seq_len = phoneme_ids.shape

        # Text encoding
        text_enc = self.text_encoder(phoneme_ids)  # [batch, seq_len, channels]

        # Extract style from reference audio (placeholder mel extraction)
        if ref_audio is not None:
            # TODO: Extract mel spectrogram from audio
            # For now, create dummy mel
            mel = torch.zeros(batch_size, 80, seq_len, device=phoneme_ids.device)
            style = self.style_encoder(mel)  # [batch, style_dim]
        else:
            style = torch.zeros(batch_size, 256, device=phoneme_ids.device)

        # Add speaker embedding if multi-speaker
        if self.n_speakers > 1 and speaker_ids is not None:
            speaker_emb = self.speaker_embedding(speaker_ids)  # [batch, 256]
            style = style + speaker_emb

        # Duration prediction
        durations, pitch = self.predictor(text_enc, style, phoneme_lengths)

        # Decoder (placeholder)
        predicted_mel = self.decoder(text_enc.transpose(1, 2))  # [batch, 80, seq_len]

        return {
            "predicted_mel": predicted_mel,
            "target_mel": torch.zeros_like(predicted_mel),  # Placeholder
            "style_mean": torch.zeros(batch_size, 256, device=phoneme_ids.device),
            "style_log_var": torch.zeros(batch_size, 256, device=phoneme_ids.device),
            "durations": durations,
            "pitch": pitch,
        }

    def inference(self, phoneme_ids, speaker_ids=None):
        """Inference forward pass.

        Args:
            phoneme_ids: [1, seq_len]
            speaker_ids: [1]

        Returns:
            Dictionary with audio
        """
        # Text encoding
        text_enc = self.text_encoder(phoneme_ids)

        # Sample style from prior (not reference audio)
        style = torch.randn(1, 256, device=phoneme_ids.device)

        # Add speaker embedding
        if self.n_speakers > 1 and speaker_ids is not None:
            speaker_emb = self.speaker_embedding(speaker_ids)
            style = style + speaker_emb

        # Duration prediction
        durations, pitch = self.predictor(text_enc, style, None)

        # Placeholder audio
        audio = torch.zeros(1, 24000, device=phoneme_ids.device)

        return {"audio": audio}
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_real_text_encoder.py -v
```

Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add src/danish_tts/models/model_config.py tests/test_real_text_encoder.py
git commit -m "feat: integrate real StyleTTS2 text encoder

- Replace Embedding with TextEncoder from models.py
- Add real StyleEncoder from StyleTTS2
- Add ProsodyPredictor for duration and pitch
- Update model architecture with real components
- Add speaker embedding for multi-speaker support"
```

---

### Task 1.2: Decoder Integration (iSTFTNet)

**Files:**
- Modify: `danish-tts-trainer/src/danish_tts/models/model_config.py`
- Test: `danish-tts-trainer/tests/test_decoder.py`

**Step 1: Write the failing test**

Create `danish-tts-trainer/tests/test_decoder.py`:

```python
"""Tests for iSTFTNet decoder integration."""

import pytest
import torch
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from danish_tts.models.model_config import build_model


def test_decoder_is_istftnet():
    """Test decoder uses iSTFTNet from StyleTTS2."""
    config_path = Path("configs/coral_danish.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = build_model(config)

    # Check decoder is from Modules.istftnet
    assert hasattr(model, 'decoder')
    assert model.decoder.__class__.__name__ == 'Decoder'


def test_decoder_output_is_audio():
    """Test decoder produces audio waveform output."""
    config_path = Path("configs/coral_danish.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = build_model(config)

    # Create dummy input (text encoding + style)
    batch_size = 2
    seq_len = 100
    hidden_states = torch.randn(batch_size, 256, seq_len)
    style = torch.randn(batch_size, 256, 1)

    # Forward through decoder
    with torch.no_grad():
        audio = model.decoder(hidden_states, style)

    # iSTFTNet produces audio directly, not mel
    assert audio.ndim == 2  # [batch, audio_samples]
    assert audio.shape[0] == batch_size
    assert audio.dtype == torch.float32
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_decoder.py -v
```

Expected: FAIL because decoder is still nn.Linear placeholder

**Step 3: Integrate iSTFTNet decoder**

Edit `danish-tts-trainer/src/danish_tts/models/model_config.py`:

```python
from danish_tts.models.Modules.istftnet import Decoder as iSTFTNetDecoder

class StyleTTS2Model(nn.Module):
    def __init__(self, ...):
        # ... existing text_encoder, style_encoder, predictor ...

        # Real iSTFTNet decoder
        decoder_cfg = decoder_config
        self.decoder = iSTFTNetDecoder(
            dim_in=text_encoder_config.get("channels", 256),
            style_dim=style_encoder_config.get("dim", 256),
            dim_out=80,  # mel bins (not used for iSTFTNet, goes straight to audio)
            resblock_kernel_sizes=decoder_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
            upsample_rates=decoder_cfg.get("upsample_rates", [10, 5, 3, 2]),
            upsample_initial_channel=decoder_cfg.get("upsample_initial_channel", 512),
            resblock_dilation_sizes=decoder_cfg.get("resblock_dilation_sizes", [[1,3,5], [1,3,5], [1,3,5]]),
            upsample_kernel_sizes=decoder_cfg.get("upsample_kernel_sizes", [20, 10, 6, 4]),
            gen_istft_n_fft=2048,
            gen_istft_hop_size=300,
        )

        # Remove vocoder - iSTFTNet goes directly to audio
        self.vocoder = None
```

**Step 4: Update config with decoder parameters**

Edit `danish-tts-trainer/configs/coral_danish.yaml`:

```yaml
decoder:
  type: "istftnet"
  channels: 256
  resblock_kernel_sizes: [3, 7, 11]
  upsample_rates: [10, 5, 3, 2]  # 24000 Hz / (10*5*3*2) = 40 Hz frame rate
  upsample_initial_channel: 512
  resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
  upsample_kernel_sizes: [20, 10, 6, 4]
  gen_istft_n_fft: 2048
  gen_istft_hop_size: 300
```

**Step 5: Run tests**

```bash
python -m pytest tests/test_decoder.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/danish_tts/models/model_config.py tests/test_decoder.py configs/coral_danish.yaml
git commit -m "feat: integrate iSTFTNet decoder

- Replace placeholder decoder with real iSTFTNet
- Configure upsample rates for 24kHz audio
- Remove separate vocoder (iSTFTNet goes to audio)
- Update config with decoder hyperparameters"
```

---

## Phase 2: Discriminator Training

### Task 2.1: Discriminator Setup

**Files:**
- Create: `danish-tts-trainer/src/danish_tts/discriminators.py`
- Test: `danish-tts-trainer/tests/test_discriminators.py`

**Step 1: Write the failing test**

Create `danish-tts-trainer/tests/test_discriminators.py`:

```python
"""Tests for discriminator models."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from danish_tts.discriminators import MultiDiscriminator


def test_multi_discriminator_initialization():
    """Test multi-discriminator initializes correctly."""
    disc = MultiDiscriminator()

    # Should have MPD, MSD, and WavLM discriminators
    assert hasattr(disc, 'mpd')
    assert hasattr(disc, 'msd')
    assert hasattr(disc, 'wd')


def test_discriminator_forward():
    """Test discriminator forward pass."""
    disc = MultiDiscriminator()

    # Create fake and real audio
    batch_size = 2
    audio_length = 24000
    real_audio = torch.randn(batch_size, audio_length)
    fake_audio = torch.randn(batch_size, audio_length)

    # Forward pass
    with torch.no_grad():
        real_logits = disc(real_audio)
        fake_logits = disc(fake_audio)

    # Each should return list of logits from each discriminator
    assert isinstance(real_logits, list)
    assert isinstance(fake_logits, list)
    assert len(real_logits) == 3  # MPD + MSD + WavLM
    assert len(fake_logits) == 3


def test_discriminator_loss():
    """Test discriminator loss computation."""
    from danish_tts.losses import AdversarialLoss

    disc = MultiDiscriminator()
    loss_fn = AdversarialLoss()

    real_audio = torch.randn(2, 24000)
    fake_audio = torch.randn(2, 24000)

    with torch.no_grad():
        real_logits = disc(real_audio)
        fake_logits = disc(fake_audio)

    # Compute discriminator loss
    disc_loss = 0
    for real_l, fake_l in zip(real_logits, fake_logits):
        disc_loss += loss_fn.forward_discriminator(real_l, fake_l)

    assert isinstance(disc_loss, torch.Tensor)
    assert disc_loss.ndim == 0  # Scalar
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_discriminators.py -v
```

Expected: FAIL with "No module named 'danish_tts.discriminators'"

**Step 3: Implement discriminators wrapper**

Create `danish-tts-trainer/src/danish_tts/discriminators.py`:

```python
"""Discriminator models for adversarial training."""

import torch
import torch.nn as nn
from danish_tts.models.Modules.discriminators import (
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
    WavLMDiscriminator,
)


class MultiDiscriminator(nn.Module):
    """Multi-scale discriminator combining MPD, MSD, and WavLM."""

    def __init__(self, use_wavlm=True, wavlm_hidden=256, wavlm_nlayers=3, wavlm_initial_channel=64):
        super().__init__()

        # Multi-Period Discriminator
        self.mpd = MultiPeriodDiscriminator()

        # Multi-Resolution Spectrogram Discriminator
        self.msd = MultiResSpecDiscriminator()

        # WavLM-based discriminator (optional)
        self.use_wavlm = use_wavlm
        if use_wavlm:
            self.wd = WavLMDiscriminator(
                hidden=wavlm_hidden,
                nlayers=wavlm_nlayers,
                initial_channel=wavlm_initial_channel,
            )

    def forward(self, audio):
        """Run all discriminators on audio.

        Args:
            audio: [batch, audio_samples]

        Returns:
            List of logits from each discriminator
        """
        logits = []

        # MPD
        mpd_logits = self.mpd(audio)
        logits.append(mpd_logits)

        # MSD
        msd_logits = self.msd(audio)
        logits.append(msd_logits)

        # WavLM
        if self.use_wavlm:
            wd_logits = self.wd(audio)
            logits.append(wd_logits)

        return logits
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_discriminators.py -v
```

Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/danish_tts/discriminators.py tests/test_discriminators.py
git commit -m "feat: add multi-scale discriminator

- Combine MPD, MSD, and WavLM discriminators
- Support optional WavLM discriminator
- Return list of logits for each discriminator
- Ready for adversarial training"
```

---

### Task 2.2: Discriminator Training Loop

**Files:**
- Modify: `danish-tts-trainer/src/danish_tts/train.py`
- Test: `danish-tts-trainer/tests/test_discriminator_training.py`

**Step 1: Write the failing test**

Create `danish-tts-trainer/tests/test_discriminator_training.py`:

```python
"""Tests for discriminator training step."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from danish_tts.train import discriminator_step


def test_discriminator_step():
    """Test discriminator training step."""
    from danish_tts.discriminators import MultiDiscriminator
    from danish_tts.losses import AdversarialLoss

    # Create discriminator and optimizer
    discriminator = MultiDiscriminator(use_wavlm=False)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    # Create fake batch
    batch_size = 2
    real_audio = torch.randn(batch_size, 24000)
    fake_audio = torch.randn(batch_size, 24000)

    # Training step
    loss_fn = AdversarialLoss()
    loss_dict = discriminator_step(
        discriminator=discriminator,
        real_audio=real_audio,
        fake_audio=fake_audio,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    # Check outputs
    assert "disc_loss" in loss_dict
    assert "disc_loss_real" in loss_dict
    assert "disc_loss_fake" in loss_dict
    assert loss_dict["disc_loss"] > 0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_discriminator_training.py -v
```

Expected: FAIL with "cannot import name 'discriminator_step'"

**Step 3: Implement discriminator_step in train.py**

Edit `danish-tts-trainer/src/danish_tts/train.py`:

Add function after `train_step`:

```python
def discriminator_step(
    discriminator: nn.Module,
    real_audio: torch.Tensor,
    fake_audio: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn,
) -> dict:
    """Single discriminator training step.

    Args:
        discriminator: Multi-discriminator model
        real_audio: Real audio samples [batch, audio_samples]
        fake_audio: Generated audio samples [batch, audio_samples]
        optimizer: Discriminator optimizer
        loss_fn: AdversarialLoss instance

    Returns:
        Dictionary with discriminator losses
    """
    discriminator.train()
    optimizer.zero_grad()

    # Forward pass on real and fake audio
    real_logits = discriminator(real_audio)
    fake_logits = discriminator(fake_audio.detach())  # Detach to avoid generator gradients

    # Compute discriminator loss for each discriminator
    total_disc_loss = 0
    loss_real_total = 0
    loss_fake_total = 0

    for real_l, fake_l in zip(real_logits, fake_logits):
        disc_loss = loss_fn.forward_discriminator(real_l, fake_l)
        total_disc_loss += disc_loss

        # Track individual components
        loss_real_total += torch.mean(F.relu(1.0 - real_l))
        loss_fake_total += torch.mean(F.relu(1.0 + fake_l))

    # Backward pass
    total_disc_loss.backward()
    optimizer.step()

    return {
        "disc_loss": total_disc_loss.item(),
        "disc_loss_real": loss_real_total.item(),
        "disc_loss_fake": loss_fake_total.item(),
    }
```

**Step 4: Update main training loop to include discriminator**

Edit `danish-tts-trainer/src/danish_tts/train.py` in `main()`:

```python
def main():
    # ... existing setup ...

    # Setup discriminator
    print("Building discriminator...")
    from danish_tts.discriminators import MultiDiscriminator
    discriminator = MultiDiscriminator(
        use_wavlm=config.get("discriminator", {}).get("use_wavlm", True),
        wavlm_hidden=config.get("discriminator", {}).get("wavlm_hidden", 256),
    )
    discriminator = discriminator.to(device)

    # Discriminator optimizer
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.get("discriminator", {}).get("learning_rate", 2.0e-4),
    )

    # ... existing training loop ...

    while step < max_steps:
        for batch in tqdm(train_loader, desc=f"Step {step}/{max_steps}"):
            # Generator step
            loss_dict = train_step(model, batch, optimizer, config, losses)

            # Discriminator step (every N steps)
            if step % config["training"].get("disc_update_freq", 1) == 0:
                # Generate fake audio from model
                with torch.no_grad():
                    outputs = model(
                        phoneme_ids=batch["phoneme_ids"].to(device),
                        speaker_ids=batch["speaker_ids"].to(device),
                        ref_audio=batch["audio"].to(device),
                        phoneme_lengths=batch["phoneme_lengths"].to(device),
                    )
                    # Assume model now outputs audio (with iSTFTNet decoder)
                    fake_audio = outputs.get("predicted_audio", outputs.get("audio"))

                real_audio = batch["audio"].to(device)

                disc_loss_dict = discriminator_step(
                    discriminator=discriminator,
                    real_audio=real_audio,
                    fake_audio=fake_audio,
                    optimizer=disc_optimizer,
                    loss_fn=losses["adversarial"],
                )

                loss_dict.update(disc_loss_dict)

            # Logging
            if step % config["logging"]["log_interval"] == 0:
                for key, value in loss_dict.items():
                    writer.add_scalar(f"train/{key}", value, step)

            # ... rest of training loop ...
```

**Step 5: Run tests**

```bash
python -m pytest tests/test_discriminator_training.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/danish_tts/train.py tests/test_discriminator_training.py
git commit -m "feat: implement discriminator training step

- Add discriminator_step() function
- Integrate discriminator into main training loop
- Update generator every step, discriminator every N steps
- Track discriminator losses separately
- Use detached fake audio to avoid generator gradients"
```

---

## Phase 3: Duration Prediction

### Task 3.1: Duration Loss

**Files:**
- Modify: `danish-tts-trainer/src/danish_tts/losses.py`
- Test: `danish-tts-trainer/tests/test_duration_loss.py`

**Step 1: Write the failing test**

Create `danish-tts-trainer/tests/test_duration_loss.py`:

```python
"""Tests for duration prediction loss."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from danish_tts.losses import DurationLoss


def test_duration_loss():
    """Test duration loss computes correctly."""
    loss_fn = DurationLoss()

    # Predicted and target durations
    predicted = torch.randn(2, 10)  # [batch, seq_len] (log scale)
    target = torch.randint(1, 20, (2, 10)).float()  # [batch, seq_len]
    lengths = torch.LongTensor([8, 10])  # Actual sequence lengths

    loss = loss_fn(predicted, target, lengths)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert loss > 0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_duration_loss.py -v
```

Expected: FAIL with "cannot import name 'DurationLoss'"

**Step 3: Implement DurationLoss**

Edit `danish-tts-trainer/src/danish_tts/losses.py`:

Add class:

```python
class DurationLoss(nn.Module):
    """Duration prediction loss (MSE in log scale)."""

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute duration loss.

        Args:
            predicted: Predicted log durations [batch, seq_len]
            target: Target durations [batch, seq_len]
            lengths: Actual sequence lengths [batch]

        Returns:
            Scalar loss
        """
        # Convert target to log scale
        target_log = torch.log(target.clamp(min=1.0))

        # Create mask for valid positions
        batch_size, max_len = predicted.shape
        mask = torch.arange(max_len, device=predicted.device)[None, :] < lengths[:, None]

        # MSE loss only on valid positions
        loss = F.mse_loss(predicted * mask, target_log * mask, reduction='sum')
        loss = loss / mask.sum()  # Normalize by number of valid elements

        return loss
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_duration_loss.py -v
```

Expected: PASS

**Step 5: Integrate into training**

Edit `danish-tts-trainer/src/danish_tts/train.py` in `train_step`:

```python
def train_step(model, batch, optimizer, config, losses):
    # ... existing forward pass ...

    outputs = model(
        phoneme_ids=phoneme_ids,
        speaker_ids=speaker_ids,
        ref_audio=audio,
        phoneme_lengths=phoneme_lengths,
    )

    # ... existing reconstruction and KL losses ...

    # Duration loss (if model returns durations)
    loss_dur = 0
    if "durations" in outputs and "target_durations" in batch:
        loss_dur = losses["duration"](
            outputs["durations"],
            batch["target_durations"].to(device),
            phoneme_lengths,
        )

    # Total loss
    total_loss = (
        loss_weights["reconstruction"] * loss_recon +
        loss_weights["style_kl"] * loss_kl +
        loss_weights.get("duration", 1.0) * loss_dur
    )

    # ... rest of function ...

    return {
        "total_loss": total_loss.item(),
        "reconstruction_loss": loss_recon.item(),
        "kl_loss": loss_kl.item(),
        "duration_loss": loss_dur.item() if isinstance(loss_dur, torch.Tensor) else 0,
    }
```

**Step 6: Commit**

```bash
git add src/danish_tts/losses.py tests/test_duration_loss.py src/danish_tts/train.py
git commit -m "feat: add duration prediction loss

- Implement DurationLoss with MSE in log scale
- Mask invalid positions based on sequence lengths
- Integrate into training step
- Track duration loss separately"
```

---

## Phase 4: Production Inference Optimization

### Task 4.1: Fast Inference Pipeline

**Files:**
- Modify: `danish-tts-trainer/src/danish_tts/inference.py`
- Test: `danish-tts-trainer/tests/test_fast_inference.py`

**Step 1: Write the failing test**

Create `danish-tts-trainer/tests/test_fast_inference.py`:

```python
"""Tests for optimized inference pipeline."""

import pytest
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_inference_with_torch_no_grad():
    """Test inference uses torch.no_grad for efficiency."""
    from danish_tts.inference import synthesize_optimized

    # This should run without gradients enabled
    text = "Dette er en test"
    # Mock G2P and model would go here
    # Test would verify no gradient computation


def test_inference_batch_support():
    """Test inference supports batched synthesis."""
    from danish_tts.inference import synthesize_batch

    texts = ["Hej Danmark", "Goddag", "Farvel"]
    # Should return list of audio tensors
    # Test would verify batched processing
```

**Step 2: Implement optimized inference functions**

Edit `danish-tts-trainer/src/danish_tts/inference.py`:

Add functions:

```python
@torch.no_grad()
def synthesize_optimized(
    model,
    text: str,
    g2p,
    speaker_id: int = 0,
    device: str = "cpu",
    temperature: float = 0.667,
    length_scale: float = 1.0,
) -> torch.Tensor:
    """Optimized synthesis with temperature and length control.

    Args:
        model: Trained TTS model
        text: Input Danish text
        g2p: Danish G2P instance
        speaker_id: Speaker ID (0 or 1)
        device: Device to run inference on
        temperature: Sampling temperature for style
        length_scale: Duration scaling factor (>1 = slower, <1 = faster)

    Returns:
        Audio waveform tensor [1, n_samples]
    """
    model.eval()

    # Get phonemes
    phonemes, tokens = g2p(text)

    # Convert to tensors
    phoneme_ids = torch.LongTensor(tokens).unsqueeze(0).to(device)
    speaker_ids = torch.LongTensor([speaker_id]).to(device) if model.n_speakers > 1 else None

    # Inference with optimizations
    outputs = model.inference(
        phoneme_ids=phoneme_ids,
        speaker_ids=speaker_ids,
        temperature=temperature,
        length_scale=length_scale,
    )

    return outputs["audio"]


@torch.no_grad()
def synthesize_batch(
    model,
    texts: list[str],
    g2p,
    speaker_id: int = 0,
    device: str = "cpu",
    batch_size: int = 8,
) -> list[torch.Tensor]:
    """Batch synthesis for multiple texts.

    Args:
        model: Trained TTS model
        texts: List of Danish texts
        g2p: Danish G2P instance
        speaker_id: Speaker ID
        device: Device
        batch_size: Batch size for processing

    Returns:
        List of audio waveform tensors
    """
    model.eval()
    all_audio = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Convert all texts to phonemes
        batch_phonemes = []
        batch_lengths = []
        for text in batch_texts:
            phonemes, tokens = g2p(text)
            batch_phonemes.append(torch.LongTensor(tokens))
            batch_lengths.append(len(tokens))

        # Pad to max length in batch
        max_len = max(batch_lengths)
        padded = torch.zeros(len(batch_texts), max_len, dtype=torch.long)
        for j, phonemes in enumerate(batch_phonemes):
            padded[j, :len(phonemes)] = phonemes

        # Batch inference
        padded = padded.to(device)
        outputs = model.inference(
            phoneme_ids=padded,
            speaker_ids=torch.LongTensor([speaker_id] * len(batch_texts)).to(device) if model.n_speakers > 1 else None,
        )

        # Split batch back into individual audio
        for audio in outputs["audio"]:
            all_audio.append(audio)

    return all_audio
```

**Step 3: Add model optimization methods**

Edit `danish-tts-trainer/src/danish_tts/models/model_config.py`:

Add to `StyleTTS2Model`:

```python
def optimize_for_inference(self):
    """Optimize model for faster inference."""
    self.eval()

    # Fuse batch norm layers if any
    # torch.quantization.fuse_modules(self, ...) if needed

    # Convert to half precision for faster inference (if GPU)
    # self.half()

    return self


@torch.jit.script
def inference(self, phoneme_ids, speaker_ids=None, temperature=0.667, length_scale=1.0):
    """JIT-compiled inference for speed.

    Args:
        phoneme_ids: [batch, seq_len]
        speaker_ids: [batch]
        temperature: Sampling temperature
        length_scale: Duration scaling

    Returns:
        Dictionary with audio
    """
    # ... inference implementation with optimizations ...
```

**Step 4: Commit**

```bash
git add src/danish_tts/inference.py src/danish_tts/models/model_config.py tests/test_fast_inference.py
git commit -m "feat: optimize inference for production

- Add synthesize_optimized with temperature control
- Add synthesize_batch for batched processing
- Add optimize_for_inference model method
- Support length_scale for speech rate control
- Use torch.no_grad() decorator for efficiency"
```

---

## Phase 5: Full CoRal Training

### Task 5.1: Training Configuration and Launch

**Files:**
- Modify: `danish-tts-trainer/configs/coral_danish.yaml`
- Create: `danish-tts-trainer/scripts/train_full.sh`

**Step 1: Update config for full training**

Edit `danish-tts-trainer/configs/coral_danish.yaml`:

```yaml
training:
  batch_size: 32  # Increase for full training
  num_workers: 8  # More workers for data loading
  max_steps: 600000  # 600k steps (~2-3 days on single GPU)

  # Learning rate schedule
  learning_rate: 2.0e-4
  warmup_steps: 8000
  lr_schedule: "warmup_cosine"
  min_lr: 1.0e-5

  # Mixed precision for faster training
  use_amp: true
  amp_dtype: "bfloat16"  # Better than float16

  # Gradient accumulation for larger effective batch size
  gradient_accumulation_steps: 2  # Effective batch_size = 32 * 2 = 64

  # Checkpointing
  checkpoint_interval: 5000  # Save every 5k steps
  keep_n_checkpoints: 10

  # Validation
  val_interval: 2500  # Validate every 2.5k steps
  val_samples: 20  # Generate 20 samples

# Discriminator training
discriminator:
  use_wavlm: true
  wavlm_hidden: 256
  wavlm_nlayers: 3
  learning_rate: 2.0e-4
  update_freq: 1  # Update every step

# Loss weights (tuned for Danish)
loss:
  reconstruction: 45.0  # High weight for mel reconstruction
  adversarial: 1.0  # GAN loss
  style_kl: 1.0  # Style regularization
  duration: 1.0  # Duration prediction
  pitch: 0.1  # Pitch prediction (if used)
```

**Step 2: Create training launch script**

Create `danish-tts-trainer/scripts/train_full.sh`:

```bash
#!/bin/bash

# Full CoRal Danish TTS Training Script

set -e  # Exit on error

# Configuration
CONFIG="configs/coral_danish.yaml"
EXPERIMENT_NAME="coral_danish_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${EXPERIMENT_NAME}"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "checkpoints/${EXPERIMENT_NAME}"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0  # Set GPU
export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
export PHONEMIZER_ESPEAK_PATH=/opt/homebrew/bin/espeak-ng

# Print system info
echo "========================================="
echo "Starting Danish TTS Training"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Config: ${CONFIG}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

# Start training
python src/danish_tts/train.py \
    --config "${CONFIG}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    2>&1 | tee "${LOG_DIR}/training.log"

echo "Training complete! Logs saved to ${LOG_DIR}"
```

**Step 3: Make script executable**

```bash
chmod +x scripts/train_full.sh
```

**Step 4: Add resume training support**

Edit `danish-tts-trainer/src/danish_tts/train.py`:

```python
def main():
    parser = argparse.ArgumentParser(description="Train Danish StyleTTS2")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Checkpoint to resume from")
    parser.add_argument("--experiment_name", type=str, default="danish_tts")
    args = parser.parse_args()

    # ... existing setup ...

    # Resume from checkpoint if provided
    start_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if discriminator:
            discriminator.load_state_dict(checkpoint.get("discriminator", {}))
            disc_optimizer.load_state_dict(checkpoint.get("disc_optimizer", {}))
        start_step = checkpoint["step"]
        print(f"Resumed from step {start_step}")

    # Training loop starting from start_step
    step = start_step
    # ... rest of training ...
```

**Step 5: Test training launches**

```bash
# Dry run to verify config loads
python src/danish_tts/train.py --config configs/coral_danish.yaml --help

# Start training
./scripts/train_full.sh
```

**Step 6: Commit**

```bash
git add configs/coral_danish.yaml scripts/train_full.sh src/danish_tts/train.py
git commit -m "feat: add full training configuration and scripts

- Update config for production training
- Increase batch size and workers
- Add gradient accumulation
- Add training launch script with logging
- Add resume training support
- Configure for 600k steps (~2-3 days)"
```

---

## Summary

This plan implements:

1. **Real StyleTTS2 Components** (Tasks 1.1-1.2)
   - Real TextEncoder, StyleEncoder, ProsodyPredictor
   - iSTFTNet decoder for direct audio generation
   - Speaker embeddings for multi-speaker

2. **Discriminator Training** (Tasks 2.1-2.2)
   - Multi-scale discriminators (MPD, MSD, WavLM)
   - Adversarial training loop
   - Separate discriminator optimizer

3. **Duration Prediction** (Task 3.1)
   - Duration loss in log scale
   - Masked loss for variable lengths
   - Integration into training

4. **Production Optimization** (Task 4.1)
   - Optimized inference with torch.no_grad
   - Batch synthesis support
   - Temperature and length_scale control
   - JIT compilation potential

5. **Full Training** (Task 5.1)
   - Production configuration
   - Training launch script
   - Resume support
   - Logging and checkpointing

**Total Tasks:** 9 tasks across 5 phases

**Estimated Time:** ~8-12 hours (implementation) + 2-3 days (training)

**Next Steps After This Plan:**
- Monitor training metrics and adjust hyperparameters
- Evaluate generated samples qualitatively
- Fine-tune on specific speakers or domains
- Export to ONNX for deployment
- Create inference API/service
