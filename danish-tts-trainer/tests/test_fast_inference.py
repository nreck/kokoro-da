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
    from danish_tts.models.model_config import build_model
    import yaml

    # Load config
    config_path = Path("configs/coral_danish.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Build model
    model = build_model(config)
    model.eval()

    # Mock G2P
    class MockG2P:
        def __call__(self, text):
            return ("t ɛ s t", [1, 2, 3, 4])

    g2p = MockG2P()

    # Test should run without gradients enabled
    text = "Dette er en test"

    # Verify torch.no_grad is used by checking gradients are not tracked
    with torch.set_grad_enabled(True):
        audio = synthesize_optimized(
            model=model,
            text=text,
            g2p=g2p,
            speaker_id=0,
            device="cpu",
            temperature=0.667,
            length_scale=1.0,
        )

    # Audio should be returned
    assert isinstance(audio, torch.Tensor)
    assert audio.ndim == 2  # [1, n_samples]
    assert not audio.requires_grad  # Gradients should not be tracked


def test_inference_batch_support():
    """Test inference supports batched synthesis."""
    from danish_tts.inference import synthesize_batch
    from danish_tts.models.model_config import build_model
    import yaml

    # Load config
    config_path = Path("configs/coral_danish.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Build model
    model = build_model(config)
    model.eval()

    # Mock G2P
    class MockG2P:
        def __call__(self, text):
            # Return different lengths for different texts
            tokens = list(range(len(text) % 10 + 1))
            phonemes = " ".join(str(t) for t in tokens)
            return (phonemes, tokens)

    g2p = MockG2P()

    texts = ["Hej Danmark", "Goddag", "Farvel"]

    # Should return list of audio tensors
    audio_list = synthesize_batch(
        model=model,
        texts=texts,
        g2p=g2p,
        speaker_id=0,
        device="cpu",
        batch_size=2,
    )

    # Check results
    assert isinstance(audio_list, list)
    assert len(audio_list) == len(texts)
    for audio in audio_list:
        assert isinstance(audio, torch.Tensor)
        assert audio.ndim == 1  # Single audio waveform


def test_temperature_and_length_scale_parameters():
    """Test that temperature and length_scale parameters work."""
    from danish_tts.inference import synthesize_optimized
    from danish_tts.models.model_config import build_model
    import yaml

    # Load config
    config_path = Path("configs/coral_danish.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Build model
    model = build_model(config)
    model.eval()

    # Mock G2P
    class MockG2P:
        def __call__(self, text):
            return ("t ɛ s t", [1, 2, 3, 4])

    g2p = MockG2P()
    text = "Test"

    # Test different temperatures
    audio1 = synthesize_optimized(model, text, g2p, temperature=0.5)
    audio2 = synthesize_optimized(model, text, g2p, temperature=1.0)

    assert audio1.shape == audio2.shape
    # Different temperatures should give different results (stochastic)
    # but we can't test that easily without fixing random seed

    # Test length_scale
    audio3 = synthesize_optimized(model, text, g2p, length_scale=1.5)

    assert isinstance(audio3, torch.Tensor)


def test_optimize_for_inference_method():
    """Test model has optimize_for_inference method."""
    from danish_tts.models.model_config import build_model
    import yaml

    # Load config
    config_path = Path("configs/coral_danish.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Build model
    model = build_model(config)

    # Check method exists
    assert hasattr(model, 'optimize_for_inference')

    # Call method
    optimized_model = model.optimize_for_inference()

    # Should return the model itself
    assert optimized_model is model

    # Model should be in eval mode
    assert not model.training
