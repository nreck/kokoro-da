#!/usr/bin/env python
"""Manual test runner for TTS dataset tests."""

import os
import sys
from pathlib import Path

# Set espeak environment variables for phonemizer
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/lib/libespeak-ng.dylib'
os.environ['PHONEMIZER_ESPEAK_PATH'] = '/opt/homebrew/bin/espeak-ng'

# Add paths
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')
sys.path.insert(0, str(Path(__file__).parent.parent / 'misaki'))

# Import the test functions
from test_tts_dataset import (
    test_tts_dataset_initialization,
    test_tts_dataset_getitem_returns_correct_format,
    test_tts_dataset_speaker_mapping
)

# Run the tests
print('='*60)
print('Running TTS Dataset Tests')
print('='*60)

try:
    print('\n1. Running test_tts_dataset_initialization...')
    test_tts_dataset_initialization()
    print('   ✓ test_tts_dataset_initialization PASSED')
except Exception as e:
    print(f'   ✗ test_tts_dataset_initialization FAILED: {e}')
    import traceback
    traceback.print_exc()

try:
    print('\n2. Running test_tts_dataset_getitem_returns_correct_format...')
    test_tts_dataset_getitem_returns_correct_format()
    print('   ✓ test_tts_dataset_getitem_returns_correct_format PASSED')
except Exception as e:
    print(f'   ✗ test_tts_dataset_getitem_returns_correct_format FAILED: {e}')
    import traceback
    traceback.print_exc()

try:
    print('\n3. Running test_tts_dataset_speaker_mapping...')
    test_tts_dataset_speaker_mapping()
    print('   ✓ test_tts_dataset_speaker_mapping PASSED')
except Exception as e:
    print(f'   ✗ test_tts_dataset_speaker_mapping FAILED: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '='*60)
print('All tests completed')
print('='*60)
