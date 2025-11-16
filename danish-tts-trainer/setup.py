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
