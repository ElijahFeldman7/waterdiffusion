from setuptools import setup, find_packages

setup(
    name="waterdiffusion",
    version="0.1.0",
    description="Watermarking for Diffusion Language Models",
    packages=find_packages(include=["models", "watermarks", "scripts"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "datasets>=2.14.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
)
