from setuptools import setup, find_packages

setup(
    name="CMCCOOD",
    version="0.1.0",
    description="Research project on multi-class classification in continual learning with OOD detection using foundation models",
    author="Alexander Cannon",
    author_email="acannon37@gatech.edu",
    url="https://github.com/alexscannon/CMCCOOD",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "torch>=1.9.0",
        "torchvision>=0.10.0",

        # Configuration management
        "hydra-core>=1.1.0",
        "omegaconf>=2.1.0",

        # Foundation model access
        "timm>=0.5.4",  # For vision foundation models

        # Data processing
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "pillow>=8.2.0",
        "matplotlib>=3.4.0",

        # Progress and logging
        "tqdm>=4.61.0",
        "wandb>=0.12.0",  # Weights & Biases integration

        # Evaluation metrics
        "scikit-learn>=0.24.0",

        # Utilities
        "pyyaml>=5.4.0",
        "jsonschema>=3.2.0",
    ],
    entry_points={
        "console_scripts": [
            # Command-line entry points
            "omccood-train=src.main:main",                       # Main training script
            "omccood-prepare-data=scripts.prepare_imagenet:main", # Data preparation script
            "omccood-evaluate=scripts.evaluate:main",             # Evaluation script
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json"],  # Include all YAML and JSON files
    },
)