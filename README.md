# Continual Multi-class classification Out-of-Distribution Detection Repository

A comprehensive framework for researching multi-class classification in continual learning settings using foundation models with out-of-distribution detection capabilities.

## Overview

This research framework addresses the challenging problem of continual learning, where models must adapt to new classes over time while preserving performance on previously learned classes. It focuses on class-incremental learning scenarios with the ability to detect out-of-distribution examples, using foundation models as the backbone architecture.

Key research areas addressed by this framework:
- **Catastrophic Forgetting**: Preventing models from forgetting previously learned knowledge
- **Knowledge Transfer**: Leveraging prior knowledge to learn new tasks more efficiently
- **Out-of-Distribution Detection**: Identifying examples that don't belong to any learned class
- **Foundation Model Adaptation**: Utilizing pre-trained vision models for continual learning

## Features

- **Modular Architecture**: Easily swap foundation models, continual learning methods, and datasets
- **Hydra Configuration System**: Flexible experiment configuration without code changes
- **Multiple Continual Learning Approaches**:
  - Regularization-based methods (EWC, SI, LwF)
  - Replay-based methods (Experience Replay, DER)
  - Parameter isolation methods (PackNet, HAT, Adapters)
- **OOD Detection**: Support for multiple OOD detection strategies
- **Experiment Tracking**: Full integration with Weights & Biases
- **Device-Agnostic**: Run on different hardware setups with automatic configuration
- **Comprehensive Metrics**: Track forgetting, forward/backward transfer, and OOD performance

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended but not required)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/alexscannon/cmccood.git
   cd cmccood
   ```

2. Create a virtual environment (recommended):
   ```bash
   conda create --name <env_name>
   conda activate <env_name>
   ```

3. Install the package and dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Set up Weights & Biases:
   ```bash
   wandb login
   ```

## Project Structure

```
continual_learning_research/
├── configs/                 # Hydra configuration files
├── src/                     # Source code
│   ├── models/              # Foundation models and adaptations
│   ├── datasets/            # Dataset implementations
│   ├── continual_learning/  # Continual learning methods
│   ├── training/            # Training utilities
│   ├── utils/               # Helper functions
│   └── main.py              # Main entry point
├── scripts/                 # Utility scripts
├── notebooks/               # Analysis notebooks
├── tests/                   # Unit tests
└── data/                    # Data directory
```

## Usage

### Data Preparation

Prepare the Split-ImageNet dataset:

```bash
cmccood-prepare-data --data_dir ./data
```

### Training

Train a model with a specific configuration:

```bash
cmccood-train model=vit dataset=split_imagenet continual_learning.method=replay/er
```

Modify any configuration parameter using Hydra's override syntax:

```bash
cmccood-train model=vit dataset=split_imagenet \
  continual_learning.method=replay/er \
  training.optimizer.lr=0.001 \
  training.batch_size=64
```

### Evaluation

Evaluate a trained model:

```bash
cmccood-evaluate checkpoint_path=outputs/2025-03-14/12-34-56/checkpoints/final.pt
```

## Configuration with Hydra

This project uses Hydra for configuration management. The main configuration is in `configs/config.yaml` with component-specific configurations in subdirectories.

Example configuration structure:

```yaml
# configs/config.yaml
defaults:
  - model: vit
  - dataset: split_imagenet
  - continual_learning/scenario: class_incremental
  - continual_learning/method: replay/er
  - continual_learning/ood_detection: energy
  - training: default
  - logger: wandb
  - _self_

# General parameters
seed: 42
device: cuda
...
```

## Extending the Framework

### Adding a New Continual Learning Method

1. Create a new configuration file in `configs/continual_learning/method/`
2. Implement the method in `src/continual_learning/methods/`
3. Register it in the appropriate registry

Example implementation:

```python
# src/continual_learning/methods/your_method.py
from src.continual_learning.base import BaseCLMethod

class YourMethod(BaseCLMethod):
    def __init__(self, model, scenario, **kwargs):
        super().__init__(model, scenario)
        # Your initialization code here

    def before_task(self, task_id):
        # Preparation before learning a new task
        pass

    def after_task(self, task_id):
        # Consolidation after learning a task
        pass

    def compute_loss(self, outputs, targets, task_id):
        # Compute task-specific loss with regularization
        pass
```

### Adding a New Foundation Model

1. Create a configuration file in `configs/model/`
2. Implement the adapter in `src/models/foundation_models/`

## Metrics and Evaluation

The framework tracks the following metrics during training and evaluation:

- **Task Accuracy**: Per-task accuracy after each training phase
- **Average Accuracy**: Overall accuracy across all seen tasks
- **Forgetting**: Difference in performance on previous tasks
- **Forward Transfer**: Performance improvement on future tasks
- **Backward Transfer**: Performance change on previous tasks
- **OOD Detection**: AUROC, FPR@95TPR for out-of-distribution detection

## Contributing

Contributions are welcome! Please check the [contributing guidelines](CONTRIBUTING.md) for more information.

To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-amazing-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This research framework incorporates ideas from various continual learning papers and codebases
- The foundation model integration builds on top of the TIMM library