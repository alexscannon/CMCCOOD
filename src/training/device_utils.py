# src/training/device_utils.py
import torch
import os
from typing import Union, Dict, Any, Tuple

def get_device(device_str: str = None) -> torch.device:
    """Get the appropriate device based on availability"""
    if device_str is None:
        device_str = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = 'cpu'

    return torch.device(device_str)

def to_device(data: Union[torch.Tensor, Dict, list, tuple], device: torch.device) -> Any:
    """Move data to device, handling different data types"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    else:
        return data

def get_training_config(device: torch.device) -> Dict[str, Any]:
    """Get device-specific training configuration"""
    config = {
        'device': device,
        'use_mixed_precision': device.type == 'cuda',  # Only use mixed precision on CUDA
        'gradient_accumulation_steps': 1,
    }

    # Adjust batch size based on device
    if device.type == 'cuda':
        config['batch_size'] = 128
        # Check if CUDA device has limited memory
        if torch.cuda.get_device_properties(0).total_memory < 10e9:  # Less than 10GB
            config['batch_size'] = 64
            config['gradient_accumulation_steps'] = 2
    else:
        config['batch_size'] = 32
        config['gradient_accumulation_steps'] = 4

    return config