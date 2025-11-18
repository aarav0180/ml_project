"""Device detection and setup utilities."""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device (CUDA or CPU) for training/inference.
    
    Args:
        use_cuda: Whether to use CUDA if available
        
    Returns:
        torch.device: The device to use
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✓ CUDA available!")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        if use_cuda:
            logger.warning("⚠ CUDA not available. Using CPU (training will be slower)")
    
    logger.info(f"Using device: {device}")
    return device

