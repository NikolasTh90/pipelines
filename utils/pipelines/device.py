import os
import sys

def set_device(use_gpu=False):
    """
    Set the device for models with fallback handling
    
    Args:
        use_gpu: Whether to use GPU if available
    """
    if use_gpu:
        try:
            # Check if CUDA is available
            import torch
            if torch.cuda.is_available():
                os.environ["DEVICE"] = "cuda"
                print("Set device to use GPU (CUDA)")
                # Check memory to warn about potential issues
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                free_memory_gb = free_memory / (1024**3)
                if free_memory_gb < 2.0:  # Less than 2GB free
                    print(f"Low GPU memory detected ({free_memory_gb:.2f} GB free). This may cause issues.")
            else:
                print("CUDA not available. Falling back to CPU.")
                os.environ["DEVICE"] = "cpu"
        except Exception as e:
            print(f"Error checking GPU: {e}. Falling back to CPU.")
            os.environ["DEVICE"] = "cpu"
    else:
        os.environ["DEVICE"] = "cpu" 
        print("Set device to use CPU")

def clear_gpu_memory():
    """
    Clear GPU memory when using CUDA
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Report memory stats for debugging
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            return {"allocated": allocated, "reserved": reserved}
    except Exception as e:
        print(f"Failed to clear GPU memory: {e}")
    return None

def print_environment_info(logger):
    """
    Print diagnostic information about the environment
    
    Args:
        logger: Logger to use for output
    """
    logger.info("=== Environment Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    logger.info(f"DEVICE: {os.environ.get('DEVICE', 'Not set')}")
    logger.info(f"sys.path: {sys.path}")
    
    # Check GPU availability if using GPU
    if os.environ.get('DEVICE') == 'cuda':
        try:
            import torch
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                # Print memory information
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                logger.info(f"GPU free memory: {free_memory / (1024**3):.2f} GB")
        except ImportError:
            logger.warning("Torch not available - cannot check CUDA status")
    
    logger.info("===============================")
