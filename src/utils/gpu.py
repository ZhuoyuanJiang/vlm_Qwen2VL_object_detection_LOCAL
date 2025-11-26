"""
GPU Management Utilities for Qwen2-VL Training

QUICK REFERENCE:
- WHEN: Run at the very beginning before importing torch
- PURPOSE: Automatically find and configure available GPUs
- CRITICAL: Sets CUDA_VISIBLE_DEVICES before torch import
"""

import os
import sys
import subprocess


def find_available_gpus(num_gpus_needed=1):
    """
    Find available GPUs by checking memory usage.

    WHEN: Called at startup to detect available GPUs

    INPUT:
        num_gpus_needed (int): Number of GPUs required (default: 1)

    OUTPUT:
        list: GPU indices that are available (e.g., [0, 1])

    AVAILABILITY CRITERIA:
        - GPU memory used < 2GB
        - GPU total memory > 40GB
        - Returns EXACTLY the number requested (not more!)

    Example:
        >>> available = find_available_gpus(num_gpus_needed=2)
        >>> print(available)  # [0, 1] or [2, 3] etc.

    Note:
        This function is conservative to avoid occupying all lab resources.
        It only returns the exact number of GPUs requested.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )

        available = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            gpu_idx = int(parts[0])
            mem_used = int(parts[1])
            mem_total = int(parts[2])

            # Consider GPU available if <2GB used and has at least 40GB total
            if mem_used < 2000 and mem_total > 40000:
                available.append(gpu_idx)

        # Return exactly the number of GPUs requested (not more!)
        # IMPORTANT: This ensures we only use the requested number of GPUs
        # to avoid occupying all lab resources
        if len(available) >= num_gpus_needed:
            return available[:num_gpus_needed]  # Take ONLY what we need
        else:
            return available  # Return what we have (might be less)

    except Exception as e:
        print(f"Error checking GPUs: {e}")
        return []


def find_best_fallback_gpu():
    """
    Find the GPU with most free memory as fallback.

    WHEN: Called when no GPUs meet the standard criteria

    OUTPUT:
        tuple: (gpu_idx, free_memory_mb) or (None, 0) if no GPU found

    FALLBACK CRITERIA:
        - At least 20GB free memory
        - Returns the GPU with most free memory
    """
    best_gpu = None
    best_free_mem = 0

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )

        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            gpu_idx = int(parts[0])
            mem_used = int(parts[1])
            mem_total = int(parts[2])
            free_mem = mem_total - mem_used

            print(f"   GPU {gpu_idx}: {free_mem/1024:.1f} GB free")

            if free_mem > best_free_mem and free_mem > 20000:  # At least 20GB free
                best_gpu = gpu_idx
                best_free_mem = free_mem
    except:
        pass

    return best_gpu, best_free_mem


def setup_gpus(use_dual_gpu=True, max_gpus=2, verbose=True):
    """
    Automatically configure GPUs for training.

    WHEN: Call this BEFORE importing torch!

    INPUT:
        use_dual_gpu (bool): Whether to prefer 2 GPUs (default: True)
        max_gpus (int): Maximum number of GPUs to use (default: 2)
        verbose (bool): Whether to print configuration info (default: True)

    OUTPUT:
        None (sets environment variables and imports torch)

    SIDE EFFECTS:
        - Sets os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        - Sets os.environ["CUDA_VISIBLE_DEVICES"]
        - Imports torch
        - Exits program if no suitable GPU found

    Example Usage:
        >>> # At the very beginning of your script:
        >>> from src.utils.gpu import setup_gpus
        >>> setup_gpus(use_dual_gpu=True, max_gpus=2)
        >>> # Now you can safely import other modules that use torch

    Why This Matters:
        - CUDA_VISIBLE_DEVICES must be set BEFORE importing torch
        - Ensures reproducible GPU selection
        - Prevents accidentally using all GPUs in shared lab environment
    """
    # Set memory management for better GPU utilization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Determine how many GPUs to use (enforce MAX_GPUS limit)
    num_gpus = 2 if use_dual_gpu else 1
    num_gpus = min(num_gpus, max_gpus)  # Never exceed MAX_GPUS

    if verbose:
        print("=" * 60)
        print("🖥️  AUTOMATIC GPU CONFIGURATION")
        print("=" * 60)
        mode = 'DUAL GPU (like original paper)' if use_dual_gpu else 'SINGLE GPU (test mode)'
        print(f"Configuration: {mode}")
        print(f"Looking for {num_gpus} available GPU(s) (max {max_gpus} to preserve lab resources)...")

    # Find available GPUs
    available_gpus = find_available_gpus(num_gpus)

    if len(available_gpus) > 0:
        # Ensure we never use more than MAX_GPUS even if more are found
        available_gpus = available_gpus[:max_gpus]
        gpu_string = ','.join(str(g) for g in available_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_string

        if verbose:
            print(f"✅ Found {len(available_gpus)} available GPU(s): {available_gpus}")
            if use_dual_gpu and len(available_gpus) == 1:
                print("⚠️ Warning: Only 1 GPU available, but 2 were requested")
    else:
        # Try to find ANY available GPU as fallback
        if verbose:
            print("⚠️ No free GPUs found with standard criteria, checking all GPUs...")

        best_gpu, best_free_mem = find_best_fallback_gpu()

        if best_gpu is not None:
            # For fallback, still respect the MAX_GPUS limit (use only 1 GPU)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
            if verbose:
                print(f"✅ Using GPU {best_gpu} with {best_free_mem/1024:.1f} GB free memory")
                print(f"   (Using single GPU in fallback mode to preserve lab resources)")
        else:
            print("❌ No GPUs available with sufficient memory!")
            print("   Please free up GPU memory or wait for GPUs to become available")
            sys.exit(1)

    # Verify configuration
    if verbose:
        print(f"\nCUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    # Import torch AFTER setting CUDA_VISIBLE_DEVICES
    import torch

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        total_memory = 0

        if verbose:
            print(f"PyTorch sees {device_count} GPU(s):")

        # Verify we're not using more than MAX_GPUS
        if device_count > max_gpus:
            print(f"⚠️ ERROR: PyTorch sees {device_count} GPUs but MAX_GPUS={max_gpus}")
            print("   This should not happen - CUDA_VISIBLE_DEVICES may not be set correctly")

        for i in range(device_count):
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            total_memory += mem_gb
            if verbose:
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_gb:.1f} GB)")

        if verbose:
            print(f"\nTotal GPU memory available: {total_memory:.1f} GB")
            print(f"\n✅ CONFIRMED: Using exactly {device_count} GPU(s) (max allowed: {max_gpus})")

            if device_count > 1:
                print("\n📝 Multi-GPU Training Notes:")
                print("  - Model will automatically use DataParallel")
                print("  - Effective batch size = per_device_batch_size × num_gpus")
            print("=" * 60)
    else:
        print("❌ CUDA not available! Exiting...")
        sys.exit(1)


def clear_memory(verbose: bool = True):
    """
    Clear GPU memory and run garbage collection.

    This function:
    1. Deletes common training variables from global scope
    2. Runs Python garbage collection
    3. Empties CUDA cache
    4. Synchronizes CUDA operations

    Args:
        verbose: Print memory usage after clearing (default: True)

    Usage:
        >>> from src.utils.gpu import clear_memory
        >>> clear_memory()
        GPU allocated memory: 0.12 GB
        GPU reserved memory: 0.50 GB

    Note:
        Call this between training runs or when switching models
        to free up GPU memory. The sleep intervals help ensure
        memory is fully released.
    """
    import gc
    import time
    import torch

    # Delete variables if they exist in the current global scope
    # Note: This only works for variables in the caller's global scope
    # if called from the main script
    global_vars_to_clear = ['inputs', 'model', 'processor', 'trainer', 'peft_model', 'bnb_config']

    import inspect
    caller_globals = inspect.currentframe().f_back.f_globals
    for var_name in global_vars_to_clear:
        if var_name in caller_globals:
            del caller_globals[var_name]

    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)

    gc.collect()
    time.sleep(2)

    if verbose and torch.cuda.is_available():
        print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def check_flash_attention_version(verbose: bool = True) -> bool:
    """
    Check Flash Attention version compatibility with Qwen2-VL.

    Qwen2-VL requires Flash Attention >= 2.5.0 for proper operation.
    This function checks the installed version and warns if incompatible.

    Args:
        verbose: Print version information and warnings (default: True)

    Returns:
        bool: True if version is compatible, False otherwise

    Example:
        >>> from src.utils.gpu import check_flash_attention_version
        >>> is_compatible = check_flash_attention_version()
        Flash Attention version: 2.6.3
        ✅ Flash Attention version is compatible

    Note:
        To fix incompatible versions, run:
        pip install flash-attn==2.6.3 --no-build-isolation
    """
    import torch

    if verbose:
        print("\n" + "="*60)
        print("CHECKING FLASH ATTENTION AND ENVIRONMENT")
        print("="*60)

    try:
        import flash_attn
        flash_version = flash_attn.__version__

        if verbose:
            print(f"Flash Attention version: {flash_version}")
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        # Check if version is sufficient for Qwen2-VL
        major, minor = map(int, flash_version.split('.')[:2])
        is_compatible = not (major < 2 or (major == 2 and minor < 5))

        if verbose:
            if not is_compatible:
                print("\n⚠️ WARNING: Flash Attention version is too old for Qwen2-VL!")
                print("   Required: >= 2.5.0")
                print("   To fix, run: pip install flash-attn==2.6.3 --no-build-isolation")
            else:
                print("✅ Flash Attention version is compatible")
            print("="*60 + "\n")

        return is_compatible

    except ImportError:
        if verbose:
            print("❌ Flash Attention is not installed!")
            print("   To install, run: pip install flash-attn==2.6.3 --no-build-isolation")
            print("="*60 + "\n")
        return False


# For backward compatibility - standalone script execution
if __name__ == "__main__":
    # Default configuration from original script
    setup_gpus(use_dual_gpu=True, max_gpus=2, verbose=True)
