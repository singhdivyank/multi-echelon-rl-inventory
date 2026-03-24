# RL Project Setup & Training Guide

This document summarizes the setup process, bug fixes, and instructions for running the Multi-Echelon Inventory System (MEIS) training pipeline on the newly created `ishan-workingBranch`.

## Training Command

To run the full training pipeline end-to-end (50,000 episodes) on your RTX 5070 Ti GPU, use the following command:

```bash
.venv\Scripts\python main.py --config configs/custom.yaml --seed 42 --save-dir ./results
```

**Note:** Always ensure you are using the virtual environment `.venv` when running commands.

## How Checkpointing Works

The training pipeline now fully supports saving and resuming from checkpoints. This means if training is interrupted, you won't lose your progress.

*   **Save Location:** `./results/checkpoints/checkpoint_ep{N}.pt`
*   **Final Checkpoint:** `./results/checkpoints/checkpoint_final.pt` (saved at the end)
*   **Save Frequency:** Every 5000 episodes (as defined in `configs/custom.yaml`).
*   **What is Saved:** Model weights, optimizer state, the current training episode, total steps taken, and the complete training history.
*   **How to Resume:** Simply re-run the exact same training command shown above. The script will automatically detect the most recent `checkpoint_ep*.pt` file and resume training from that episode.

## Summary of Changes Made

The following modifications and bug fixes were applied to the codebase to ensure robust, error-free training on the GPU:

### 1. GPU Support (PyTorch Upgrade)
*   **Issue:** The existing `PyTorch 2.5.1+cu121` installation was incompatible with the RTX 5070 Ti (Blackwell architecture, compute capability 12.0), causing `CUDA device-side assertion` errors.
*   **Fix:** Upgraded PyTorch to the latest nightly build with CUDA 12.8 support (`torch-2.12.0.dev20260324+cu128`).
*   **Config:** Set `device: 'auto'` in `configs/custom.yaml` so the agent actively utilizes the GPU for training.

### 2. Code Bug Fixes
*   **`a3c_agent.py`:**
    *   Moved the `from typing import Dict` import to the top of the file (it was previously used in a type hint before being imported).
    *   Fixed the `_get_device()` method to properly handle the `device` parameter passed from the configuration. Added a robust check that actually attempts a tensor operation (`torch.zeros`) to validate CUDA capability before returning it, with a graceful fallback to CPU if it fails.
    *   Removed a duplicate `import torch` at the bottom of the file.
*   **`main.py`:** Changed the hardcoded fallback device generation from `'cuda' if torch.cuda.is_available() else 'cpu'` to `'auto'`, delegating the decision to the now-robust `_get_device()` method in the agent.
*   **`__init__.py` files:** Added missing empty `__init__.py` files to the `agents`, `baselines`, `envs`, and `utils` folders so Python properly recognizes them as packages.
*   **`.gitignore`:** Corrected `*.venv` to `.venv/` to successfully ignore the virtual environment directory instead of files ending with that extension.

### 3. Checkpointing Implementation (`trainer.py`)
*   **`_save_checkpoint`:** Upgraded to save the episode number, total steps, and the `training_history` dictionary alongside the network and optimizer states into the `.pt` file.
*   **`_find_latest_checkpoint`:** Created a new method that scans the checkpoint directory using regex to find the `.pt` file with the highest episode number.
*   **`_load_checkpoint`:** Created a new method to restore the model, optimizer, episode number, and training history from the loaded file.
*   **`train()` Loop:** Modified the main training loop to check for the latest checkpoint at startup and automatically resume the loop and the `tqdm` progress bar from where it left off.
