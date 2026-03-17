# Homework 2: Policy Gradients

## Setup

For general setup and Modal instructions, see Homework 1's README.

### Quick Setup (Linux/WSL)

Run the setup script to install system dependencies and create the virtual environment:

```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup

1. Install system dependencies (Linux/WSL):
   ```bash
   sudo apt-get update && sudo apt-get install -y swig python3.10-dev
   ```
   On Mac with Homebrew:
   ```bash
   brew install swig cmake
   ```

2. Install Python dependencies (from the `hw2` directory):
   ```bash
   uv sync
   ```

## Examples

Here are some example commands. Run them in the `hw2` directory.

* To run on a local machine:
  ```bash
  uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
  ```


* To run on Modal:
  ```bash
  uv run modal run src/scripts/modal_run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
  ```
  * Note that Modal is likely not necessary for this assignment.
In testing, training was much faster on a local laptop CPU than on Modal.
However, you may still use Modal if you wish.
  * You may request a different GPU type, CPU count, and memory size by changing variables in `src/scripts/modal_run.py`
  * Use `modal run --detach` to keep your job running in the background.

## Troubleshooting

* **`swig` not found**: Install with `sudo apt-get install -y swig` (Linux/WSL) or `brew install swig` (Mac).
* **`Python.h` not found**: Install with `sudo apt-get install -y python3.10-dev` (Linux/WSL). Adjust the version to match your Python (e.g., `python3.11-dev`).
* **VIRTUAL_ENV mismatch warning**: This is harmless — it means another hw's venv is active. `uv run` uses its own `.venv` regardless.
* **`box2d-py` build failure**: Ensure both `swig` and `python3.X-dev` are installed, then retry.
