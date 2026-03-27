# Environment Setup with uv

## Prerequisites

Install `uv` if you don't have it yet:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or via pip:

```bash
pip install uv
```

## Installing the Environment

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd <project-directory>
```

Install dependencies from the lock file:

```bash
uv sync
```

This will create a virtual environment in `.venv` and install all dependencies at the exact versions specified in `uv.lock`.

## Running Commands

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Or run commands directly without activating:

```bash
uv run python your_script.py
```

## Adding New Dependencies

```bash
uv add <package-name>
```

This updates both `pyproject.toml` and `uv.lock`. Commit both files after adding dependencies.

## Updating Dependencies

```bash
uv sync --upgrade
```

## Entry Points

### Training

```bash
uv run python train.py
```

### Zero-Shot Inference Evaluation

```bash
uv run python evaluate_mymodel_zeroshot.py
```