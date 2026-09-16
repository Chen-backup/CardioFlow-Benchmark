# CardioFlow

This repository provides the CardioFlow model implementation, default model
configuration, ECG preprocessing utilities, and a minimal training demo.

The repository does not include datasets, pretrained weights, training outputs,
comparison models, or result files.

## Contents

```text
model/
  model.py                  CardioFlow wrapper and model builder
  core/                     CardioFlow computational core
config/
  cardioflow.yaml           Default model and training configuration
preprocessing/
  download.py               Dataset download utilities
  preprocess.py             Dataset preprocessing code
  dataset.py                Loader for prepared datasets
  constants.py              Dataset definitions and split metadata
demo.py                     Minimal training demo
requirements.txt            Python dependencies
setup.sh                    Environment setup script
```

## Installation

Python 3.11 or 3.12 is recommended.

```bash
git clone <repository-url>
cd CardioFlow
bash setup.sh
source .venv/bin/activate
```

To use a custom Python executable:

```bash
PYTHON_BIN=python3.12 bash setup.sh
```

To install with a PyPI mirror:

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash setup.sh
```

## Data Available

The preprocessing code supports the following public ECG datasets:

- PTB-XL 1.0.3: https://physionet.org/content/ptb-xl/1.0.3/
- PhysioNet/CinC Challenge 2020 data: https://physionet.org/content/challenge-2020/1.0.2/
- LUDB 1.0.1: https://physionet.org/content/ludb/1.0.1/

Dataset access and reuse are governed by the terms of the original providers.

## Download and Preprocess Data

The commands below download one dataset into `data/raw/<dataset>` and export the
prepared arrays into `data/processed/<dataset>`.

```bash
python -c "from preprocessing.download import download; download('ptbxl', 'data/raw', workers=16)"
python -c "from preprocessing.preprocess import prepare; prepare('ptbxl', 'data/raw', 'data/processed')"
```

Replace `ptbxl` with `cpsc2018` or `ludb` to prepare another dataset:

```bash
python -c "from preprocessing.download import download; download('cpsc2018', 'data/raw', workers=16)"
python -c "from preprocessing.preprocess import prepare; prepare('cpsc2018', 'data/raw', 'data/processed')"

python -c "from preprocessing.download import download; download('ludb', 'data/raw', workers=16)"
python -c "from preprocessing.preprocess import prepare; prepare('ludb', 'data/raw', 'data/processed')"
```

The downloader verifies checksums and can be rerun after interruptions.

## Run the Training Demo

After preparing a dataset, run:

```bash
python demo.py \
  --dataset ptbxl \
  --data-root data/processed \
  --epochs 1 \
  --max-batches 20 \
  --output outputs/cardioflow_ptbxl_demo.pt
```

For CPSC2018:

```bash
python demo.py --dataset cpsc2018 --data-root data/processed --epochs 1
```

For LUDB delineation:

```bash
python demo.py --dataset ludb --data-root data/processed --epochs 1 --batch-size 16
```

The demo trains CardioFlow on the training split only and saves a checkpoint to
the requested output path. It is intended as a compact usage example rather than
a full training pipeline.

## Model Usage

```python
import torch
from model.model import build_model

model = build_model(
    task="classification",
    num_classes=5,
    config={"population_width": 96, "dropout": 0.15},
)

x = torch.randn(2, 12, 1000)
lead_mask = torch.ones(2, 12)
valid_mask = torch.ones(2, 1, 1000)
output = model(x, lead_mask=lead_mask, valid_mask=valid_mask)
logits = output["logits"]
```

For LUDB delineation, use `task="delineation"` and `num_classes=4`.
