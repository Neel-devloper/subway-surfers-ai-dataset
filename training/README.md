# Subway Surfers action classifier — training

This directory trains a small CNN that looks at a Subway Surfers screenshot and
predicts which swipe the player should make: **UP, DOWN, LEFT, RIGHT, or NONE**.

## Data

Screenshots live under `screen_collector/screens/<speed>/<ACTION>/*.png`. The
`<speed>` grouping (`slow | medium | fast | legacy`) only reflects how the data
was collected; training ignores it and uses a single classifier over the action
labels.

| Label | Images |
|-------|-------:|
| UP    |   714  |
| DOWN  |   439  |
| LEFT  |   780  |
| RIGHT |   723  |
| NONE  |  1799  |
| **Total** | **4455** |

The classes are imbalanced (NONE ≈ 40%). Two mechanisms counteract this:
inverse-frequency **class weights** in the loss, and a **WeightedRandomSampler**
that oversamples rare classes so each batch is roughly balanced. Balanced
accuracy (mean per-class recall) is used to select the best checkpoint so NONE
cannot dominate the metric.

## Files

- `model.py` — the `SubwayCNN` architecture + canonical class order and the
  input/normalisation constants (single source of truth, shared with inference).
- `dataset.py` — image discovery, in-memory caching, stratified split, light
  brightness augmentation (no horizontal flip — it would swap LEFT/RIGHT).
- `train.py` — training loop, evaluation, checkpoint + metrics writing.

## Usage

```bash
pip install -r training/requirements.txt
python training/train.py --epochs 40 --batch-size 64 --lr 1e-3
```

Outputs land in `models/`:

- `subway_surfers_cnn.pth` — checkpoint: `state_dict` plus the class order,
  input geometry, and normalisation constants, so inference needs nothing else.
- `training_metrics.json` — per-epoch history and the final confusion matrix.

## Model

`SubwayCNN` is a compact 4-block convolutional net (~98K params): each block is
`Conv3x3 → BatchNorm → ReLU → MaxPool`, widening 3→16→32→64→128, followed by
global average pooling, dropout, and a linear classifier. Input is a
128×72 RGB image (portrait, ~9:16 to match phone screens). It is intentionally
small so it trains in minutes on CPU and runs in real time during live play.

## Results

Trained for 40 epochs on CPU (~9 minutes). Best checkpoint, on the held-out
15% validation split:

- **Validation accuracy: 79.3%**
- **Balanced accuracy (mean per-class recall): 82.6%**

| Label | Recall |
|-------|-------:|
| UP    | 0.86 |
| DOWN  | 0.94 |
| LEFT  | 0.79 |
| RIGHT | 0.82 |
| NONE  | 0.72 |

Confusion matrix (rows = true, cols = predicted):

```
           UP   DOWN   LEFT  RIGHT   NONE
UP         92      3      6      4      2
DOWN        1     62      0      0      3
LEFT        8      3     92      8      6
RIGHT       5      0      9     89      5
NONE       14      6     38     17    195
```

Chance for 5 balanced classes is 20%; NONE-always would score ~40% raw accuracy
but only 20% balanced. The full per-epoch history is in
`models/training_metrics.json`.

## Loading the checkpoint for inference

```python
import torch
from model import build_model

ckpt = torch.load("models/subway_surfers_cnn.pth", map_location="cpu", weights_only=False)
model = build_model(num_classes=len(ckpt["classes"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()
# ckpt["classes"], ckpt["input"], ckpt["norm"] describe preprocessing.
```
