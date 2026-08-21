# EEG Alcoholism Classification (EEGNet)

Utilizing a convultional neural network (CNN) to predict classification of a person clinically assessed as an alcoholic or not based on EEG input.

Implemented in PyTorch using EEGNet, a compact CNN designed for EEG signals, to assign the classes **alcoholic** vs **control**

## Dataset

- **Source:** UCI EEG Database (alcoholism study)
- **Input file:** `compiled.csv`, long-format with columns `(subject, trial, condition, channel, sample, value)`
- **Signal shape:** 63 channels X 256 time samples per trial (1 second @ 256 Hz)
- Group label is parsed from the subject ID (`co2a...` -> alcoholic, `co2c...` -> control)
- After pivoting to `(trial, channel, sample)` tensors: **1,176 trials** (593 alcoholic / 583 control)

Example raw signals for one trial from each class:

![Sample EEG signals](images/sample_signals.png)

## Pipeline

1. **Load & reshape**: pivot the long-format CSV into a `(n_trials, 63, 256)` NumPy array, one label per trial
2. **Dataset/DataLoader**: wrap as a PyTorch `Dataset`, split 70% train / 15% val / 15% test (824 / 176 / 176 trials)
3. **Model**: EEGNet-8,2 (Lawhern et al., 2018)
4. **Train**: 30 epochs, Adam (lr=1e-3, weight decay=1e-4), cosine annealing LR schedule, cross-entropy loss; best checkpoint (by val loss) saved to `best_eegnet.pt`
5. **Evaluate**: load best checkpoint, report classification metrics and confusion matrix on the held-out test set

## Model architecture (EEGNet-8,2)

| Block | Operation | Purpose |
|---|---|---|
| Temporal conv | `Conv2d(1, F1, (1, 64))` | Spectral feature extraction |
| Depthwise spatial conv | `Conv2d(F1, D×F1, (63, 1), groups=F1)` | Learns spatial (electrode) filters |
| Separable conv | `Conv2d -> pointwise` | Compact temporal summary |
| Classifier | `Linear -> softmax` | Binary output |

- F1 = 8 temporal filters, D = 2 depth multiplier, F2 = 16 separable filters, dropout = 0.5
- **2,370 trainable parameters**

## Results

Training converged steadily over 30 epochs:

![Training curves](images/training_curves.png)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.688 | 0.568 | 0.641 | 0.642 |
| 10 | 0.401 | 0.825 | 0.384 | 0.858 |
| 20 | 0.301 | 0.881 | 0.331 | 0.875 |
| 30 | 0.297 | 0.886 | 0.311 | 0.869 |

**Test set performance:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Alcoholic | 0.80 | 0.90 | 0.85 | 81 |
| Control | 0.91 | 0.81 | 0.86 | 95 |
| **Accuracy** | | | **0.85** | 176 |

MSE: 0.148

![Confusion matrix](images/confusion_matrix.png)

## Requirements

```
numpy
pandas
matplotlib
seaborn
torch
scikit-learn
```

## Usage

1. Place `compiled.csv` in the project root (or update `CSV_PATH` in the config cell)
2. Run the notebook top to bottom: hyperparameters (`BATCH_SIZE`, `EPOCHS`, `LR`, split ratios) are set in the config cell and can be edited freely
3. The best model checkpoint is saved as `best_eegnet.pt`
