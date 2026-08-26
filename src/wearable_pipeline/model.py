"""Stage 4 (sequence modeling / pain-proxy inference): a small PyTorch LSTM mapping a
window's physiological feature sequence to a probabilistic activation score -- the "ouch
meter." Deliberately tiny (single-layer, 16 hidden units): with 5 subjects and on the order
of ~30 event windows total, a larger model would just memorize.

No genuine pain self-reports exist in this dataset (it's an exam-stress corpus, not a pain
one), so there's no ground truth to train against directly. The training target is instead a
heuristic composite autonomic-activation index -- elevated heart rate, suppressed HRV
(RMSSD), and elevated EDA -- z-scored across the cohort and squashed through a sigmoid,
matching the architecture doc's own description of the ouch meter as "a probabilistic
measure of pain-associated physiological activation." The LSTM still does real supervised
learning: it has to learn to approximate that scalar target from the raw windowed feature
sequence, not just recompute the heuristic identically.
"""

import numpy as np
import torch
import torch.nn as nn

from src.wearable_pipeline.feature_extraction import FEATURE_NAMES, NUM_FEATURES
from src.wearable_pipeline.models import WearableWindow
from src.wearable_pipeline.preprocessing import apply_normalizer, fit_normalizer

HIDDEN_SIZE = 16
_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


class OuchMeterLSTM(nn.Module):
    def __init__(self, input_size: int = NUM_FEATURES, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.lstm(x)
        hidden = h_n[-1]  # (batch, hidden_size)
        score = torch.sigmoid(self.head(hidden)).squeeze(-1)  # (batch,)
        return score, hidden


def compute_heuristic_targets(mean_features: np.ndarray) -> np.ndarray:
    """mean_features: (num_windows, NUM_FEATURES), each row a window's own step-averaged raw
    feature vector. Returns a (num_windows,) array of proxy activation targets in (0, 1)."""
    mean, std = fit_normalizer(mean_features)
    z = apply_normalizer(mean_features, mean, std)
    activation = (
        z[:, _FEATURE_INDEX["hr_mean"]]
        - z[:, _FEATURE_INDEX["hrv_rmssd"]]
        + z[:, _FEATURE_INDEX["eda_tonic_mean"]]
        + z[:, _FEATURE_INDEX["eda_phasic_std"]]
    )
    return 1.0 / (1.0 + np.exp(-activation))


def train_model(
    windows: list[WearableWindow], epochs: int = 150, lr: float = 0.01, seed: int = 0
) -> tuple[OuchMeterLSTM, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    model = OuchMeterLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    all_steps = np.concatenate([w.feature_sequence for w in windows], axis=0)
    mean, std = fit_normalizer(all_steps)

    sequences = [
        torch.tensor(apply_normalizer(w.feature_sequence, mean, std), dtype=torch.float32)
        for w in windows
    ]
    mean_features_per_window = np.array([w.feature_sequence.mean(axis=0) for w in windows])
    targets = torch.tensor(compute_heuristic_targets(mean_features_per_window), dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        for seq, target in zip(sequences, targets):
            score, _ = model(seq.unsqueeze(0))
            loss_fn(score, target.unsqueeze(0)).backward()
        optimizer.step()

    model.eval()
    return model, mean, std


def score_window(
    model: OuchMeterLSTM, mean: np.ndarray, std: np.ndarray, feature_sequence: np.ndarray
) -> tuple[float, np.ndarray]:
    with torch.no_grad():
        x = torch.tensor(apply_normalizer(feature_sequence, mean, std), dtype=torch.float32)
        score, hidden = model(x.unsqueeze(0))
    return float(score.item()), hidden.squeeze(0).numpy()
