"""Balance a skewed dataset by generating synthetic minority rows.

``LanguageModelOverSampler`` implements the imbalanced-learn sampler API
(``fit_resample``). For each under-represented class it conditions generation on
that label and synthesizes the remaining features -- so, unlike SMOTE, it works
directly on mixed/categorical tables without numeric encoding.

Here we take Iris (balanced, 50 rows/class), throw away most of one class to
make it minority, oversample it back, then compare the synthesized rows against
the *real* Iris rows of that class to see whether the model recovered the
distribution it never fully saw.

Run with: python examples/04-imbalanced-oversampler.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from sklm import LanguageModelOverSampler, TrainingConfig

MODEL = "gabfssilva/distilgpt2"
BACKEND = "mlx"
MINORITY = 2  # iris class to under-sample, then regenerate
KEEP = 10  # real rows of the minority class left for training


def main() -> None:
    iris = load_iris(as_frame=True)
    X_full = iris.data.set_axis(
        ["sepal_length", "sepal_width", "petal_length", "petal_width"], axis=1
    )
    y_full = iris.target.to_numpy()

    seed = 42
    rng = np.random.RandomState(seed)
    minority_idx = np.where(y_full == MINORITY)[0]
    drop = rng.choice(minority_idx, size=len(minority_idx) - KEEP, replace=False)
    mask = np.ones(len(y_full), dtype=bool)
    mask[drop] = False
    X = X_full[mask].reset_index(drop=True)
    y = y_full[mask]
    print("before:", pd.Series(y).value_counts().sort_index().to_dict())

    X_res, y_res = LanguageModelOverSampler(
        model=MODEL,
        backend=BACKEND,
        training=TrainingConfig(epochs=30, batch_size=16),
        random_state=seed,
    ).fit_resample(X, y)

    print("after: ", pd.Series(y_res).value_counts().sort_index().to_dict())
    print(f"rows: {len(X)} -> {len(X_res)} ({len(X_res) - len(X)} synthesized)\n")

    synth = X_res.iloc[len(X) :].reset_index(drop=True)
    real = X_full[y_full == MINORITY]
    comparison = pd.DataFrame(
        {
            "real_mean": real.mean(),
            "synth_mean": synth.mean(),
            "real_std": real.std(),
            "synth_std": synth.std(),
        }
    ).round(2)
    print(f"distribution of class {MINORITY} -- real Iris vs synthesized:")
    print(comparison)


if __name__ == "__main__":
    main()
