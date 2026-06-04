"""Estimate a language-model classifier's accuracy with stratified k-fold CV.

Because ``LanguageModelClassifier`` honors the scikit-learn estimator contract,
it drops straight into ``cross_val_score``: each fold gets a freshly ``clone``-d
estimator (so no fitted state leaks across folds) and is fine-tuned from the base
model independently. ``StratifiedKFold`` keeps each fold's class proportions
close to the full set's -- the right default for classification, and essential on
small or imbalanced data where a random split can drop a class from a fold.

distilgpt2 on iris is a weak classifier; the point here is the evaluation
protocol, not the score.

Run with: python examples/07-stratified-cv.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklm import (
    GenerationConfig,
    LanguageModelClassifier,
    LoggingCallback,
    TrainingConfig,
)


def main() -> None:
    seed = 42

    iris = load_iris(as_frame=True)
    X = iris.data.rename(columns=lambda c: c.removesuffix(" (cm)"))
    y = iris.target_names[iris.target]

    clf = LanguageModelClassifier(
        backend="mlx",
        model="gabfssilva/distilgpt2",
        training=TrainingConfig(
            epochs=3,
            batch_size=16,
            lr_scheduler='constant',
            augmentation_factor=24,
        ),
        generation=GenerationConfig(n_samples=24),
        random_state=seed,
        callbacks=LoggingCallback(log_every='epoch'),
    )

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    print("per-fold accuracy: " + ", ".join(f"{s:.3f}" for s in scores))
    print(f"mean +/- std:      {scores.mean():.3f} +/- {scores.std():.3f}")
    lo, hi = np.percentile(scores, 2.5), np.percentile(scores, 97.5)
    print(f"95% CI:            [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
