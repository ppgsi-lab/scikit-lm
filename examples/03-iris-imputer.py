"""Fill missing cells by conditioning a language model on the observed ones.

``LanguageModelImputer`` learns ``p(any column | the rest)`` from a single
fine-tune, so it fills a missing cell by conditioning on whatever else the row
exposes, respecting cross-column structure. Here we knock out ~15% of a single
numeric column (``petal width``) and let the model fill it from the other three
features.

Because petal width (rounded to 0.1 cm) spans a small, discrete support, we turn
on ``discretization`` for that column: instead of generating the value, the model
*scores* the observed petal widths by likelihood and reduces the distribution to
its single most likely candidate (argmax / mode) -- deterministic, and often
sharper than sampling when the continuous space is small. The per-column mapping
leaves any other column on the generative path.

Run with: python examples/03-iris-imputer.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

from sklm import (
    DiscretizationConfig,
    GenerationConfig,
    JSONSerializer,
    LanguageModelImputer,
    RichCallback,
    TrainingConfig,
)


def main() -> None:
    seed = 42

    iris = load_iris(as_frame=True)
    frame = iris.data.round(1)
    target_col = "petal width"

    frame = frame.rename(columns=lambda c: c.removesuffix(" (cm)"))

    rng = np.random.default_rng(seed)
    mask = rng.random(len(frame)) < 0.15  # knock out ~15% of one column's cells
    corrupt = frame.copy()
    corrupt.loc[mask, target_col] = np.nan

    filled = LanguageModelImputer(
        model="gpt2-large",
        precision="fp32",
        backend="mlx",
        serializer=JSONSerializer(),
        discretization={target_col: DiscretizationConfig(bins=1.0, estimate="mode")},
        generation=GenerationConfig(n_samples=6),
        training=TrainingConfig(
            epochs=8,
            lr_scheduler="linear",
            augmentation_factor=4,
            batch_size=4,
            loss_on_target_only=True,
            neftune_noise_alpha=2,
        ),
        random_state=seed,
        callbacks=RichCallback(),
    ).fit_transform(corrupt)

    print(f"missing cells before: {int(corrupt.isna().sum().sum())}")
    print(f"missing cells after:  {int(filled.isna().sum().sum())}")

    truth = frame.loc[mask, target_col].to_numpy()
    imputed = filled.loc[mask, target_col].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((truth - imputed) ** 2)))
    mae = float(np.abs(truth - imputed).mean())
    nrmse = rmse / float(frame[target_col].std())  # RMSE in units of the column's spread
    print(
        f"{target_col}: RMSE {rmse:.3f}, NRMSE {nrmse:.3f}, MAE {mae:.3f} "
        f"over {int(mask.sum())} masked cells"
    )


if __name__ == "__main__":
    main()
