"""Predict a continuous target with a language-model regressor.

Greedy decoding would return the mode, not the mean, so ``predict`` draws
``generation.n_samples`` completions per row and averages them -- a Monte-Carlo
estimate of the conditional mean. Here it estimates diabetes disease progression
from ten physiological measurements.

gpt2-large is still a small base model, so this is a demonstration of the API
rather than a strong regressor; the printed baseline (predicting the training
mean) is the bar to beat.

Run with: python examples/02-diabetes-regressor.py
"""

from __future__ import annotations

from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from sklm import (
    DiscretizationConfig,
    LanguageModelRegressor,
    RichCallback,
    TrainingConfig,
)


def main() -> None:
    data = load_diabetes(as_frame=True, scaled=False)
    X = data.data.round(3)
    y = data.target

    seed = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    reg = LanguageModelRegressor(
        model="gabfssilva/distilgpt2",
        backend="mlx",
        precision="fp32",
        training=TrainingConfig(
            epochs=4,
            augmentation_factor=30,
            batch_size=8,
            loss_on_target_only=True,
        ),
        discretization=DiscretizationConfig(bins=0.5),
        random_state=seed,
        callbacks=RichCallback(),
    ).fit(X_train, y_train)

    pred = reg.predict(X_test)
    baseline = [y_train.mean()] * len(y_test)

    print(f"{'#':>3}  {'predicted':>9}  {'target':>7}  {'error':>7}")
    for i, (yhat, ytrue) in enumerate(zip(pred[:15], y_test[:15], strict=True)):
        print(f"{i:>3}  {yhat:>9.1f}  {ytrue:>7.1f}  {yhat - ytrue:>+7.1f}")

    print(f"model MAE: {mean_absolute_error(y_test, pred):7.2f}")
    print(f"model RMSE: {root_mean_squared_error(y_test, pred):7.2f}")
    print(f"model R2:  {r2_score(y_test, pred):7.2f}")
    print(f"mean  MAE: {mean_absolute_error(y_test, baseline):7.2f}  (predict the training mean)")


if __name__ == "__main__":
    main()
