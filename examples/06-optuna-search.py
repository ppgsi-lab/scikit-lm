"""Tune a language-model classifier inside a scikit-learn Pipeline + Optuna's OptunaSearchCV.

Because the estimators honor the scikit-learn parameter contract -- and the
config objects (``TrainingConfig``, ``LoRAConfig``, ``GenerationConfig``)
subclass ``BaseEstimator`` -- every knob is addressable per field through the
pipeline step name, with the usual ``__`` nesting:

* ``lm__precision`` -- a flat model-loading field.
* ``lm__training__epochs`` -- a field of the nested
  ``TrainingConfig``.
* ``lm__lora__rank`` -- a field of the nested
  ``LoRAConfig``.

The fixed hyperparameters are declared once on the estimator; only the swept
fields go in ``param_distributions``, and OptunaSearchCV samples them with a TPE
sampler -- no re-declaring a whole ``TrainingConfig`` per trial. distilgpt2 on
iris is a weak classifier; the point here is the tuning ergonomics, not accuracy.

Run with: python examples/06-optuna-search.py
"""

from __future__ import annotations

import os

from optuna import create_study

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.integration import OptunaSearchCV
from optuna.samplers import TPESampler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklm import (
    KeyValueSerializer,
    LanguageModelClassifier,
    LoRAConfig,
    SpacedDigits,
    TqdmCallback,
    TrainingConfig,
)


def main() -> None:
    seed = 42

    clf = LanguageModelClassifier(
        model="gpt2-large",
        backend="mlx",
        random_state=seed,
        precision="fp32",
        serializer=KeyValueSerializer(number=SpacedDigits()),
        training=TrainingConfig(
            augmentation_factor=4,
            neftune_noise_alpha=2,
            loss_on_target_only=True,
        ),
        lora=LoRAConfig(),
        callbacks=TqdmCallback(n_train_examples=3),
    )

    pipe = Pipeline([("lm", clf)])

    param_distributions = {
        "lm__training__learning_rate": CategoricalDistribution([0.0001, 0.0002, 0.0003]),
        "lm__training__lr_scheduler": CategoricalDistribution(["linear", "cosine"]),
        "lm__training__epochs": IntDistribution(2, 5),
        "lm__training__batch_size": CategoricalDistribution([8, 16]),
        "lm__lora__rank": CategoricalDistribution([16, 32]),
        "lm__lora__dropout": FloatDistribution(0.05, 0.2),
    }

    search = OptunaSearchCV(
        pipe,
        param_distributions,
        cv=4,
        scoring="accuracy",
        n_trials=15,
        random_state=seed,
        verbose=2,
        study=create_study(
            direction="maximize",
            sampler=TPESampler(multivariate=True, seed=seed),
        ),
    )

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target_names[iris.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    search.fit(X_train, y_train)

    print(f"best params:      {search.best_params_}")
    print(f"best CV accuracy: {search.best_score_:.3f}")
    print(f"test accuracy:    {accuracy_score(y_test, search.predict(X_test)):.3f}")


if __name__ == "__main__":
    main()
