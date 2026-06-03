"""Classify iris flowers with a fine-tuned language model.

``LanguageModelClassifier`` serializes each row to JSON, fine-tunes
``gabfssilva/distilgpt2`` on the MLX (Apple Silicon) backend, then ranks the
candidate species by likelihood. Because it scores a fixed candidate set rather
than free-generating, every prediction is a valid class and ``predict_proba``
is a genuine distribution.

Run with: python examples/01-iris-classifier.py
"""

from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklm import (
    GenerationConfig,
    LanguageModelClassifier,
    RichCallback,
    TrainingConfig,
)


def main() -> None:
    df = load_iris(as_frame=True)
    X = df.data.rename(columns=lambda c: c.removesuffix(" (cm)"))
    y = df.target_names[df.target]
    seed = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    classifier = LanguageModelClassifier(
        backend="mlx",
        model="gabfssilva/distilgpt2",
        training=TrainingConfig(
            epochs=4,
            batch_size=8,
            lr_scheduler='constant',
            augmentation_factor=24,
        ),
        generation=GenerationConfig(n_samples=24),
        random_state=42,
        callbacks=RichCallback(),
    )

    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)
    print(f"accuracy: {accuracy_score(y_test, pred):.3f}\n")

    cm = confusion_matrix(y_test, pred, labels=classifier.classes_)
    print("confusion matrix (rows=true, cols=pred):")
    print(" " * 12 + "".join(f"{c:>12}" for c in classifier.classes_))
    for label, row in zip(classifier.classes_, cm, strict=True):
        print(f"{label:>12}" + "".join(f"{v:>12}" for v in row))
    print()

    proba = classifier.predict_proba(X_test.head(3))
    for i in range(len(proba)):
        dist = ", ".join(f"{c}={p:.2f}" for c, p in zip(classifier.classes_, proba[i], strict=True))
        print(f"row {i}: predicted={pred[i]:<11} ({dist})")


if __name__ == "__main__":
    main()
