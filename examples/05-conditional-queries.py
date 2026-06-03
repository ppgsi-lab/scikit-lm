"""Query any column from any subset -- the mechanism behind every estimator.

Every scikit-lm estimator wraps one ``TabularLanguageModel``: a single
fine-tune over randomly permuted column orders, which learns
``p(any column | any subset of the others)``. The classifier, regressor,
imputer and oversampler are just different views on it. This example drives that
core object directly -- scoring candidates for one column (what the classifier
does) and generating another (what the imputer/regressor do) -- without going
through an estimator.

Run with: python examples/05-conditional-queries.py
"""

from __future__ import annotations

from sklearn.datasets import load_iris

from sklm import (
    GenerationConfig,
    JSONSerializer,
    MLXBackend,
    ModelConfig,
    TabularLanguageModel,
    TrainingConfig,
)

MODEL = "gabfssilva/distilgpt2"


def main() -> None:
    iris = load_iris(as_frame=True)
    frame = iris.data.round(1)
    frame["species"] = iris.target_names[iris.target]

    lm = TabularLanguageModel(
        backend=MLXBackend(),
        serializer=JSONSerializer(),
        training=TrainingConfig(epochs=40, batch_size=16),
        model_config=ModelConfig(model=MODEL),
        random_state=0,
    ).fit(frame)

    # Score a fixed candidate set -- conditioning only on the petal measurements.
    known = {"petal length (cm)": 1.4, "petal width (cm)": 0.2}
    proba = lm.predict_proba(known, "species", list(iris.target_names))
    ranked = ", ".join(f"{c}={p:.2f}" for c, p in zip(iris.target_names, proba, strict=True))
    print(f"p(species | small petals) -> {ranked}")

    # Generate a numeric column -- conditioning on the species instead.
    gen = GenerationConfig(max_new_tokens=8)
    out = lm.complete({"species": "setosa"}, ["petal length (cm)"], gen)
    sampled = out["petal length (cm)"] if out is not None else "(malformed)"
    print(f"sample petal length | species=setosa -> {sampled}")


if __name__ == "__main__":
    main()
