"""Synthesize a whole table from scratch -- unconditional joint sampling.

The four estimators each fix *which* columns go in the prompt. Drop that
constraint entirely and the same ``TabularLanguageModel`` becomes a tabular
synthesizer: generate every column from an empty context, so each row is a draw
from the learned joint ``p(features, label)``. The first column is sampled from
its marginal, every later one conditions on the cells already produced.

Here we fit on the full Iris table (features + species), sample a fresh synthetic
table of the same size, then compare per-feature moments and the class balance to
check whether the joint distribution was recovered.

Run with: python examples/08-synthesizer.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris

from sklm import (
    GenerationConfig,
    MLXBackend,
    ModelConfig,
    RichCallback,
    TabularLanguageModel,
    TrainingConfig, QuantizationConfig, LoRAConfig, LoggingCallback,
)

N_SYNTH = 150

def main() -> None:
    iris = load_iris(as_frame=True)
    numeric = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    real = iris.data.set_axis(numeric, axis=1).round(1)
    real["species"] = iris.target_names[iris.target]

    lm = TabularLanguageModel(
        backend=MLXBackend(),
        model_config=ModelConfig(
            model="gabfssilva/distilgpt2",
            lora=LoRAConfig(dropout=0.1)
        ),
        training=TrainingConfig(epochs=2, batch_size=1, augmentation_factor=12),
        random_state=42,
        callbacks=LoggingCallback(
            log_every='epoch',
            n_predict_examples=N_SYNTH
        ),
    ).fit(real)

    columns = list(real.columns)
    labels = list(iris.target_names)
    per_class = N_SYNTH // len(labels)
    knowns = [{"species": label} for label in labels for _ in range(per_class)]
    rows = lm.complete_many(
        knowns, [numeric] * len(knowns), generation=GenerationConfig(temperature=0.7)
    )
    synth = pd.DataFrame([r for r in rows if r is not None], columns=columns)
    synth[numeric] = synth[numeric].astype(float)
    print(f"synthesized {len(synth)}/{len(knowns)} valid rows\n")

    comparison = pd.DataFrame(
        {
            "real_mean": real[numeric].mean(),
            "synth_mean": synth[numeric].mean(),
            "real_std": real[numeric].std(),
            "synth_std": synth[numeric].std(),
        }
    ).round(2)
    print("per-feature distribution -- real Iris vs synthesized:")
    print(comparison, "\n")

    print("species balance -- real vs synthesized:")
    print("  real :", real["species"].value_counts().sort_index().to_dict())
    print("  synth:", synth["species"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

# per-feature distribution -- real Iris vs synthesized:
#               real_mean  synth_mean  real_std  synth_std
# sepal_length       5.84        5.90      0.83       0.76
# sepal_width        3.06        2.98      0.44       0.06
# petal_length       3.76        3.68      1.77       1.59
# petal_width        1.20        1.28      0.76       0.86
#
# species balance -- real vs synthesized:
#   real : {'setosa': 50, 'versicolor': 50, 'virginica': 50}
#   synth: {np.str_('setosa'): 50, np.str_('versicolor'): 50, np.str_('virginica'): 50}

#               real_mean  synth_mean  real_std  synth_std
# sepal_length       5.84        5.95      0.83       0.73
# sepal_width        3.06        3.00      0.44       0.02
# petal_length       3.76        3.75      1.77       1.64
# petal_width        1.20        1.27      0.76       0.86
