# scikit-lm

[![PyPI](https://img.shields.io/pypi/v/scikit-lm.svg)](https://pypi.org/project/scikit-lm/)
[![Python](https://img.shields.io/pypi/pyversions/scikit-lm.svg)](https://pypi.org/project/scikit-lm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**scikit-learn estimators backed by a fine-tuned autoregressive language model.**

scikit-lm gives you a classifier, a regressor, a missing-value imputer, and an  imbalanced-learn oversampler that all work directly on tabular data — mixed numeric and categorical columns, no one-hot encoding, no scaling required — by  fine-tuning a small language model on your table and then performing inference on the learned representation. Every estimator follows the familiar scikit-learn API (`fit` / `predict` / `transform` / `fit_resample`), drops into a `Pipeline`, and is tunable with `GridSearchCV` or Optuna.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklm import LanguageModelClassifier

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target_names[iris.target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LanguageModelClassifier(random_state=42)   # distilgpt2 on the Hugging Face backend
clf.fit(X_train, y_train)

clf.predict(X_test)        # -> array(['setosa', 'versicolor', ...])
clf.predict_proba(X_test)  # -> per-row distribution over clf.classes_
```

---

## Table of contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [The four estimators](#the-four-estimators)
  - [Classifier](#classifier)
  - [Regressor](#regressor)
  - [Imputer](#imputer)
  - [Oversampler](#oversampler)
- [The shared core: `TabularLanguageModel`](#the-shared-core-tabularlanguagemodel)
  - [Tabular synthesis](#tabular-synthesis)
- [Configuration](#configuration)
  - [Serialization](#serialization)
  - [Training](#training-trainingconfig)
  - [Generation](#generation-generationconfig)
  - [Discretization](#discretization-discretizationconfig)
  - [LoRA & model loading](#lora--model-loading)
- [Backends](#backends)
- [scikit-learn integration](#scikit-learn-integration)
- [Callbacks](#callbacks)
- [Requirements](#requirements)
- [License](#license)

---

## How it works

Everything in scikit-lm is built on **a single mechanism**.

A tabular row is turned into a short piece of text (JSON by default), and a small autoregressive language model is fine-tuned to produce that text. The trick is in *how* the rows are presented during training: **the order of the columns is randomly re-permuted for every row at every epoch.**

```
        row                         serialized (one random order per epoch)
  ┌──────────────┐
  │ sepal  = 5.1 │   epoch 1 ─▶  {"species": "setosa", "sepal": 5.1, "petal": 1.4}
  │ petal  = 1.4 │   epoch 2 ─▶  {"petal": 1.4, "sepal": 5.1, "species": "setosa"}
  │ species= ... │   epoch 3 ─▶  {"sepal": 5.1, "species": "setosa", "petal": 1.4}
  └──────────────┘        ...
```

Because an autoregressive model predicts each token from the tokens before it, and because every column shows up in every position across epochs, the model is forced to learn to predict **any column from any subset of the others**:

$$p(\text{any column} \mid \text{any subset of the other columns})$$

That single conditional distribution is all four estimators need. Each one is just a choice of *which columns go into the prompt* and *which column the model produces*:

| Estimator    | Conditions on (prompt)      | Produces (target)        | How it reads the answer            |
|--------------|-----------------------------|--------------------------|------------------------------------|
| Classifier   | all features                | the class label          | **scores** each candidate label, ranks them |
| Regressor    | all features                | the numeric target       | **generates** the value `n` times, averages |
| Imputer      | a row's observed cells      | that row's missing cells | **generates** each missing value   |
| Oversampler  | a minority class label      | the features             | **generates** synthetic rows       |

Two primitives implement those two reading strategies:

- **Scoring** — build a prompt that stops right before the target value, then compute the likelihood the model assigns to each candidate (`setosa`, `versicolor`, `virginica`) and normalize into a probability distribution. This is what makes `predict_proba` well-defined and guarantees the classifier only ever predicts a real class. Scoring is deterministic.
- **Generation** — build the same prompt and let the model sample the value as text, then parse it back to a Python value. Used wherever the answer space is open (numbers, free categories, whole synthetic rows).

Missing cells (`None`, `NaN`, `inf`) are never serialized — training drops them, and at inference the model conditions only on the cells that are present. That is why the imputer needs no separate "missingness" handling: a row with holes is simply a shorter prompt.

Drop the prompt entirely — put *no* columns in the context — and the same conditional generates every column from scratch, so each row is a draw from the learned joint $p(\text{features}, \text{label})$. That turns the fitted model into a [tabular synthesizer](#tabular-synthesis) on top of the four estimators.

Because it is a real fine-tune of a real LM, the quality scales with the base model and the training budget. The defaults (`distilgpt2`, the smallest GPT-2) are chosen so examples run anywhere; if needed, swap in a larger model for better accuracy.

---

## Installation

```bash
pip install scikit-lm
```

The base install pulls only the light stack (numpy, pandas, scikit-learn, imbalanced-learn). To actually fine-tune and run a model you need **a backend extra**:

```bash
pip install "scikit-lm[hf]"          # Hugging Face / PyTorch backend (any platform)
pip install "scikit-lm[mlx]"         # MLX on Apple Silicon (Metal)
pip install "scikit-lm[mlx-cpu]"     # MLX on Linux, CPU
pip install "scikit-lm[mlx-cuda12]"  # MLX on Linux, NVIDIA (CUDA 12)
pip install "scikit-lm[mlx-cuda13]"  # MLX on Linux, NVIDIA (CUDA 13)
```

Optional extras:

| Extra        | Adds                                   | Enables                                                        |
|--------------|----------------------------------------|----------------------------------------------------------------|
| `hf`         | torch, transformers, peft, accelerate  | the Hugging Face backend (`HFBackend`), LoRA                   |
| `mlx`        | mlx (Metal), mlx-lm                    | the MLX backend (`MLXBackend`) on Apple Silicon               |
| `mlx-cpu`    | mlx (CPU), mlx-lm                      | the MLX backend on Linux, CPU                                 |
| `mlx-cuda12` | mlx (CUDA 12), mlx-lm                  | the MLX backend on Linux, NVIDIA (driver ≥ 550)              |
| `mlx-cuda13` | mlx (CUDA 13), mlx-lm                  | the MLX backend on Linux, NVIDIA (driver ≥ 580)              |
| `quant`    | bitsandbytes / mps-bitsandbytes        | 4-/8-bit quantized HF base weights (CUDA / Apple MPS)         |
| `hqq`      | hqq                                    | 2-/3-bit quantized HF base weights via HQQ (CUDA or CPU)      |
| `tqdm`     | tqdm                                   | live progress bars (`TqdmCallback`)                           |
| `rich`     | rich, plotext                          | live fine-tuning dashboard (`RichCallback`)                  |
| `optuna`   | optuna, optuna-integration             | `OptunaSearchCV` hyperparameter tuning                        |

Extras combine, e.g. `pip install "scikit-lm[hf,quant,tqdm]"`. The `all` extra pulls every optional dependency at once — platform markers keep it resolvable on any OS: `pip install "scikit-lm[all]"`.

Requires **Python ≥ 3.12**.

---

## The four estimators

All four share the same constructor knobs (model, backend, serializer, training, generation, LoRA, quantization, …) — documented under [Configuration](#configuration) — and differ only in their task-specific method. The examples below work on both Hugging Face and MLX versions.

### Classifier

`LanguageModelClassifier` conditions on all features and **ranks the fixed set of class labels by likelihood**. Because it scores a closed candidate set rather than free-generating, every prediction is a valid member of `classes_` and `predict_proba` is a genuine distribution.

```python
from sklm import LanguageModelClassifier

clf = LanguageModelClassifier(model="distilgpt2", random_state=0)
clf.fit(X_train, y_train)

clf.predict(X_test)         # labels from clf.classes_
clf.predict_proba(X_test)   # shape (n_rows, n_classes), columns ordered as clf.classes_
```

Scoring is deterministic, so the `GenerationConfig` sampling knobs (`temperature`, `top_p`, …) are inert here — only `inference_batch_size` matters. When the model assigns no finite likelihood to any candidate the row falls back to a uniform distribution; if it assigns infinite likelihood, all mass goes to those candidates.

### Regressor

`LanguageModelRegressor` conditions on all features and **generates the numeric target**. Greedy decoding would return the most likely single value (the mode), so `predict` instead draws `n_samples` completions per row and averages them — a Monte-Carlo estimate of the conditional mean.

```python
from sklm import GenerationConfig, LanguageModelRegressor

reg = LanguageModelRegressor(
    model="distilgpt2",
    generation=GenerationConfig(n_samples=10),
    random_state=0,
)
reg.fit(X_train, y_train)
reg.predict(X_test)
```

If *every* draw for a row comes back malformed after retries, `predict` raises `RuntimeError` rather than silently substituting a baseline — a model that can't produce valid numbers never masquerades as a working regressor.

### Imputer

`LanguageModelImputer` fits on the table as-is (missing cells are simply omitted from each row's serialization), then **fills every `NaN` by conditioning on that row's observed cells**. It implements the scikit-learn transformer API, so `fit_transform` works and it drops into a `Pipeline`.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklm import LanguageModelImputer

iris = load_iris(as_frame=True)
frame = iris.data.round(1)
frame["species"] = iris.target_names[iris.target]          # mixed numeric + categorical
corrupt = frame.mask(np.random.default_rng(0).random(frame.shape) < 0.15)

filled = LanguageModelImputer(random_state=0).fit_transform(corrupt)
```

Numeric and categorical columns are imputed by the same mechanism — no encoding — and cross-column structure is respected because the model conditions on the whole observed row. A DataFrame in returns a DataFrame out (same shape and row order); an array returns an array. A row whose missing cells stay malformed after retries raises `RuntimeError`.

### Oversampler

`LanguageModelOverSampler` implements the imbalanced-learn sampler API. For each under-represented class it **conditions generation on that class label and synthesizes the remaining features**, appending the new rows until the classes are balanced. Unlike SMOTE it operates on text, so categorical columns and feature correlations need no numeric encoding.

```python
from sklm import LanguageModelOverSampler

X_res, y_res = LanguageModelOverSampler(
    sampling_strategy="auto", random_state=0,
).fit_resample(X, y)
```

The `sampling_strategy` parameter is forwarded to imbalanced-learn (string, float, dict, or callable). Integer-typed feature columns are rounded so the restored dtype isn't truncated. If a class can't be filled within its attempt budget, `fit_resample` raises `RuntimeError`.

---

## The shared core: `TabularLanguageModel`

Every estimator is a thin adapter over one fitted object, `TabularLanguageModel`, which exposes the two conditional primitives directly. Reach for it when you want to query *any* column from *any* subset without committing to a single estimator's task.

```python
from sklearn.datasets import load_iris
from sklm import (
    TabularLanguageModel, HFBackend, JSONSerializer,
    ModelConfig, TrainingConfig, GenerationConfig,
)

iris = load_iris(as_frame=True)
frame = iris.data.round(1)
frame["species"] = iris.target_names[iris.target]

lm = TabularLanguageModel(
    backend=HFBackend(),
    serializer=JSONSerializer(),
    training=TrainingConfig(epochs=40),
    model_config=ModelConfig(model="distilgpt2"),
    random_state=0,
).fit(frame)

# Score a fixed candidate set — conditioning only on the petal measurements:
lm.predict_proba(
    {"petal length (cm)": 1.4, "petal width (cm)": 0.2},
    "species", list(iris.target_names),
)   # -> array of class probabilities

# Generate a numeric column — conditioning on the species instead:
lm.complete({"species": "setosa"}, ["petal length (cm)"], GenerationConfig())
#   -> {"species": "setosa", "petal length (cm)": 1.4}
```

| Method                 | Does                                                            |
|------------------------|----------------------------------------------------------------|
| `fit(frame, *, target_cols=…)` | fine-tune on the table; `target_cols` marks which columns are supervised under `loss_on_target_only` |
| `predict_proba(known, target, candidates)`     | rank `candidates` for `target` by likelihood (single row) |
| `predict_proba_many(knowns, target, candidates, generation)` | the same, batched across rows |
| `complete(known, targets, generation)`         | generate each `target` column in turn (single row) |
| `complete_many(knowns, targets, generation)`   | the same, batched across rows |

The completion methods return `None` for a row whose targets stay malformed after `max_retries` (default 15); the estimators turn that `None` into the loud `RuntimeError` described above.

### Tabular synthesis

The four estimators each fix *which* columns go in the prompt. Fix *none* of them and the same fitted model becomes a **tabular synthesizer**: generate every column from an empty context, so each row is a draw from the learned joint $p(\text{features}, \text{label})$ — the first column sampled from its marginal, every later one conditioning on the cells already produced. No new estimator class is involved; it is `complete_many` called with empty (or label-only) contexts.

```python
import pandas as pd
from sklm import TabularLanguageModel, GenerationConfig

lm = TabularLanguageModel(...).fit(frame)   # fit on the whole table, no target_cols
columns = list(frame.columns)

# Unconditional — sample whole rows from p(features, label):
rows = lm.complete_many(
    [{}] * 150,                       # empty context per row
    [columns] * 150,                  # produce every column, in order
    GenerationConfig(temperature=0.7),
)

# Conditional — pin a column and synthesize the rest (e.g. class-balanced rows):
features = [c for c in columns if c != "species"]
rows = lm.complete_many(
    [{"species": "setosa"}] * 50,
    [features] * 50,
    GenerationConfig(temperature=0.7),
)

synth = pd.DataFrame([r for r in rows if r is not None])
```

Sampling with `temperature > 0` is what gives the rows their diversity (greedy decoding would collapse every row to the same mode). Each result is a dict, or `None` if it stayed malformed after retries, so filter before building the frame. [`examples/08-synthesizer.py`](examples/08-synthesizer.py) runs the conditional path end to end and checks the synthesized per-feature moments and class balance against the real Iris table.

---

## Configuration

The estimators accept their hyperparameters as flat keyword arguments. The commonly-tuned knobs are covered below; every field is documented in full in the class docstrings (`help(LanguageModelClassifier)`, `help(TrainingConfig)`, …).

### Serialization

How a row becomes text is split into two orthogonal choices: the **structure** (`Serializer`) and the **number format** (`NumberFormat`).

```python
from sklm import LanguageModelClassifier, KeyValueSerializer, SpacedDigits

LanguageModelClassifier(
    serializer=KeyValueSerializer(number=SpacedDigits(max_decimals=2)),
)
```

**Structure** — pass `"json"` (default), `"key-value"`, or `"bracket"` for the plain-number versions, or a `Serializer` instance for full control:

| Serializer            | A row looks like                          |
|-----------------------|-------------------------------------------|
| `JSONSerializer`      | `{"age": 39, "city": "SP"}`               |
| `KeyValueSerializer`  | `age:39\|city:SP`                         |
| `BracketSerializer`   | `age[39] city[SP]`                        |

`KeyValueSerializer` takes custom `key_value_separator` / `pair_separator` (e.g. `" is "`, `";"`).

**Number format** — how numeric cells are rendered, composable with any structure:

- `PlainNumber` (default) — `25.7` → `"25.7"`; a float `100.0` → `"100.0"`, an int `100` → `"100"`. Optional `max_decimals` rounding.
- `SpacedDigits` — `25.7` → `"2 5 . 7"`. One token per digit, which helps the model treat numbers positionally.

The `max_decimals` *constructor* argument on the estimators (default `3`) only applies when you select a serializer by string; a `Serializer` instance carries its own number format.

A custom serializer just needs to implement the `Serializer` protocol (`serialize` / `prefix` / `split` / `encode_value` / `decode_value`); the invariants it must uphold are documented on the protocol.

### Training (`TrainingConfig`)

Fine-tuning hyperparameters. Held by the estimator as a nested, tunable object.

```python
from sklm import TrainingConfig

TrainingConfig(
    epochs=50,                 # passes over the rows
    batch_size=16,
    learning_rate="auto",      # 2e-5 full-weight, 2e-4 with LoRA; or pass a float
    lr_scheduler="cosine",     # "constant" | "linear" | "cosine"
    augmentation_factor=1,     # distinct column orders emitted per row each epoch
    loss_on_target_only=False, # supervise only the target column(s), not the context
)
```

Two knobs are specific to this library's mechanism:

- **`augmentation_factor`** — how many distinct column permutations to emit per row each epoch (a row with `m` present columns has at most `m!`). Raising it is a cheap form of data augmentation.
- **`loss_on_target_only`** — when `True`, the context tokens are masked out of the loss and the model is supervised only on the column(s) it must actually predict (the label for the classifier/regressor, the missing cells for the imputer). Inert for the oversampler.

Other fields cover the usual levers: `weight_decay`, `grad_accumulation_steps`, `warmup_ratio`, `max_grad_norm`, `optimizer`, `label_smoothing`, `neftune_noise_alpha`, `gradient_checkpointing`, `max_seq_length`, and `max_steps`. See the docstring for the full list and defaults.

### Generation (`GenerationConfig`)

Sampling hyperparameters for the generative estimators (regressor, imputer, oversampler) and the `TabularLanguageModel` completion methods.

```python
from sklm import GenerationConfig, aggregate_default

GenerationConfig(
    temperature=0.7,           # <= 0 is greedy
    top_p=1.0,                 # nucleus threshold; 1.0 disables
    top_k=0,                   # 0 disables
    max_new_tokens=None,       # token budget per generated value; None resolves to max_seq_length
    repetition_penalty=None,   # None disables
    inference_batch_size=None, # defaults to the training batch_size
    n_samples=1,               # draws per cell (regressor/imputer); scored column orders (classifier)
    permute_order=True,        # re-permute conditioning columns per draw/order when n_samples > 1
    aggregate=aggregate_default, # (draws, numeric) -> value; mean if numeric, else mode
    score_pool=None,           # classifier only: pool per-order distributions; None averages softmaxes
)
```

`inference_batch_size` controls how many prompts go to the backend per call (for both generation and scoring); leaving it `None` keeps the inference footprint in line with training. Results are **batch-size-invariant** — the batch size changes throughput, never the output.

`n_samples` and `permute_order` work together to ensemble over column order: the generative estimators draw `n_samples` completions per cell and collapse them with `aggregate` (default `aggregate_default` — the mean of numeric draws, the mode otherwise), while the classifier scores each candidate under `n_samples` distinct column orders and pools the per-order distributions with `score_pool` (default `None`, which averages the per-order softmaxes). With `permute_order` on (default), the orders are re-permuted per draw so the samples marginalize over feature order rather than fixing one; it has no effect when `n_samples == 1`.

### Discretization (`DiscretizationConfig`)

The regressor and imputer normally **generate** a numeric value as text and parse it back. `DiscretizationConfig` switches the numeric path to **scoring** instead: it ranks a fixed set of candidate values by conditional likelihood (the same mechanism the classifier uses) and reduces the resulting distribution to one number. The candidates are real observed values of the target, so the model only ever scores tokens it saw during fine-tuning — deterministic, and often sharper than sampling when the numeric support is small and discrete.

```python
from sklm import DiscretizationConfig, LanguageModelRegressor

LanguageModelRegressor(
    discretization=DiscretizationConfig(
        bins=0.3,              # 0/0.0 (default) keeps generation; int K = K candidates;
                               #   float in (0, 1] = that fraction of the distinct support
        strategy="quantile",   # "quantile" (equal-mass) | "uniform" (equal-width)
        representative="median", # candidate per partition: "median" | "mode" | "mean"
        estimate="mean",       # collapse the scored distribution: "mean" (expectation) | "mode" (argmax)
    ),
)
```

Where it applies:

- **Regressor** — pass a single `DiscretizationConfig`; it discretizes the numeric target. Default off (`bins=0`).
- **Imputer** — pass a single `DiscretizationConfig` (applies to every numeric column) or a `Mapping[str, DiscretizationConfig]` for per-column control; columns absent from the mapping stay on the generative path. Categorical cells always generate.

`bins` is the on/off switch as well as the candidate count: `0` (default) keeps the generative path; an `int` `K` scores `K` candidates (capped at the number of distinct observed values); a `float` in `(0, 1]` keeps that fraction of the distinct support (`1.0` = every distinct value).

### LoRA & model loading

Model-loading options are passed as flat estimator arguments and reassembled internally into a `ModelConfig`:

```python
from sklm import LanguageModelClassifier, LoRAConfig

LanguageModelClassifier(
    model="gpt2-large",
    lora=LoRAConfig(rank=16, alpha=32, dropout=0.0),  # None = full-weight fine-tune
    quantization="4bit",                              # "4bit" | "8bit" | None
    precision="bf16",                                 # "fp32" | "bf16" | "fp16"
    device="auto",                                    # "cuda" | "mps" | "cpu" | "auto"
    tokenizer=None,
    trust_remote_code=False,
    attn_implementation=None,                         # e.g. "flash_attention_2"
)
```

`LoRAConfig` additionally supports `target_modules`, `rank_pattern`, `alpha_pattern`, `rslora`, and `dora`. For `target_modules`, the portable choice is `"all-linear"` or `None` (auto-discovery) — explicit module names differ between backends (see below).

---

## Backends

A **backend** is the execution engine that actually fine-tunes, generates, and scores. It is the only abstraction the rest of the library depends on, which is what keeps torch/mlx optional. Select one with the `backend` argument:

| `backend=`        | Engine                              | Needs       |
|-------------------|-------------------------------------|-------------|
| `"huggingface"`   | transformers + peft (`HFBackend`)   | `[hf]`      |
| `"mlx"`           | mlx-lm (`MLXBackend`)               | `[mlx]` / `[mlx-cpu]` / `[mlx-cuda12]` / `[mlx-cuda13]` |
| `"auto"` (\*)     | the best installed stack            | either      |
| a `LanguageModelBackend` instance | injected directly   | —           |

(\*) `"auto"` resolves to whichever backend is installed, by platform-aware preference. On macOS it picks MLX (Metal). Elsewhere it walks HF-GPU → MLX-GPU → HF-CPU → MLX-CPU — an accelerated backend first, and HF ahead of MLX within a tier. The default across the estimators is `"huggingface"`.

A few cross-backend gotchas worth knowing:

- **Quantization** uses bitsandbytes on CUDA / mps-bitsandbytes on Apple MPS for the HF backend (CPU is unsupported); the MLX backend converts to its native 4-/8-bit format at load time, cached under `~/.cache/sklm/mlx`.
- **`LoRAConfig.target_modules`** matching differs (HF matches a name suffix like `"c_attn"`; MLX matches the in-block path like `"attn.c_attn"`). Use `"all-linear"` / `None` to stay portable.
- **MLX model loading** — some HF repos aren't mlx-loadable. distilgpt2's own repo isn't; use an mlx-loadable mirror such as `gabfssilva/distilgpt2` or `openai-community/gpt2`.

---

## scikit-learn integration

The estimators honor the full scikit-learn estimator contract, and the config objects (`TrainingConfig`, `GenerationConfig`, `LoRAConfig`, `DiscretizationConfig`, `QuantizationConfig`) subclass `BaseEstimator`. That means `clone`, `set_params`, `Pipeline`, and any cross-validation search work out of the box, with nested fields addressable through the usual `__` separator:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklm import LanguageModelClassifier

pipe = Pipeline([
    ("scaler", StandardScaler().set_output(transform="pandas")),
    ("lm", LanguageModelClassifier(random_state=0)),
])

search = GridSearchCV(pipe, {
    "lm__precision": ["fp32", "bf16"],     # a flat model-loading field
    "lm__training__epochs": [2, 4],        # a field of the nested TrainingConfig
    "lm__lora__rank": [8, 16],             # a field of the nested LoRAConfig
})
search.fit(X_train, y_train)
```

The fixed hyperparameters are declared once on the estimator; only the swept fields go in the grid. The same pattern drives Optuna's `OptunaSearchCV` — see [`examples/06-optuna-search.py`](examples/06-optuna-search.py).

Input handling follows scikit-learn conventions: DataFrame columns are matched by name and reordered to the training order at predict time; array input is accepted too. Fitted attributes end with `_` (`classes_`, `n_features_in_`, `feature_names_in_` — the last only for DataFrame input).

---

## Callbacks

Pass a `callbacks=` object to watch fitting and inference. `Callback` is a concrete base class that folds the granular event stream into a running `TrainingState` and dispatches a single `on_event(state, event)` — subclass it and override `on_event`. Three implementations ship:

```python
from sklm import LanguageModelClassifier, LoggingCallback, RichCallback, TqdmCallback

# Live progress bars (needs the [tqdm] extra); print a few serialized rows at fit start:
LanguageModelClassifier(callbacks=TqdmCallback(n_train_examples=5))

# A live dashboard with an in-terminal loss curve (needs the [rich] extra):
LanguageModelClassifier(callbacks=RichCallback())

# Or route every event through the standard logging module:
import logging
logging.basicConfig(level=logging.INFO)
LanguageModelClassifier(callbacks=LoggingCallback())
```

Every change arrives at `on_event` as an `Event` — `FitStart`, `TrainExamples`, `TrainReport`, `EvalReport`, `Memory`, `FitEnd`, `PredictStart`, `RowEnd`, `PredictEnd`, `Generation`, `Score`, `Retry` — alongside the running `TrainingState` (loss series, derived epoch, peak memory, …). `match` on the event to react; the state carries the aggregated history so a renderer never re-derives it:

```python
from sklm import Callback, Event, TrainingState, TrainReport

class PrintLoss(Callback):
    def on_event(self, state: TrainingState, event: Event) -> None:
        if isinstance(event, TrainReport):
            print(f"step {state.step}: loss={state.loss:.4f} (epoch {state.epoch})")
```

Each shipped dashboard takes `n_train_examples` to preview the exact text the model trains on each epoch — useful for sanity-checking a serializer (`LoggingCallback` and `TqdmCallback` default to `0`; `RichCallback` previews `5`).

---

## Requirements

- Python ≥ 3.12
- A backend extra to fine-tune and run a model: `[hf]` (any platform), or an MLX variant — `[mlx]` (Apple Silicon / Metal), `[mlx-cpu]`, `[mlx-cuda12]` or `[mlx-cuda13]` (Linux)

---

## License

Released under the [MIT License](LICENSE).
