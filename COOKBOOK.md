# The scikit-lm Cookbook

A single-file guide to using `scikit-lm` and understanding how it works. The
short sections at the top (1–2) carry all the theory; everything after that is
practical and refers back to them. Runnable, end-to-end versions of most
recipes live in [`examples/`](examples/) as Jupyter notebooks.

## Contents

1. [What is scikit-lm?](#1-what-is-scikit-lm)
2. [The one mechanism](#2-the-one-mechanism)
3. [Installation and backends](#3-installation-and-backends)
4. [Quickstart: classification on Iris](#4-quickstart-classification-on-iris)
5. [The estimators](#5-the-estimators)
6. [Serialization](#6-serialization)
7. [Configuration](#7-configuration)
8. [scikit-learn integration](#8-scikit-learn-integration)
9. [Observability: callbacks](#9-observability-callbacks)
10. [Below the estimators: TabularLanguageModel](#10-below-the-estimators-tabularlanguagemodel)
11. [Practical guidance](#11-practical-guidance)

[Appendix A. Public API index](#appendix-a-public-api-index) ·
[Appendix B. HF vs. MLX cheat sheet](#appendix-b-hf-vs-mlx-cheat-sheet)

---

## 1. What is scikit-lm?

### 1.1 The idea in one paragraph

`scikit-lm` provides scikit-learn estimators whose underlying model is a
fine-tuned autoregressive language model. Each tabular row is serialized into
a short text (by default a JSON object), a small LM is fine-tuned on those
texts, and predictions are made by either *generating* text from a partial row
or *scoring* candidate continuations by likelihood. Four estimators ship on
top of this: a classifier, a regressor, a missing-value imputer, and an
imbalanced-learn oversampler. They are thin adapters over one shared core —
the same fitted model, prompted differently.

```python
from sklm import LanguageModelClassifier

clf = LanguageModelClassifier("distilgpt2")
clf.fit(X_train, y_train)          # fine-tunes the LM on serialized rows
clf.predict(X_test)                # scores each class label by likelihood
```

### 1.2 When to use it (and when not to)

The approach plays to a language model's strengths, and those strengths define
the use cases:

- **Mixed-type tables.** Categorical and text columns need no encoding — they
  are just text. There is no one-hot explosion and no ordinal assumption.
- **Conditioning on arbitrary subsets of columns.** The training scheme
  (section 2.2) produces a model that can predict *any* column from *any*
  subset of the others. This is what makes a single fitted model serve as
  classifier, imputer, and synthesizer at once.
- **Missing data as a first-class citizen.** Missing cells are simply omitted
  from a row's serialization, at training and at inference. No imputation is
  required before fitting.
- **Small-to-medium tables.** Fine-tuning a small LM on a few hundred to a few
  thousand rows is practical on a laptop GPU; the per-row cost of inference is
  a forward pass (or several), not a tree traversal.

It is the wrong tool when you need millisecond inference over millions of
rows, when your features are purely numeric and a gradient-boosted tree
already works, or when you cannot afford a fine-tune per `fit` call (every
`fit` — including every cross-validation fold — trains a model; see
section 8.4).

## 2. The one mechanism

Everything in the library reduces to one mechanism. This section explains it
once; the rest of the cookbook refers back here.

### 2.1 Rows as text: serialization

A `Serializer` turns a (possibly partial) row into text. The default renders
a JSON object:

```text
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "species": "setosa"}
```

Three structural formats ship built in — JSON, `key:value|` pairs, and
`col[value]` brackets — and how *numbers* are rendered is an orthogonal choice
(plain text vs. one token per digit). Section 6 covers all of them, plus the
contract a custom serializer must satisfy. Two properties matter right now:

- A serializer can also produce a **prefix**: the row's text cut off right
  before a target column's value, e.g.
  `{"sepal_length": 5.1, "species": ` — which is exactly a prompt asking the
  model for that value.
- **Missing cells are never serialized.** A row with NaNs simply omits those
  key/value pairs, so the model trains and is prompted only on observed cells.

### 2.2 Column-order permutation: learning `p(any column | any subset)`

A standard fine-tune on serialized rows would teach the model one thing:
predict the columns in the order they appear. If every training example reads
`{"sepal_length": 5.1, "sepal_width": 3.5, "species": "setosa"}`, the model
learns $p(\texttt{species} \mid \texttt{sepal\_length}, \texttt{sepal\_width})$
— and nothing else. That is
enough for a classifier with a fixed target, but it cannot impute
`sepal_width` from the other two columns, because that conditional was never
trained.

scikit-lm's training loop therefore **shuffles the column order of every row,
throughout training**. One epoch the model sees the row above; another epoch
it sees `{"species": "setosa", "sepal_length": 5.1, "sepal_width": 3.5}`.
Since an autoregressive model predicts each token from everything to its
left, each ordering trains a different factorization of the joint
distribution. Across many orderings, the model converges toward a single
object:

$$p(\text{any column} \mid \text{any subset of the other columns})$$

This is the property every estimator relies on. The classifier conditions on
all features and asks about the label; the imputer conditions on whatever
cells a row happens to have and asks about the missing ones; the oversampler
conditions on a class label alone and asks for everything else. None of these
require separate models or separate training runs — they are different
prompts against the same fitted model.

Two `TrainingConfig` knobs shape this behavior:

- `augmentation_factor` — how many *distinct* column orders of each row are
  emitted per epoch. The default `1` gives one fresh permutation per row per
  epoch (a different one each epoch); raising it multiplies the epoch's data
  with re-ordered copies, capped at the $m!$ possible orders of a row with
  $m$ present columns.
- `loss_on_target_only` — when `True`, the target columns are serialized last
  and the context tokens are masked out of the loss, so the model is
  supervised only on what it must predict at inference. The default `False`
  supervises every token, which is what trains the full conditional structure
  (and is the only mode that makes sense for the oversampler, which must
  generate whole rows).

### 2.3 Two inference modes: scoring vs. generating

Once fitted, the model can answer a question about a cell in two ways, and
the distinction runs through the entire library.

**Scoring** asks: *given this prompt, how likely is this specific
continuation?* The serialized row is cut right before the target value, each
candidate value is appended in turn, and the model computes the mean
per-token log-likelihood of each candidate $c$, tokenized as $(y_1, \dots, y_{T_c})$:

$$\operatorname{score}(c) = \frac{1}{T_c} \sum_{t=1}^{T_c} \log p_\theta\bigl(y_t \mid \text{prompt},\, y_{<t}\bigr)$$

A softmax over the candidates' scores then gives a proper probability
distribution. No text is generated, so nothing can be malformed, and the answer set is
closed: the result is always one of the candidates you provided. This is how
`LanguageModelClassifier` works — the candidates are `classes_` — and it is
why its `predict_proba` is well defined.

Scoring is deterministic by construction: the scoring primitive
(`backend.score(prompts, continuations)`) does not even receive the
generation config, so the stochastic decoding knobs — `temperature`,
`top_p`, `top_k`, `repetition_penalty`, `max_new_tokens` — *cannot* affect
any scoring path. The `GenerationConfig` fields that **do** apply to scoring
are `inference_batch_size` (how many prompt/candidate pairs per backend
call), and the order-marginalization trio `n_samples` / `permute_order` /
`score_pool` (section 7.2.2).

**Generating** asks: *given this prompt, what comes next?* The model samples
tokens until the serializer can parse a value out of them. The answer set is
open — any number, any string — which is exactly what the regressor, the
imputer, and the oversampler need. The price is that sampled text can fail to
parse; scikit-lm retries (up to 15 times per value), and if every attempt
stays malformed it raises `RuntimeError` rather than silently substituting a
baseline value (section 5.5). One special case: with `temperature <= 0`
(greedy decoding) every retry would reproduce the same text byte for byte, so
a single attempt is made.

A useful rule of thumb falls out of this: **when the set of valid answers is
small and known, score; when it is open-ended, generate.** The library
applies the rule for you — but it also lets you move the regressor (and the
imputer's numeric columns) from the generating column to the scoring column
via `DiscretizationConfig` (section 5.2.2), which is often the more stable
choice for numeric targets.

### 2.4 How the four estimators map onto the mechanism

| Estimator | Prompt (conditions on) | Asks for | Mode |
|---|---|---|---|
| `LanguageModelClassifier` | all features | the label, ranked over `classes_` | scoring |
| `LanguageModelRegressor` | all features | the numeric target | generating (or scoring, with discretization) |
| `LanguageModelImputer` | a row's observed cells | each missing cell | generating (or scoring per numeric column) |
| `LanguageModelOverSampler` | a class label | every feature | generating |

All four wrap the same fitted `TabularLanguageModel` (section 10), exposed on
a fitted estimator as `estimator.lm_`.

## 3. Installation and backends

### 3.1 Extras

The base package keeps its dependencies light (numpy, pandas, scikit-learn,
imbalanced-learn); the deep-learning stack is an explicit opt-in:

```bash
pip install scikit-lm                  # estimators only; no backend yet

pip install "scikit-lm[hf]"            # Hugging Face / PyTorch backend (any platform)
pip install "scikit-lm[mlx]"           # MLX on Apple Silicon (Metal)
pip install "scikit-lm[mlx-cpu]"       # MLX on Linux, CPU
pip install "scikit-lm[mlx-cuda12]"    # MLX on Linux, NVIDIA (CUDA 12)
pip install "scikit-lm[mlx-cuda13]"    # MLX on Linux, NVIDIA (CUDA 13)
```

Extras combine (`pip install "scikit-lm[hf,quant,tqdm]"`), and the `all`
extra pulls every optional dependency with platform markers keeping it
resolvable on any OS.

### 3.2 Choosing a backend

A backend is a pure execution engine implementing three primitives — `fit`,
`generate`, `score` — behind the `LanguageModelBackend` protocol. It stores
no hyperparameters of its own; the estimator hands it the configs at call
time and the base model is reloaded on every `fit`. Two implementations ship:

- `HFBackend` — torch + transformers; CUDA, Apple MPS, or CPU.
- `MLXBackend` — mlx + mlx-lm; Metal on Apple Silicon, CUDA or CPU on Linux.

Every estimator takes a `backend` parameter: `"huggingface"` (the default),
`"mlx"`, `"auto"`, or a `LanguageModelBackend` instance. `"auto"` picks from
what is installed, by platform-aware preference: MLX on macOS; elsewhere
HF-GPU → MLX-GPU → HF-CPU → MLX-CPU.

```python
clf = LanguageModelClassifier("distilgpt2", backend="auto")
```

Backend-specific behavior differences (quantization widths, LoRA module
naming, training internals) are collected in Appendix B.

### 3.3 Picking a base model

The `model` parameter — the first positional argument of every estimator — is
a Hugging Face model id or local path. The default is `"distilgpt2"`, an
82M-parameter model that fine-tunes in minutes on consumer hardware and is a
sensible starting point for tables with up to a few dozen columns. Anything
loadable as an autoregressive causal LM works; larger models buy accuracy at
the cost of fit and inference time (section 11.1).

One MLX caveat: the upstream `distilgpt2` repository is not mlx-loadable; use
an MLX-compatible mirror such as `gabfssilva/distilgpt2` (or
`openai-community/gpt2`) when running on the MLX backend.

## 4. Quickstart: classification on Iris

A complete, runnable example ([notebook 01](examples/01-iris-classifier.ipynb)):

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklm import LanguageModelClassifier, TrainingConfig

X, y = load_iris(return_X_y=True, as_frame=True)
y = y.map(dict(enumerate(load_iris().target_names)))   # readable labels help the LM

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = LanguageModelClassifier(
    "distilgpt2",
    training=TrainingConfig(epochs=30, batch_size=16),
    random_state=42,
)
clf.fit(X_train, y_train)

print(accuracy_score(y_test, clf.predict(X_test)))
print(clf.predict_proba(X_test[:3]))   # rows sum to 1, columns ordered as clf.classes_
```

What happens, step by step:

1. `fit` appends `y` as one more column, serializes each row to JSON, and
   fine-tunes distilgpt2 on the texts — re-permuting every row's column order
   each epoch (section 2.2). A progress dashboard appears automatically
   (section 9).
2. `predict_proba` serializes each test row into a prompt ending right before
   the label value, scores every member of `classes_` as the continuation,
   and softmaxes the per-candidate log-likelihoods.
3. `predict` is the argmax of those probabilities, so predictions are always
   valid labels — the model cannot hallucinate a class that doesn't exist.

Two small practices visible above carry real weight: **readable string labels**
(`"setosa"`, not `0`) give the model meaningful tokens to score, and
**DataFrame input** preserves column names, which become the JSON keys the
model learns. Plain arrays work too, but columns are then named `x0, x1, ...`.

## 5. The estimators

All four estimators share the same constructor surface — `model` plus the
keyword knobs of section 7 — and the scikit-learn conventions: fitted
attributes end in `_`, `feature_names_in_` is set for DataFrame input,
`get_params`/`set_params`/`clone` work, and string/categorical/NaN input is
declared via estimator tags. What differs is each estimator's prompt/target
split, summarized in section 2.4.

### 5.1 LanguageModelClassifier

```python
from sklm import LanguageModelClassifier

clf = LanguageModelClassifier("distilgpt2").fit(X_train, y_train)
clf.predict(X_test)          # labels drawn from clf.classes_
clf.predict_proba(X_test)    # (n_rows, n_classes), rows sum to 1
```

The classifier is the pure-scoring estimator. At `fit`, the labels become one
more column of the training table; at predict time, each row is serialized
into a prefix ending right before the label and every member of `classes_` is
scored as the continuation (mean per-token log-likelihood, then softmax). The
consequences:

- `predict_proba` is well defined and the columns are exactly `classes_`
  (sorted unique labels from `fit`).
- Prediction is deterministic. The stochastic decoding fields of
  `GenerationConfig` have no effect here (section 2.3); the fields that *do*
  matter are `inference_batch_size` and the order-marginalization trio
  `n_samples` / `permute_order` / `score_pool` (section 7.2.2).
- Rows with missing feature values are fine: NaN cells are dropped from the
  prompt and the model conditions on what remains.

`predict` and `predict_proba` accept a per-call `generation=` override that
applies to that call only, without mutating the estimator:

```python
from sklm import GenerationConfig

# Marginalize each candidate's likelihood over 8 feature orders, this call only.
clf.predict_proba(X_test, generation=GenerationConfig(n_samples=8, permute_order=True))
```

### 5.2 LanguageModelRegressor

#### 5.2.1 The generative path (default)

```python
from sklm import LanguageModelRegressor, GenerationConfig

reg = LanguageModelRegressor(
    "distilgpt2",
    generation=GenerationConfig(n_samples=10),   # average 10 draws per row
).fit(X_train, y_train)
reg.predict(X_test)
```

The regressor conditions on all features and *generates* the numeric target.
Sampling at `temperature > 0` draws from the model's predictive distribution
$p(y \mid x)$; a single draw is one sample from it, which is noisy. `predict`
therefore draws `generation.n_samples` values per row and averages them — a
Monte-Carlo estimate of the conditional mean:

$$\hat{y} = \frac{1}{n} \sum_{i=1}^{n} y^{(i)}, \qquad y^{(i)} \sim p(y \mid x) \;\xrightarrow{\;n \to \infty\;}\; \mathbb{E}[y \mid x]$$

The default `n_samples=1` takes a single draw; raise it for stabler
predictions at proportional cost.

Greedy decoding (`temperature=0`) is not the fix it appears to be: it returns
the *mode* of the token-level distribution, not the mean of $p(y \mid x)$, and
collapses all `n_samples` draws into identical values.

#### 5.2.2 The scoring path: discretization

For numeric targets the generative path has a structural weakness — the model
must emit a well-formed number token by token, and the average of free-form
draws can be unstable. `DiscretizationConfig` switches the regressor to the
classifier's mechanism:

```python
from sklm import LanguageModelRegressor, DiscretizationConfig

reg = LanguageModelRegressor(
    "distilgpt2",
    discretization=DiscretizationConfig(bins=20),
).fit(X_train, y_train)
reg.predict(X_test)   # scores 20 candidate values, returns the expectation
```

With `bins` non-zero, `predict` builds a candidate set of **real observed
target values** (kept in-distribution, so the model scores tokens it saw
during fine-tuning), scores each candidate's likelihood per row, and reduces
the resulting distribution to one number. The knobs:

- `bins` — the on/off switch and the candidate count. An int $K$ keeps $K$
  candidates; a float in $(0, 1]$ keeps that fraction of the distinct
  observed values (`1.0` = the full support). `0` (default) keeps the
  generative path.
- `strategy` — how the observed support is partitioned before one
  representative is drawn per partition: `"quantile"` (default; equal-mass,
  more resolution where data concentrates) or `"uniform"` (equal-width).
- `representative` — the value taken from each partition: `"median"`
  (default) or `"mode"` (both real observed values), or `"mean"` (synthetic;
  may serialize to tokens the model never emitted).
- `estimate` — how the scored distribution collapses: `"mean"` (default) is
  the probability-weighted expectation over the candidates,
  $\hat{y} = \sum_k p_k\, c_k$ — smooth; `"mode"` is the single most likely
  candidate, $\hat{y} = c_{\arg\max_k p_k}$ — sharp. Match this to your loss:
  expectation-like targets (RMSE) want `"mean"`, exact-hit metrics want
  `"mode"`.
- `sharpness` — a temperature on the scored distribution, applied before the
  `"mean"` estimate: $p_k \mapsto p_k^{\alpha} / \sum_j p_j^{\alpha}$. `1.0`
  (default) keeps the distribution as scored; larger values concentrate mass
  on the top candidates, sliding the expectation continuously from the plain
  mean toward the argmax (`"mode"` itself is invariant — an argmax does not
  move under a monotone power). The failure modes it mediates: `"mode"`
  discards the mass parked on the *neighbors* of the right value, while the
  plain mean lets the long tail of implausible candidates drag the estimate
  toward the column's center — a moderate $\alpha$ (2–8) suppresses the tail
  and keeps the neighborhood structure, which is exactly what an
  *underconfident* model needs. Rule of thumb: sharpen targets that the other
  columns predict well (a quick CV $R^2$ of the target regressed on the rest
  is a serviceable proxy); on a genuinely noisy target the spread *is* the
  honest answer, and tempering it fabricates confidence — keep `1.0`.

The scoring path always yields a value (a distribution always exists), so it
never raises the malformed-generation `RuntimeError`. Both `discretization=`
and `generation=` can be overridden per call:

```python
reg.predict(X_test, discretization=DiscretizationConfig(bins=50, estimate="mode"))
```

— letting you fit once and compare decoders afterwards ([notebook
02](examples/02-autompg-regressor.ipynb)).

### 5.3 LanguageModelImputer

```python
import numpy as np
from sklm import LanguageModelImputer

imp = LanguageModelImputer("distilgpt2").fit(X_with_nans)
X_filled = imp.transform(X_with_nans)   # same shape, NaNs filled
```

The imputer is where "missing cells are never serialized" pays off twice:

- **`fit` uses the data as-is.** Rows with NaNs train on their observed cells
  only — no pre-imputation, no row dropping. Columns containing any NaN are
  registered as targets for `loss_on_target_only` masking (if enabled).
- **`transform` fills each row by conditioning on that row's own observed
  cells.** Targets are filled sequentially within a row, each conditioning on
  the prior fills; `generation.n_samples` draws per cell are aggregated
  (mean for numeric cells, mode otherwise — customizable via
  `generation.aggregate`).

It is a standard scikit-learn transformer (`OneToOneFeatureMixin`), so it
slots into a `Pipeline` and returns DataFrame-in → DataFrame-out with the
caller's column order restored.

Numeric columns can be moved to the scoring path per column, with the same
`DiscretizationConfig` as the regressor — a single config applies to every
numeric column, a mapping picks columns individually (absent columns
generate; categorical columns always generate):

```python
from sklm import DiscretizationConfig

imp = LanguageModelImputer(
    "distilgpt2",
    discretization={
        "age": DiscretizationConfig(bins=30),                  # scored
        "income": DiscretizationConfig(bins=0.5, estimate="mode"),  # scored
        # other columns: generated
    },
).fit(X_with_nans)
```

Scored cells are deterministic and never fail; only generated cells can
exhaust their retries and raise. `transform` accepts per-call `generation=`
and `discretization=` overrides, like the regressor. See [notebook
03](examples/03-iris-imputer.ipynb).

### 5.4 LanguageModelOverSampler

```python
from sklm import LanguageModelOverSampler

sampler = LanguageModelOverSampler(sampling_strategy="auto", model="distilgpt2")
X_res, y_res = sampler.fit_resample(X_train, y_train)
```

The oversampler implements imbalanced-learn's sampler API: `fit_resample`
fine-tunes on the labeled table, then, for each class that needs more rows,
prompts the model with **only the class label** and generates every feature —
the row synthesizer of section 2.4. `sampling_strategy` is forwarded to
imbalanced-learn unchanged (`"auto"`, a float ratio, a dict of per-class
counts, or a callable).

Compared to SMOTE, generation happens in text space: categorical columns need
no numeric encoding, and feature correlations come from the model rather than
from linear interpolation between neighbors. Malformed rows are discarded and
regenerated within an attempt budget (5× the requested count, plus a margin);
if the budget runs out before the quota is met, `fit_resample` raises. The
original rows pass through untouched — only the synthetic rows are appended.
Integer-typed columns are rounded so imbalanced-learn can restore their
dtype. See [notebook 04](examples/04-imbalanced-oversampler.ipynb), and
[notebook 08](examples/08-synthesizer.ipynb) for using the same mechanism as
a general synthesizer.

Note that `n_samples` is inert here — each draw *is* a distinct synthetic
row, so there is nothing to aggregate — and `loss_on_target_only` is ignored
with a warning (the features are the output; there is no fixed target column
to supervise).

### 5.5 Failure is loud

A design rule worth knowing before you ship: **the generative estimators
never fall back silently.** Each generated value gets up to 15 attempts
(`max_retries` on the core model); a regressor row whose every draw is
malformed, an imputer row whose generated cell never parses, an oversampler
that cannot meet its quota — all raise `RuntimeError` with a message naming
the row and the retry budget. A model that cannot produce valid values never
masquerades as a working estimator, and the failure points at the real
problem (almost always an under-trained model — too few epochs, too small a
model, or a serializer mismatch). Scored cells are exempt: scoring always
yields a distribution, so the classifier and the discretized paths cannot
raise this.

## 6. Serialization

### 6.1 Built-in formats

The `serializer` parameter takes a string selector or an instance:

| Selector | Class | One row looks like |
|---|---|---|
| `"json"` (default) | `JSONSerializer` | `{"age": 39, "city": "SP"}` |
| `"key-value"` | `KeyValueSerializer` | `age:39\|city:SP` |
| `"bracket"` | `BracketSerializer` | `age[39] city[SP]` |

JSON is the safest default: categorical values are quoted, so any string —
including one containing `|`, `]`, or `,` — round-trips. The key-value and
bracket formats write categorical values verbatim, which makes them a few
tokens cheaper per row but means a value containing the pair separator (or
`]`) cannot be recovered. Pick them when your categories are clean and your
rows are long; the instances also expose the separators:

```python
from sklm import KeyValueSerializer

ser = KeyValueSerializer(key_value_separator=" is ", pair_separator="; ")
# age is 39; city is SP
```

`max_decimals` (default 3) rounds numeric cells at serialization — fewer
digits, fewer tokens, less for the model to memorize. It applies to the
string selectors only; a serializer *instance* carries its own number format.

### 6.2 Number formats

How a numeric cell becomes text is orthogonal to row structure, and every
built-in serializer composes a `NumberFormat`:

- `PlainNumber` (default) — `25.7` → `"25.7"`.
- `SpacedDigits` — `25.7` → `"2 5 . 7"`: one token per digit, which keeps the
  tokenizer from merging digit runs and helps small models treat numbers
  positionally.

```python
from sklm import JSONSerializer, SpacedDigits

ser = JSONSerializer(number=SpacedDigits(max_decimals=2))
# {"age": 3 9, "city": "SP"}   <- not valid JSON, and that's fine:
# decoding reads each value up to its delimiter, never the whole object.
```

### 6.3 Writing a custom serializer

`Serializer` is a protocol — implement five methods, no base class required:

```python
class Serializer(Protocol):
    def serialize(self, fields: Sequence[Field]) -> str: ...
    def prefix(self, known: Sequence[Field], target: object) -> str: ...
    def split(self, context: Sequence[Field], target: Sequence[Field]) -> tuple[str, str]: ...
    def encode_value(self, value: object, *, numeric: bool) -> str: ...
    def decode_value(self, text: str, *, numeric: bool) -> object | None: ...
```

Each `Field` carries `(name, value, numeric)`. Four invariants make the
mechanism work, and breaking any of them silently breaks training or scoring:

1. **`prefix(known, target)` ends exactly where `target`'s value would
   begin.** The model is scored/generates from that boundary; a trailing
   space too many shifts every continuation.
2. **`encode_value(v)` is exactly the text that follows the prefix.**
   Scoring concatenates `prefix + encode_value(candidate)`; any mismatch with
   how `serialize` renders the same cell corrupts the likelihood.
3. **`split(ctx, tgt)` satisfies `prompt + completion ==
   serialize(ctx + tgt)`.** Backends locate the loss-masking boundary by
   token prefix, which requires byte-identical concatenation.
4. **`decode_value` returns `None` on malformed text** — never raises, never
   guesses. `None` is what triggers the retry loop.

These invariants are enforced by the test suite for the built-ins
(`tests/test_serialize.py`, `tests/test_contract.py`); run a custom
serializer through the same style of round-trip checks before trusting it.

## 7. Configuration

Every estimator shares one constructor surface: `model` (positional) plus
keyword knobs. The four config objects — `TrainingConfig`,
`GenerationConfig`, `LoRAConfig`, `DiscretizationConfig` — are themselves
scikit-learn estimators, so they participate in `get_params`/`set_params`
with `__` addressing (section 8).

```python
from sklm import (
    LanguageModelClassifier, TrainingConfig, GenerationConfig, LoRAConfig,
)

clf = LanguageModelClassifier(
    "Qwen/Qwen2.5-0.5B",
    backend="auto",
    training=TrainingConfig(epochs=20, batch_size=32, validation_split=0.1,
                            early_stopping_patience=3),
    generation=GenerationConfig(inference_batch_size=64),
    lora=LoRAConfig(rank=16),
    quantization="4bit",
    precision="bf16",
    random_state=42,
)
```

### 7.1 TrainingConfig

Fine-tuning hyperparameters, handed to the backend at `fit`. The defaults are
deliberately plain — 50 epochs, batch size 16, AdamW, cosine schedule — and
the learning rate is `"auto"`: `2e-5` for full-weight fine-tuning, `2e-4`
when LoRA is enabled. Selected knobs beyond the obvious ones:

- `validation_split` / `stratify` — hold out a fraction of rows each `fit`
  and report validation loss through the callbacks. The hold-out stratifies
  on the target (binning it into quantiles when numeric, so regression
  targets stratify like class labels), falling back to a random split with a
  warning when stratification is infeasible.
- `early_stopping_patience` — stop after this many consecutive validations
  without improvement, restoring the best checkpoint. Requires
  `validation_split > 0`.
- `checkpoint_steps` / `checkpoint_dir` — periodic checkpoints; by default
  they live in a temporary directory that only serves early stopping's
  best-model restore.
- `max_seq_length` — `None` (default) is measured at `fit`: the longest
  serialized training row, rounded up to a multiple of 8, so nothing is
  truncated. Set it explicitly only to cap memory; over-long rows are then
  truncated from the right, which drops trailing columns.
- `augmentation_factor`, `loss_on_target_only` — the permutation knobs of
  section 2.2.
- Capacity/memory levers: `grad_accumulation_steps`, `gradient_checkpointing`,
  8-bit optimizers (`adamw_8bit`, `paged_adamw_8bit`; CUDA only, plain AdamW
  elsewhere), `neftune_noise_alpha`, `label_smoothing`, `weight_decay`,
  `warmup_ratio`, `max_grad_norm`, `max_steps`.

### 7.2 GenerationConfig

One config drives both inference modes; which fields apply where is the
single most common point of confusion, so here it is explicitly:

| Field | Generating (regressor, imputer, oversampler) | Scoring (classifier, discretized paths) |
|---|---|---|
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_new_tokens` | yes | **inert** — scoring never sees them |
| `n_samples` | draws aggregated per cell (inert for the oversampler) | column orders to marginalize over (needs `permute_order`) |
| `permute_order` | re-permute conditioning order per draw | re-permute conditioning order per scored copy |
| `aggregate` | collapses the draws of one cell | — |
| `score_pool` | — | pools the per-order likelihood vectors |
| `inference_batch_size` | yes | yes |

#### 7.2.1 Sampling and budget

`temperature` defaults to `0.7`; `<= 0` selects greedy decoding (and, since
retries would be byte-identical, a single generation attempt).
`max_new_tokens=None` resolves at generate time to the fitted
`max_seq_length` — an upper bound that covers any single value, harmless
because the serializer trims the continuation at the value's delimiter.

`inference_batch_size=None` resolves to the training `batch_size`, keeping
inference memory in line with what fine-tuning already handled. It is purely
a throughput/memory lever: **results are batch-size-invariant** — chunking
rows differently cannot change predictions (per-row randomness is seeded on
the row's absolute identity, not its position in a batch).

#### 7.2.2 Ensembling: `n_samples`, `permute_order`, and friends

For the generative estimators, `n_samples` draws per cell are collapsed by
`aggregate` — by default the mean for numeric columns and the mode otherwise.
`aggregate` is a plain callable `(draws, numeric) -> value`, so a custom
reducer (say, the median) is a one-liner.

`permute_order=True` makes the `n_samples` draws also vary the *order* of the
conditioning columns. The model was trained over many orders but any single
prompt fixes one arbitrary order; permuting marginalizes that choice away.
For the classifier this is the only ensembling there is: each candidate is
scored under up to `n_samples` distinct column orders and the per-order
softmax distributions are averaged (or pooled by your own `score_pool`
callable, which receives the raw per-order log-likelihood vectors).

### 7.3 LoRA and quantization

Both are constructor knobs on every estimator, and they compose:

```python
clf = LanguageModelClassifier(
    "Qwen/Qwen2.5-0.5B",
    lora=LoRAConfig(rank=16, alpha=32),   # train adapters, freeze the base
    quantization="4bit",                  # quantize the frozen base weights
    precision="bf16",
)
```

`lora=None` (default) fine-tunes all weights — fine for distilgpt2-sized
models, increasingly impractical as models grow. `LoRAConfig` exposes the
rank $r$ and alpha $\alpha$ (the effective scale is $\alpha/r$, or
$\alpha/\sqrt{r}$ with `rslora`; note $\alpha$ does **not** track $r$ — set
both), dropout, per-module overrides, rsLoRA and DoRA. One portability trap: `target_modules` naming differs by
backend (HF/peft matches a name suffix like `"c_attn"`; MLX matches the
relative path like `"attn.c_attn"`). `None` or `"all-linear"` auto-discover
on both — the portable choice.

`quantization` takes a `"<n>bit"` string or a `QuantizationConfig` (to pick
the library or group size). Widths depend on the backend: MLX does
2/3/4/6/8-bit natively; HF does 4/8-bit via bitsandbytes and 2/3-bit via HQQ.
`precision` sets the compute dtype for unquantized weights and autocast
(`"fp32"` default; `"bf16"` is the usual GPU choice).

### 7.4 Reproducibility

`random_state` seeds the column-permutation stream, the train/validation
split, generation sampling, and the per-row order draws. Like scikit-learn,
`None` means non-deterministic. Two details: per-epoch permutations are
seeded on `(seed, epoch)` so a fit is internally idempotent, and per-row
draws are seeded on `(seed, row_id)` — which is what makes predictions
invariant to `inference_batch_size`.

## 8. scikit-learn integration

### 8.1 Pipelines

The estimators are ordinary scikit-learn objects, so they compose. The
imputer is a transformer; the oversampler slots into imbalanced-learn's
`Pipeline`:

```python
from imblearn.pipeline import Pipeline
from sklm import LanguageModelImputer, LanguageModelOverSampler, LanguageModelClassifier

pipe = Pipeline([
    ("impute", LanguageModelImputer("distilgpt2")),
    ("balance", LanguageModelOverSampler(model="distilgpt2")),
    ("clf", LanguageModelClassifier("distilgpt2")),
])
pipe.fit(X_train, y_train)
```

(Each step fine-tunes its own model — see 8.4 before doing this casually.)

One practical note: scikit-lm estimators *want* raw, readable data. The usual
preprocessing reflex — standard-scaling numerics, one-hot-encoding categories
— actively hurts here, because it replaces meaningful text (`"city": "SP"`)
with opaque numbers. Keep the table human-readable; that is what the LM reads.

### 8.2 Params, cloning, and `__` addressing

The config objects subclass `BaseEstimator`, so scikit-learn recurses into
them:

```python
clf.get_params()["training__epochs"]          # 50
clf.set_params(training__epochs=20, generation__n_samples=8)
clone(clf)                                    # deep, independent copy
```

Each estimator deep-copies its config defaults at construction, so two
estimators never share a mutable config — `set_params` on one cannot leak
into another.

### 8.3 Hyperparameter search

`__` addressing makes the configs searchable directly:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    LanguageModelClassifier("distilgpt2", random_state=42),
    param_grid={
        "training__epochs": [20, 40],
        "training__learning_rate": [2e-5, 5e-5],
        "serializer": ["json", "key-value"],
    },
    cv=3,
)
```

For anything beyond a few combinations, prefer a budget-aware searcher —
[notebook 06](examples/06-optuna-search.ipynb) runs the same idea through
Optuna with pruning.

### 8.4 The cost model: every fit is a fine-tune

This is the one place scikit-lm's cost profile diverges sharply from
classical estimators, so budget for it explicitly. A 3-fold grid search over
4 parameter combinations is 12 fine-tunes plus a final refit. Mitigations:
fewer folds, `max_steps` to cap each fit, LoRA (less compute per step),
early stopping with a small `validation_split`, and per-call decoder
overrides (sections 5.2.2, 5.3) to compare inference settings *without*
refitting. [Notebook 07](examples/07-stratified-cv.ipynb) shows a stratified
CV setup tuned for this reality.

### 8.5 Persistence

Fitted estimators pickle with the standard tools (`pickle`, `joblib`). The
callback is deliberately excluded from the pickle — it is live observability,
possibly holding an open stream — and is restored as a no-op on load.

## 9. Observability: callbacks

Fine-tuning an LM mid-`fit` is opaque without feedback, so every estimator
emits structured events. The `callback` parameter takes a `Callback`
instance, a list (wrapped in a `CompositeCallback`), or `None` — the default,
which auto-selects a dashboard for the runtime: a live widget in Jupyter, a
Rich terminal dashboard in a TTY, plain logging otherwise. You normally
configure nothing and still see loss curves, validation loss, memory,
serialized-row previews, and per-row inference progress.

The shipped implementations: `JupyterCallback`, `RichCallback`,
`TqdmCallback` (progress bars only), `LoggingCallback` (stdlib `logging`,
no handler/level configuration imposed — that stays your choice).

### 9.1 Writing your own

`Callback` is both the wire protocol and a state aggregator: thirteen
granular `on_*` events fold into a running `TrainingState`, and every event
is re-dispatched through a single hook. Subclass and override **only**
`on_event` (the granular methods are `@final` — overriding one would
silently skip the aggregation):

```python
from sklm import Callback, TrainingState, Event, TrainReport, EvalReport

class LossLogger(Callback):
    def on_event(self, state: TrainingState, event: Event) -> None:
        match event:
            case TrainReport(step=s, loss=l):
                print(f"step {s}: train loss {l:.4f}")
            case EvalReport(loss=l):
                print(f"          eval loss {l:.4f}")

clf = LanguageModelClassifier("distilgpt2", callback=LossLogger())
```

`Event` is a union of frozen dataclasses — `FitInfo`, `FitStart`,
`TrainExamples` (the exact serialized rows of an epoch, useful for sanity
checks), `TrainReport`, `EvalReport`, `Memory`, `FitEnd`, `PredictStart`,
`RowEnd`, `Generation`, `Score`, `Retry`, `PredictEnd` — so a `match` on the
event type is the natural shape. `TrainingState` carries the accumulated
snapshot (loss series, current epoch, memory peaks, predict progress), which
is what lets a dashboard render from scratch at any moment.

The `Generation`, `Score`, and `Retry` events are the inference microscope:
they carry each prompt, the raw continuation, the decoded value, and the
per-candidate likelihoods — invaluable when you want to see *why* a model is
retrying or how confident the scoring really is.

## 10. Below the estimators: TabularLanguageModel

The four estimators are adapters over `TabularLanguageModel` — the object
that owns the fitted backend, the serializer, and the conditional
primitives. A fitted estimator exposes it as `estimator.lm_`, and using it
directly unlocks queries no fixed estimator offers
([notebook 05](examples/05-conditional-queries.ipynb)):

```python
clf = LanguageModelClassifier("distilgpt2").fit(X_train, y_train)
lm = clf.lm_

# Ask for ANY column given ANY subset -- not just the label:
lm.predict_proba(
    {"species": "setosa", "petal_width": 0.2},   # condition on these...
    "petal_length",                              # ...ask about this column
    candidates=[1.0, 1.4, 1.8, 4.5],             # over these values
)

# Or generate missing columns outright:
lm.complete(
    {"species": "virginica"},
    targets=["sepal_length", "sepal_width"],
    generation=GenerationConfig(temperature=0.7),
)
```

The full surface:

- `fit(X, y=None, *, target_cols)` — what every estimator calls internally:
  serialize, permute, fine-tune. `y` is ignored (the model is joint; a
  supervised target enters as a column of `X` — the estimators append it
  before calling this).
- `complete(known, targets, generation)` / `complete_many(...)` — generate
  target columns given known ones, sequentially within a row (each target
  conditions on the previous fills), batched across rows. Returns `None` for
  a row whose generation stayed malformed — the *estimators* are what turn
  that into a `RuntimeError`.
- `predict_proba(known, target, candidates)` / `predict_proba_many(...)` —
  rank candidates for one column by likelihood; softmax over mean per-token
  log-likelihoods, with optional order marginalization (section 7.2.2).
- `sample(n_samples, *, condition=None, generation=...)` — draw whole rows
  from the learned joint: every column is generated in training order, each
  conditioning on the cells already produced. A single `condition` mapping is
  broadcast to all rows; a sequence gives one mapping per row (and overrides
  `n_samples`). Returns a DataFrame and raises `RuntimeError` if any row
  stays malformed ([notebook 08](examples/08-synthesizer.ipynb)).

You can also construct a `TabularLanguageModel` directly (it is a public,
documented class — and a scikit-learn `BaseEstimator` in its own right:
clonable, tunable through nested params such as `training__epochs`) when you
want the mechanism without any estimator framing — e.g. `fit` + `sample` as a
pure table synthesizer.

## 11. Practical guidance

### 11.1 Model, epochs, batch size

- **Start small.** `distilgpt2` + 30–50 epochs is the baseline that the
  defaults assume. Move up (gpt2, Qwen 0.5B, ...) when the table has many
  columns, long text cells, or subtle conditionals — and add LoRA + bf16 at
  that point.
- **`batch_size` is the main speed lever**, for fitting and (via the
  `inference_batch_size` default) for inference. Raise it until memory says
  stop; results don't depend on it.
- **Watch validation, not vibes.** `validation_split=0.1` plus
  `early_stopping_patience` turns "how many epochs?" into a measured answer,
  and the auto-dashboard plots both losses live.

### 11.2 Prefer scoring for numeric targets

The generative path is the flexible default, but on numeric targets the
scoring path (discretization, sections 5.2.2 / 5.3) is usually the stabler
decoder: candidates are real observed values, nothing can be malformed, and
`estimate` gives you an explicit mean-vs-mode choice. A practical workflow is
to fit once and compare decoders with per-call overrides — no refit needed.
Match `estimate` to what the metric rewards: `"mean"` for RMSE-like losses,
`"mode"` when only exact hits count.

### 11.3 Memory checklist

In rough order of impact when a fit doesn't... fit:

1. `lora=LoRAConfig(...)` — gradients/optimizer state shrink to the adapters.
2. `quantization="4bit"` — the frozen base weights shrink.
3. `precision="bf16"` — halves activation memory on GPU.
4. `training.gradient_checkpointing=True` — trades compute for activations.
5. Lower `training.batch_size`, compensate with `grad_accumulation_steps`.

### 11.4 Things that surprise people

- **Every `fit` reloads the base model and fine-tunes from scratch.** There
  is no warm-start; that is what keeps `clone`/CV semantics honest.
- **The classifier cannot be made stochastic.** Temperature has no path into
  scoring (section 2.3). If `predict_proba` looks overconfident, ensemble
  over orders (`permute_order` + `n_samples`) instead.
- **`RuntimeError` at predict time is a verdict on training**, not a bug to
  catch: the model could not produce a parseable value in 15 attempts. More
  epochs, a bigger model, or the JSON serializer (most robust to weird
  values) are the actual fixes.
- **Array input renames your columns.** Without a DataFrame, columns become
  `x0, x1, ...` — the model loses the semantic hint that real names provide.
  Prefer DataFrames with meaningful column names and readable category
  values.
- **Duplicate column names are rejected** at `fit` (serialization would be
  last-wins), and the regressor requires a finite `y`.

---

## Appendix A. Public API index

Everything importable from `sklm` (the contents of `__all__`):

**Estimators** — `LanguageModelClassifier`, `LanguageModelRegressor`,
`LanguageModelImputer`, `LanguageModelOverSampler`.

**Core** — `TabularLanguageModel` (the shared fitted model; a scikit-learn
estimator with `fit` / `sample`); `LanguageModelBackend` (the backend
protocol); `HFBackend`, `MLXBackend` (the shipped engines).

**Configs** — `TrainingConfig`, `GenerationConfig`, `LoRAConfig`,
`QuantizationConfig`, `DiscretizationConfig`, `ModelConfig`;
`aggregate_default` (the default draw reducer); the `Literal` aliases
`Quantization`, `Precision`, `Optimizer`, `LRScheduler`.

**Serialization** — `Serializer` (protocol), `JSONSerializer`,
`KeyValueSerializer`, `BracketSerializer`; `NumberFormat` (protocol),
`PlainNumber`, `SpacedDigits`; `Field`, `TrainingExample`.

**Callbacks** — `Callback`, `CompositeCallback`, `LoggingCallback`,
`TqdmCallback`, `RichCallback`, `JupyterCallback`; `TrainingState` and the
`Event` union: `FitInfo`, `FitStart`, `TrainExamples`, `TrainReport`,
`EvalReport`, `Memory`, `FitEnd`, `PredictStart`, `RowEnd`, `Generation`,
`Score`, `Retry`, `PredictEnd`.

**Constructor typing** — `EstimatorArgs`, `RegressorArgs`, `ImputerArgs`,
`OversamplerArgs`, `AnnotatedDefault` (the TypedDicts behind the shared
keyword surface; useful for typed wrappers around the estimators).

## Appendix B. HF vs. MLX cheat sheet

| | `HFBackend` | `MLXBackend` |
|---|---|---|
| Stack | torch + transformers | mlx + mlx-lm |
| Hardware | CUDA, Apple MPS, CPU | Metal (macOS), CUDA/CPU (Linux) |
| Training loop | `transformers.Trainer`; per-epoch re-permutation via a `TrainerCallback` | `mlx_lm.tuner` trainer with a custom batch iterator |
| Quantization | 4/8-bit (bitsandbytes), 2/3-bit (HQQ), at load time | 2/3/4/6/8-bit natively, at convert time (cached under `~/.cache/sklm/mlx`) |
| LoRA `target_modules` | name suffix (`"c_attn"`) | relative module path (`"attn.c_attn"`); `None`/`"all-linear"` portable on both |
| `attn_implementation` | passed to `from_pretrained` | ignored |
| 8-bit optimizers | CUDA only (fallback: AdamW) | n/a (e.g. `lion` runs full-precision) |
| `distilgpt2` | works as-is | use an MLX-loadable mirror (`gabfssilva/distilgpt2`) |
| `score()` semantics | identical on both: mean per-token log-likelihood, prompt/continuation boundary at the longest common token prefix | same (kept behaviorally aligned by the integration tests) |
