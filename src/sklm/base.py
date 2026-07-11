"""Shared construction and input-handling helpers for the estimators."""

from __future__ import annotations

import importlib.util
import platform
from collections.abc import Collection, Mapping, Sequence
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from .backend import LanguageModelBackend
from .bridge import Model, Tokenizer
from .callbacks import Callback, resolve_callback
from .config import (
    DiscretizationConfig,
    LoRAConfig,
    ModelConfig,
    Precision,
    Quantization,
    QuantizationConfig,
    TrainingConfig,
)
from .core import TabularLanguageModel, forget, to_frame
from .hf_backend import HFBackend
from .serialize import (
    BracketSerializer,
    IfThenSerializer,
    JSONSerializer,
    KeyValueSerializer,
    PlainNumber,
    Serializer,
    SpacedDigits,
)

__all__ = [
    "align_features",
    "forget",
    "make_tabular_lm",
    "records",
    "reduce_estimate",
    "resolve_discretization",
    "resolve_quantization",
    "select_candidates",
    "to_frame",
    "unique_name",
]


class _HasBackendParams(Protocol):
    model: Model
    backend: LanguageModelBackend | str
    training: TrainingConfig
    serializer: str | Serializer
    max_decimals: int | None
    number_format: Literal["plain", "spaced"]
    random_state: int | None
    callback: Callback | list[Callback] | Literal["auto"] | None
    lora: LoRAConfig | None
    quantization: Quantization | QuantizationConfig | None
    precision: Precision
    tokenizer: Tokenizer | None
    trust_remote_code: bool
    device: str
    attn_implementation: str | None


class _Fitted(Protocol):
    n_features_in_: int
    feature_cols_: list[str]


def resolve_serializer(
    serializer: str | Serializer,
    max_decimals: int | None = None,
    number_format: Literal["plain", "spaced"] = "plain",
) -> Serializer:
    """Resolve a serializer selector or instance to a ``Serializer``.

    Parameters
    ----------
    serializer : str or Serializer
        One of ``"json"``, ``"key-value"``, ``"bracket"``, ``"if-then"`` or a
        ``Serializer`` instance. For custom delimiters, pass an instance.
    max_decimals : int or None, optional
        Decimal places to round numeric cells to when building from a string
        selector. Ignored when ``serializer`` is already an instance -- that
        carries its own :class:`~sklm.NumberFormat`. Default ``None`` (no
        rounding).
    number_format : {"plain", "spaced"}, optional
        Number rendering when building from a string selector: ``"plain"``
        (default) builds :class:`~sklm.PlainNumber`, ``"spaced"`` builds
        :class:`~sklm.SpacedDigits` (one token per character). Ignored when
        ``serializer`` is already an instance.

    Returns
    -------
    Serializer
        The resolved serializer.

    Raises
    ------
    ValueError
        If ``serializer`` or ``number_format`` is an unknown string selector.
    """
    if isinstance(serializer, str):
        builders = {
            "json": JSONSerializer,
            "key-value": KeyValueSerializer,
            "bracket": BracketSerializer,
            "if-then": IfThenSerializer,
        }
        builder = builders.get(serializer)
        if builder is None:
            raise ValueError(
                f"unknown serializer {serializer!r}; pass one of {sorted(builders)} "
                "or a Serializer instance"
            )
        numbers = {"plain": PlainNumber, "spaced": SpacedDigits}
        number = numbers.get(number_format)
        if number is None:
            raise ValueError(
                f"unknown number_format {number_format!r}; pass one of {sorted(numbers)}"
            )
        return builder(number=number(max_decimals=max_decimals))
    return serializer


def resolve_quantization(
    quantization: Quantization | QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Resolve a quantization selector or instance to a ``QuantizationConfig``.

    Parameters
    ----------
    quantization : str, QuantizationConfig or None
        A ``"<n>bit"`` string -- shorthand for ``QuantizationConfig(bits=n)`` with
        the default ``method="auto"`` -- a ``QuantizationConfig`` (returned
        unchanged), or ``None`` for no quantization.

    Returns
    -------
    QuantizationConfig or None
        The resolved configuration, or ``None``.
    """
    if isinstance(quantization, str):
        return QuantizationConfig(bits=int(quantization.removesuffix("bit")))
    return quantization


def _has_hf() -> bool:
    """Whether the Hugging Face backend's deps (torch + transformers) are importable."""
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


def _has_mlx() -> bool:
    """Whether the MLX backend's deps (mlx + mlx-lm) are importable."""
    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


def _hf_gpu_available() -> bool:
    """Whether the Hugging Face backend has an accelerator (CUDA or Apple MPS).

    Importing torch is the only way to read device availability; it happens once
    while resolving ``"auto"`` and torch is loaded by the backend at fit anyway.
    """
    if not _has_hf():
        return False
    import torch

    return bool(torch.cuda.is_available() or torch.backends.mps.is_available())


def _mlx_gpu_available() -> bool:
    """Whether MLX is installed with a usable GPU backend (Metal or CUDA).

    Which GPU variant is present is the user's install-time choice (the ``mlx`` /
    ``mlx-cpu`` / ``mlx-cuda12`` / ``mlx-cuda13`` extras), so ``"auto"`` never
    distinguishes Metal from CUDA. ``mlx.core.default_device()`` is ``gpu`` on a
    Metal (Apple Silicon) or CUDA (Linux/NVIDIA) build and ``cpu`` on a CPU-only
    build, so one check covers both accelerators. ``False`` when mlx/mlx-lm are
    absent.
    """
    if not _has_mlx():
        return False
    import mlx.core as mx

    return mx.default_device() == mx.gpu


def _mlx_backend() -> LanguageModelBackend:
    """Construct the MLX backend, imported lazily so mlx stays an optional extra."""
    from .mlx_backend import MLXBackend

    return MLXBackend()


def _resolve_auto() -> LanguageModelBackend:
    """Pick a backend from what is installed, by platform-aware preference.

    macOS prefers MLX (Metal). Anywhere else the order is HF-GPU, MLX-GPU,
    HF-CPU, MLX-CPU -- accelerated backends first, and HF ahead of MLX within a
    tier -- falling back to the Hugging Face backend (whose fit raises a guided
    ImportError) when neither stack is installed.
    """
    if platform.system() == "Darwin":
        return _mlx_backend() if _has_mlx() else HFBackend()
    for available, build in (
        (_hf_gpu_available, HFBackend),
        (_mlx_gpu_available, _mlx_backend),
        (_has_hf, HFBackend),
        (_has_mlx, _mlx_backend),
    ):
        if available():
            return build()
    return HFBackend()


def resolve_backend(backend: LanguageModelBackend | str) -> LanguageModelBackend:
    """Resolve a backend selector or instance to a ``LanguageModelBackend``.

    Parameters
    ----------
    backend : LanguageModelBackend or str
        One of ``"huggingface"``, ``"mlx"``, ``"auto"``, or a
        ``LanguageModelBackend`` instance. ``"auto"`` resolves to whichever stack
        is installed, by platform-aware preference: MLX on macOS (Metal);
        otherwise HF-GPU, MLX-GPU, HF-CPU, MLX-CPU in that order
        (:func:`_resolve_auto`).

    Returns
    -------
    LanguageModelBackend
        The resolved backend.

    Raises
    ------
    ValueError
        If ``backend`` is an unknown string selector.
    """
    if isinstance(backend, str):
        if backend == "huggingface":
            return HFBackend()
        if backend == "mlx":
            return _mlx_backend()
        if backend == "auto":
            return _resolve_auto()
        raise ValueError(
            f"unknown backend {backend!r}; pass 'huggingface', 'mlx', 'auto' "
            "or a LanguageModelBackend instance"
        )
    return backend


def make_tabular_lm(estimator: _HasBackendParams) -> TabularLanguageModel:
    """Build the shared ``TabularLanguageModel`` from an estimator's configuration.

    The backend is resolved fresh from a string selector (so refitting reloads
    the base model) or taken as the injected instance; fine-tuning
    hyperparameters travel in ``estimator.training`` and the flat model-loading
    fields are reassembled into a :class:`ModelConfig` here.
    """
    return TabularLanguageModel(
        backend=resolve_backend(estimator.backend),
        serializer=resolve_serializer(
            estimator.serializer, estimator.max_decimals, estimator.number_format
        ),
        training=estimator.training,
        model=ModelConfig(
            model=estimator.model,
            lora=estimator.lora,
            quantization=resolve_quantization(estimator.quantization),
            precision=estimator.precision,
            tokenizer=estimator.tokenizer,
            trust_remote_code=estimator.trust_remote_code,
            device=estimator.device,
            attn_implementation=estimator.attn_implementation,
        ),
        random_state=estimator.random_state,
        callback=resolve_callback(estimator.callback),
    )


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return ``frame`` as a list of column-name -> value row mappings."""
    return frame.to_dict("records")


def align_features(estimator: _Fitted, X: object) -> pd.DataFrame:
    """Return ``X`` with columns set to ``estimator.feature_cols_``.

    Validates the feature count, reorders a DataFrame's columns to the training
    order by name when names were recorded at fit (raising if any are missing),
    and renames the columns to the stringified names used for serialization.

    Parameters
    ----------
    estimator : _Fitted
        Fitted estimator exposing ``n_features_in_`` and ``feature_cols_``.
    X : array-like or pandas.DataFrame
        Input to align.

    Returns
    -------
    pandas.DataFrame
        Frame whose columns are exactly ``estimator.feature_cols_``.
    """
    names = getattr(estimator, "feature_names_in_", None)
    X_df = to_frame(X, names)
    if X_df.shape[1] != estimator.n_features_in_:
        raise ValueError(
            f"X has {X_df.shape[1]} features, but {type(estimator).__name__} "
            f"is expecting {estimator.n_features_in_} features as input."
        )
    if isinstance(X, pd.DataFrame) and names is not None:
        absent = [c for c in names if c not in X_df.columns]
        if absent:
            raise ValueError(f"X is missing features seen during fit: {absent}")
        X_df = X_df.reindex(columns=list(names))
    return X_df.set_axis(estimator.feature_cols_, axis=1)


def unique_name(base: str, taken: Sequence[str]) -> str:
    """Return ``base``, appending ``_`` as needed, so the result is absent from ``taken``."""
    name = base
    taken_set = set(taken)
    while name in taken_set:
        name += "_"
    return name


def _representative(group: np.ndarray, kind: Literal["median", "mode", "mean"]) -> float:
    """One value standing in for a partition of observed target values.

    ``"median"`` is the lower median and ``"mode"`` the most frequent value
    (ties broken by the smaller value) -- both real observed values; ``"mean"``
    is the synthetic arithmetic mean.
    """
    if kind == "mean":
        return float(np.mean(group))
    if kind == "mode":
        values, counts = np.unique(group, return_counts=True)
        return float(values[counts == counts.max()].min())
    return float(np.percentile(group, 50, method="lower"))


def select_candidates(values: np.ndarray, config: DiscretizationConfig) -> list[float]:
    """Candidate values to score, drawn from a target column's observed values.

    Non-NaN values are partitioned into :meth:`DiscretizationConfig.resolve_k`
    strata (equal-mass quantiles or equal-width bins, per ``config.strategy``)
    and one representative is taken from each (``config.representative``). When
    the requested count reaches the number of distinct values the full sorted
    support is returned and partitioning is skipped.

    Parameters
    ----------
    values : numpy.ndarray
        Observed values of the target column (may contain NaN, dropped here).
    config : DiscretizationConfig
        The active discretization settings; assumed enabled (``bins`` truthy).

    Returns
    -------
    list of float
        Sorted, de-duplicated candidate values.

    Raises
    ------
    ValueError
        If no observed (non-NaN) values remain.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        raise ValueError("discretization target has no observed (non-NaN) values")
    unique = np.unique(v)
    k = config.resolve_k(unique.size)
    if k >= unique.size:
        return [float(x) for x in unique]
    if config.strategy == "quantile":
        labels = np.asarray(pd.qcut(v, q=k, duplicates="drop", labels=False))
    else:
        edges = np.linspace(v.min(), v.max(), k + 1)
        labels = np.clip(np.digitize(v, edges[1:-1]), 0, k - 1)
    reps = {_representative(v[labels == lbl], config.representative) for lbl in np.unique(labels)}
    return sorted(reps)


def reduce_estimate(
    proba_row: np.ndarray, candidates: Sequence[object], config: DiscretizationConfig
) -> float:
    """Collapse a candidate probability distribution into one prediction.

    ``config.estimate == "mean"`` returns the probability-weighted expectation
    over ``candidates``; ``"median"`` the probability-weighted median (the
    smallest candidate whose cumulative probability reaches half); ``"mode"``
    the single most likely candidate. ``config.sharpness`` re-tempers the
    distribution first (probabilities are raised to that power and
    renormalized), pulling ``"mean"`` and ``"median"`` toward the top
    candidates; the argmax (``"mode"``) is invariant to it.

    Parameters
    ----------
    proba_row : numpy.ndarray
        Probabilities over ``candidates`` (sums to 1), as returned by
        :meth:`~sklm.TabularLanguageModel.predict_proba_many`.
    candidates : Sequence of float
        The scored candidate values, aligned with ``proba_row``.
    config : DiscretizationConfig
        The active discretization settings.

    Returns
    -------
    float
        The reduced prediction.
    """
    c = np.asarray(candidates, dtype=float)
    if config.estimate == "mode":
        return float(c[int(np.argmax(proba_row))])
    p = np.asarray(proba_row, dtype=float)
    if config.sharpness != 1.0:
        p = p**config.sharpness
        p = p / p.sum()
    if config.estimate == "median":
        order = np.argsort(c)
        cum = np.cumsum(p[order])
        idx = int(np.searchsorted(cum, 0.5 * cum[-1]))
        return float(c[order[min(idx, c.size - 1)]])
    return float(np.dot(p, c))


def resolve_discretization(
    discretization: DiscretizationConfig | Mapping[str, DiscretizationConfig],
    numeric_cols: Collection[str],
) -> dict[str, DiscretizationConfig]:
    """Map each numeric column that should be *scored* to its effective config.

    Collapses the imputer's union parameter: a single :class:`DiscretizationConfig`
    applies to every numeric column, while a mapping is per-column (columns absent
    from it are generated, not scored). Only columns with a truthy ``bins`` are
    kept, so an empty result means "nothing to score" -- the caller falls back to
    the generative path.

    Parameters
    ----------
    discretization : DiscretizationConfig or Mapping[str, DiscretizationConfig]
        The imputer's ``discretization`` parameter.
    numeric_cols : Collection of str
        Names of the numeric columns (only these can be scored).

    Returns
    -------
    dict[str, DiscretizationConfig]
        Numeric columns to score, each mapped to its active config.
    """
    if isinstance(discretization, DiscretizationConfig):
        return {c: discretization for c in numeric_cols} if discretization.bins else {}
    return {c: cfg for c, cfg in discretization.items() if cfg.bins and c in numeric_cols}


def resolve_categorical(
    discretization: DiscretizationConfig | Mapping[str, DiscretizationConfig],
    categorical_cols: Collection[str],
) -> frozenset[str]:
    """Categorical columns to *score* over their observed levels.

    A categorical column is a finite discrete space, so it is scored by default
    (the dual of ``resolve_discretization``, where numeric columns are generated
    unless a non-zero ``bins`` opts them in). A single
    :class:`~sklm.DiscretizationConfig` leaves every categorical scored; a mapping
    can switch one off with a ``bins == 0`` entry, routing that column back to
    free-text generation.

    Parameters
    ----------
    discretization : DiscretizationConfig or Mapping[str, DiscretizationConfig]
        The estimator's ``discretization`` parameter.
    categorical_cols : Collection of str
        Names of the categorical columns (their levels are the candidate set).

    Returns
    -------
    frozenset[str]
        Categorical columns to score; the rest generate.
    """
    if isinstance(discretization, DiscretizationConfig):
        return frozenset(categorical_cols)
    disabled = {c for c, cfg in discretization.items() if c in categorical_cols and not cfg.bins}
    return frozenset(categorical_cols) - disabled
