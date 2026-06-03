"""Shared construction and input-handling helpers for the estimators."""

from __future__ import annotations

import contextlib
import importlib.util
import platform
from collections.abc import Collection, Mapping, Sequence
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from .backend import LanguageModelBackend
from .callbacks import Callback, resolve_callbacks
from .config import (
    DiscretizationConfig,
    LoRAConfig,
    ModelConfig,
    Precision,
    Quantization,
    QuantizationConfig,
    TrainingConfig,
)
from .core import TabularLanguageModel
from .hf_backend import HFBackend
from .serialize import (
    BracketSerializer,
    JSONSerializer,
    KeyValueSerializer,
    PlainNumber,
    Serializer,
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
    model: str
    backend: LanguageModelBackend | str
    training: TrainingConfig
    serializer: str | Serializer
    max_decimals: int | None
    random_state: int | None
    callbacks: Callback | None
    lora: LoRAConfig | None
    quantization: Quantization | QuantizationConfig | None
    precision: Precision
    tokenizer: str | None
    trust_remote_code: bool
    device: str
    attn_implementation: str | None


class _Fitted(Protocol):
    n_features_in_: int
    feature_cols_: list[str]


def resolve_serializer(serializer: str | Serializer, max_decimals: int | None = None) -> Serializer:
    """Resolve a serializer selector or instance to a ``Serializer``.

    Parameters
    ----------
    serializer : str or Serializer
        One of ``"json"``, ``"key-value"``, ``"bracket"`` (all with plain
        numbers) or a ``Serializer`` instance. For spaced digits or custom
        delimiters, pass an instance.
    max_decimals : int or None, optional
        Decimal places to round numeric cells to when building from a string
        selector. Ignored when ``serializer`` is already an instance -- that
        carries its own :class:`~sklm.NumberFormat`. Default ``None`` (no
        rounding).

    Returns
    -------
    Serializer
        The resolved serializer.

    Raises
    ------
    ValueError
        If ``serializer`` is an unknown string selector.
    """
    if isinstance(serializer, str):
        builders = {
            "json": JSONSerializer,
            "key-value": KeyValueSerializer,
            "bracket": BracketSerializer,
        }
        builder = builders.get(serializer)
        if builder is None:
            raise ValueError(
                f"unknown serializer {serializer!r}; pass one of {sorted(builders)} "
                "or a Serializer instance"
            )
        return builder(number=PlainNumber(max_decimals=max_decimals))
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
        serializer=resolve_serializer(estimator.serializer, estimator.max_decimals),
        training=estimator.training,
        model_config=ModelConfig(
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
        callbacks=resolve_callbacks(estimator.callbacks),
    )


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return ``frame`` as a list of column-name -> value row mappings."""
    return frame.to_dict("records")


def forget(obj: object, name: str) -> None:
    """Delete attribute ``name`` from ``obj`` if present (no-op otherwise)."""
    with contextlib.suppress(AttributeError):
        delattr(obj, name)


def to_frame(X: object, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Coerce array-like input to a DataFrame, naming columns ``x0..`` (or with
    ``columns``) when the input is not already a DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    names = list(columns) if columns is not None else [f"x{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=names)


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
        raise ValueError(f"X has {X_df.shape[1]} features, expected {estimator.n_features_in_}")
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
    proba_row: np.ndarray, candidates: Sequence[float], config: DiscretizationConfig
) -> float:
    """Collapse a candidate probability distribution into one prediction.

    ``config.estimate == "mean"`` returns the probability-weighted expectation
    over ``candidates``; ``"mode"`` returns the single most likely candidate.

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
    if config.estimate == "mean":
        return float(np.dot(np.asarray(proba_row, dtype=float), c))
    return float(c[int(np.argmax(proba_row))])


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
