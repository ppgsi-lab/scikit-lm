"""The language-model backend protocol.

A backend is a pure execution engine: it receives a :class:`~sklm.ModelConfig`
(carrying the model id) and a :class:`~sklm.TrainingConfig` at fit time and a
:class:`~sklm.GenerationConfig` at generate time, and stores no user-facing
hyperparameters of its own. ``LanguageModelBackend`` is the only abstraction the
rest of the library depends on, so torch/transformers (:class:`~sklm.HFBackend`)
and mlx (:class:`~sklm.MLXBackend`) stay optional extras, and tests inject a
lightweight fake backend instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from .callbacks import Callback
from .config import GenerationConfig, ModelConfig, TrainingConfig
from .serialize import TrainingExample

__all__ = ["LanguageModelBackend", "resolve_max_new_tokens", "resolve_max_seq_length"]


def resolve_max_new_tokens(generation: GenerationConfig, max_seq_length: int) -> int:
    """Generation token budget, defaulting to the longest training row.

    ``GenerationConfig.max_new_tokens`` is ``None`` by default because the config
    is constructed before any data is seen; at generate time each backend
    substitutes its fitted ``max_seq_length`` (the longest serialized row), an
    upper bound that covers any single value the model could emit without manual
    tuning. The generated tokens are trimmed to the target value's delimiter by
    the serializer afterwards, so an over-budget is harmless.
    """
    if generation.max_new_tokens is not None:
        return generation.max_new_tokens
    return max_seq_length


def resolve_max_seq_length(
    examples: Sequence[TrainingExample],
    token_len: Callable[[str], int],
    *,
    multiple: int = 8,
) -> int:
    """Smallest ``multiple`` of tokens holding the longest serialized row.

    Fills ``TrainingConfig.max_seq_length`` when left ``None``: each backend
    passes a ``token_len`` matching how it tokenizes a row (including EOS), and
    the cap is the max over ``examples`` rounded up so no row is truncated.
    Column-order permutation reorders a row's tokens without changing their
    count, so measuring one serialization per row is representative.
    """
    longest = max((token_len(ex.text) for ex in examples), default=multiple)
    return ((longest + multiple - 1) // multiple) * multiple


@runtime_checkable
class LanguageModelBackend(Protocol):
    """Execution engine that fine-tunes, generates, and scores text.

    The only abstraction the rest of the library depends on: estimators hand it
    the configs (the model id rides on :class:`~sklm.ModelConfig`) and consume
    its three primitives, so torch and transformers stay an optional extra.

    Notes
    -----
    Shipped implementations are :class:`~sklm.HFBackend` (transformers) and
    :class:`~sklm.MLXBackend` (mlx-lm).
    """

    def fit(
        self,
        epoch_texts: Callable[[], list[TrainingExample]],
        training: TrainingConfig,
        model_config: ModelConfig,
        *,
        random_state: int | None,
        callbacks: Callback,
        eval_examples: list[TrainingExample] | None = None,
    ) -> None:
        """Fine-tune ``model_config.model`` on serialized rows. ``epoch_texts()`` returns one
        freshly (re-)permuted :class:`~sklm.TrainingExample` per training row and
        is called once per epoch, so feature-order permutation stays dynamic
        across epochs. Each example's ``prompt`` (empty unless
        ``loss_on_target_only`` masks that row) is excluded from the loss.
        ``callbacks`` receives training-loss reports.

        ``eval_examples`` is the held-out validation set (``None`` when
        ``training.validation_split`` is ``0``): a fixed list serialized once, so
        validation loss is comparable across epochs. When provided, the backend
        evaluates on it and reports the loss through
        :meth:`~sklm.Callback.on_eval_report`, drives ``early_stopping_patience``
        off it, and checkpoints per ``checkpoint_steps`` / ``checkpoint_dir``."""
        ...

    def generate(self, prompts: Sequence[str], generation: GenerationConfig) -> list[str]:
        """Sample a continuation for each prompt (the generated text only)."""
        ...

    def score(self, prompts: Sequence[str], continuations: Sequence[str]) -> list[float]:
        """Mean per-token log-likelihood of ``continuations[i]`` given ``prompts[i]``.

        The two sequences are paired element-wise (equal length) and scored as a
        single batch, so callers chunk by ``inference_batch_size`` and score many
        prompts at once."""
        ...
