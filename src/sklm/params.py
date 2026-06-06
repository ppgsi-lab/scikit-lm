"""Flattened estimator constructor parameters.

Every estimator shares the same model-loading and fine-tuning knobs. Instead of
repeating them in each ``__init__``, they are declared once here as a
:class:`~typing.TypedDict` and unpacked into the constructors via
:data:`~typing.Unpack` (PEP 692), so call sites keep full keyword autocomplete
and type-checking. Per-field defaults travel with the type as
:class:`AnnotatedDefault` metadata; :meth:`AnnotatedDefault.create_with_defaults`
materializes them at construction time.

Because ``Unpack`` routes the keys through ``**kwargs``, scikit-learn's
``_get_param_names`` (which skips ``VAR_KEYWORD``) cannot see them, which would
break ``clone``/``set_params``/``GridSearchCV``. The :class:`_FlatParams` mixin
restores the estimator contract by reporting the explicit signature parameters
together with the TypedDict keys.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from copy import deepcopy
from typing import Annotated, ClassVar, TypedDict, get_args, get_type_hints

from .backend import LanguageModelBackend
from .callbacks import Callback
from .config import (
    DiscretizationConfig,
    GenerationConfig,
    LoRAConfig,
    Precision,
    Quantization,
    QuantizationConfig,
    TrainingConfig,
)
from .serialize import Serializer

__all__ = [
    "AnnotatedDefault",
    "EstimatorArgs",
    "ImputerArgs",
    "OversamplerArgs",
    "RegressorArgs",
]


class AnnotatedDefault:
    """Carries a field's default value as ``Annotated`` metadata."""

    def __init__(self, default: object) -> None:
        self.default = default

    @classmethod
    def create_with_defaults(
        cls, typed_dict: type[EstimatorArgs], **overrides: object
    ) -> dict[str, object]:
        """Return the TypedDict's ``AnnotatedDefault`` values, with ``overrides`` applied.

        Raises :class:`TypeError` when ``overrides`` carries a key absent from
        ``typed_dict`` -- a typo'd or unsupported constructor argument -- rather
        than silently setting a dead attribute.

        ``get_type_hints`` (not ``__annotations__``) is required because the
        module uses ``from __future__ import annotations``, which leaves the raw
        annotations as unresolved strings stripped of their metadata.

        Defaults are deep-copied so a mutable default (e.g. a ``TrainingConfig``)
        is never shared across estimators -- otherwise ``set_params`` on one
        estimator's nested config would leak into every other estimator built
        from the same default.
        """
        hints = get_type_hints(typed_dict, include_extras=True)
        unknown = sorted(set(overrides) - set(hints))
        if unknown:
            raise TypeError(
                f"unexpected keyword argument(s): {', '.join(unknown)}. "
                f"Valid parameters: {', '.join(sorted(hints))}."
            )
        defaults = {
            key: deepcopy(meta.default)
            for key, hint in hints.items()
            for meta in get_args(hint)[1:]
            if isinstance(meta, cls)
        }
        return defaults | overrides


class EstimatorArgs(TypedDict, total=False):
    """Constructor knobs shared by every estimator (see each estimator's docstring)."""

    backend: Annotated[LanguageModelBackend | str, AnnotatedDefault("huggingface")]
    training: Annotated[TrainingConfig, AnnotatedDefault(TrainingConfig())]
    generation: Annotated[GenerationConfig, AnnotatedDefault(GenerationConfig())]
    serializer: Annotated[str | Serializer, AnnotatedDefault("json")]
    max_decimals: Annotated[int | None, AnnotatedDefault(3)]
    random_state: Annotated[int | None, AnnotatedDefault(None)]
    callback: Annotated[Callback | list[Callback] | None, AnnotatedDefault(None)]
    lora: Annotated[LoRAConfig | None, AnnotatedDefault(None)]
    quantization: Annotated[Quantization | QuantizationConfig | None, AnnotatedDefault(None)]
    precision: Annotated[Precision, AnnotatedDefault("fp32")]
    tokenizer: Annotated[str | None, AnnotatedDefault(None)]
    trust_remote_code: Annotated[bool, AnnotatedDefault(False)]
    device: Annotated[str, AnnotatedDefault("auto")]
    attn_implementation: Annotated[str | None, AnnotatedDefault(None)]


class RegressorArgs(EstimatorArgs, total=False):
    discretization: Annotated[DiscretizationConfig, AnnotatedDefault(DiscretizationConfig())]


class ImputerArgs(EstimatorArgs, total=False):
    discretization: Annotated[
        DiscretizationConfig | Mapping[str, DiscretizationConfig],
        AnnotatedDefault(DiscretizationConfig()),
    ]


class OversamplerArgs(EstimatorArgs, total=False):
    pass


class _FlatParams:
    """Expose ``Unpack``-routed kwargs to scikit-learn's parameter introspection.

    Subclasses set :attr:`_args` to their argument TypedDict and declare the flat
    fields as attribute annotations (so they participate in protocol matching).
    """

    _args: ClassVar[type[EstimatorArgs]]

    backend: LanguageModelBackend | str
    training: TrainingConfig
    generation: GenerationConfig
    serializer: str | Serializer
    max_decimals: int | None
    random_state: int | None
    callback: Callback | list[Callback] | None
    lora: LoRAConfig | None
    quantization: Quantization | QuantizationConfig | None
    precision: Precision
    tokenizer: str | None
    trust_remote_code: bool
    device: str
    attn_implementation: str | None

    @classmethod
    def _get_param_names(cls) -> list[str]:
        signature = inspect.signature(cls.__init__)
        explicit = {
            p.name
            for p in signature.parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        }
        keys = cls._args.__required_keys__ | cls._args.__optional_keys__
        return sorted(explicit | set(keys))
