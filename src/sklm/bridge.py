"""Cross-framework model/tokenizer type aliases.

``Model`` / ``Tokenizer`` admit a model-id string, an already-loaded HF or MLX
object, or a zero-argument factory producing one of those (deferring the load to
fit time, which keeps refits clean). The framework classes are imported only
under ``TYPE_CHECKING`` and the aliases evaluate lazily (PEP 695), so importing
this module needs none of torch/transformers/mlx. Each backend's ``_load``
resolves these against its own framework's types.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx.nn import Module as MLXModule
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

type Model = PreTrainedModel | MLXModule | str | Callable[[], Model]
type Tokenizer = PreTrainedTokenizerBase | TokenizerWrapper | str | Callable[[], Tokenizer]
