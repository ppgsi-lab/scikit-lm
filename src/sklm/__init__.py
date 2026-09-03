"""scikit-lm: scikit-learn estimators backed by language models.

All estimators share one mechanism: a serialized tabular row is fed to an
autoregressive language model fine-tuned with random feature-order permutation,
so the model learns ``p(any column | any subset of the others)``. Each estimator
is just a choice of which columns go in the prompt and which the model produces.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import logging

try:
    from ._version import __version__ as __version__
except ImportError:  # _version.py is generated at build time by hatch-vcs
    __version__: str = "0.0.0+unknown"

from .backend import LanguageModelBackend
from .callbacks import (
    Callback,
    CompositeCallback,
    EvalReport,
    Event,
    FitEnd,
    FitInfo,
    FitStart,
    Generation,
    JupyterCallback,
    LoggingCallback,
    Memory,
    PredictEnd,
    PredictStart,
    Retry,
    RichCallback,
    RowEnd,
    Score,
    TqdmCallback,
    TrainExamples,
    TrainingState,
    TrainReport,
)
from .classifier import LanguageModelClassifier
from .config import (
    CheckpointConfig,
    DiscretizationConfig,
    EvalConfig,
    GenerationConfig,
    LoRAConfig,
    LRScheduler,
    ModelConfig,
    Optimizer,
    Precision,
    Quantization,
    QuantizationConfig,
    TrainingConfig,
    aggregate_default,
)
from .core import TabularLanguageModel
from .hf_backend import HFBackend
from .imputer import LanguageModelImputer
from .mlx_backend import MLXBackend
from .oversampler import LanguageModelOverSampler
from .params import AnnotatedDefault, EstimatorArgs, ImputerArgs, OversamplerArgs, RegressorArgs
from .regressor import LanguageModelRegressor
from .serialize import (
    BracketSerializer,
    Field,
    IfThenSerializer,
    JSONSerializer,
    KeyValueSerializer,
    NumberFormat,
    NumericSpan,
    PlainNumber,
    Serializer,
    SpacedDigits,
    TrainingExample,
    ValueConstraint,
)
from .synthesizer import LanguageModelSynthesizer

__all__ = [
    "AnnotatedDefault",
    "__version__",
    "BracketSerializer",
    "Callback",
    "CheckpointConfig",
    "CompositeCallback",
    "DiscretizationConfig",
    "EstimatorArgs",
    "EvalConfig",
    "EvalReport",
    "Event",
    "Field",
    "FitEnd",
    "FitInfo",
    "FitStart",
    "GenerationConfig",
    "Generation",
    "HFBackend",
    "IfThenSerializer",
    "ImputerArgs",
    "JSONSerializer",
    "JupyterCallback",
    "KeyValueSerializer",
    "LRScheduler",
    "LanguageModelBackend",
    "LanguageModelClassifier",
    "LanguageModelImputer",
    "LanguageModelOverSampler",
    "LanguageModelRegressor",
    "LanguageModelSynthesizer",
    "LoRAConfig",
    "LoggingCallback",
    "MLXBackend",
    "Memory",
    "ModelConfig",
    "NumberFormat",
    "NumericSpan",
    "Optimizer",
    "OversamplerArgs",
    "PlainNumber",
    "Precision",
    "PredictEnd",
    "PredictStart",
    "Quantization",
    "QuantizationConfig",
    "RegressorArgs",
    "Retry",
    "RichCallback",
    "RowEnd",
    "Score",
    "Serializer",
    "SpacedDigits",
    "TabularLanguageModel",
    "TqdmCallback",
    "TrainExamples",
    "TrainReport",
    "TrainingConfig",
    "TrainingExample",
    "TrainingState",
    "ValueConstraint",
    "aggregate_default",
]

logging.getLogger("sklm").addHandler(logging.NullHandler())
