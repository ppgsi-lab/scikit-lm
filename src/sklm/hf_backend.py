# torch/transformers come from the optional 'hf' extra and are absent from the
# type-check environment, so pyright cannot resolve them; suppress the import errors.
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Hugging Face causal-LM backend built on the ``transformers`` ``Trainer``.

Requires the ``hf`` extra (``pip install scikit-lm[hf]``). torch and
transformers are imported lazily, so they stay an optional dependency. The
:class:`HFBackend` implements the :class:`~sklm.LanguageModelBackend` protocol.
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import os
import tempfile
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import Any, Literal

from .backend import common_token_prefix, resolve_max_new_tokens, resolve_max_seq_length
from .bridge import Model
from .callbacks import Callback
from .config import (
    ConstantLR,
    CosineLR,
    GenerationConfig,
    LinearLR,
    LoRAConfig,
    LRScheduler,
    ModelConfig,
    PlateauLR,
    QuantizationConfig,
    TrainingConfig,
)
from .serialize import TrainingExample

__all__ = ["HFBackend"]

_OPTIM_MAP: dict[str, str] = {
    "adamw": "adamw_torch",
    "adamw_8bit": "adamw_8bit",
    "paged_adamw_8bit": "paged_adamw_8bit",
    "adafactor": "adafactor",
    "lion": "lion_8bit",
}
# Bit widths each HF quantization library can produce. ``"auto"`` routes 4-/8-bit
# to bitsandbytes (the established QLoRA path) and the rest to HQQ.
_HF_METHODS: dict[str, frozenset[int]] = {
    "bitsandbytes": frozenset({4, 8}),
    "hqq": frozenset({1, 2, 3, 4, 8}),
}
# HF ``optim`` strings backed by bitsandbytes (CUDA-only kernels).
_BNB_OPTIM_STRINGS = frozenset({"adamw_8bit", "paged_adamw_8bit", "lion_8bit"})
# Optimizer choices for which mps-bitsandbytes provides an MPS replacement.
_MPS_OPTIM = frozenset({"adamw_8bit", "paged_adamw_8bit", "lion"})


class HFBackend:
    """A Hugging Face causal-LM backend built on the ``transformers`` ``Trainer``.

    Requires the ``hf`` extra (``pip install scikit-lm[hf]``). torch and
    transformers are imported only on first use. The model is (re)loaded from
    its pretrained weights on every :meth:`fit`, so refitting starts clean.

    Training routes through the HF ``Trainer``, with per-epoch feature-order
    permutation restored by re-sampling the dataset at each epoch boundary.
    """

    def __init__(self) -> None:
        # torch/transformers objects have no usable stubs; typed Any at the boundary.
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._device: str | None = None
        self._precision: str = "fp32"
        self._max_seq_length: int = 256
        self._mps_quantized: bool = False

    def _resolve_device(self, device: str) -> str:
        torch = self._torch
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self, model: Model, model_config: ModelConfig) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "HFBackend requires the 'hf' extra: pip install scikit-lm[hf]"
            ) from exc

        self._torch = torch
        dtype = {
            "fp32": self._torch.float32,
            "bf16": self._torch.bfloat16,
            "fp16": self._torch.float16,
        }[model_config.precision]
        device = self._resolve_device(model_config.device)

        # A factory defers loading to here; its product is a model id or a loaded model.
        if callable(model) and not isinstance(model, (str, PreTrainedModel)):
            model = model()

        tokenizer = self._resolve_tokenizer(model_config, model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._mps_quantized = False
        method: str | None = None
        lm: Any
        if isinstance(model, str):
            quantization = model_config.quantization
            if quantization is not None:
                method = _resolve_hf_method(quantization)
                if method == "bitsandbytes" and device not in ("cuda", "mps"):
                    raise ValueError(
                        f"bitsandbytes quantization needs CUDA or MPS (mps-bitsandbytes); "
                        f"got device={device!r}"
                    )

            # The MLX backend's text models carry no dropout, so silence the HF base
            # model's config dropout (GPT-2 ships resid/embd/attn_pdrop=0.1) to keep
            # fine-tuning regularization matched across backends.
            config: Any = AutoConfig.from_pretrained(
                model, trust_remote_code=model_config.trust_remote_code
            )
            _zero_base_dropout(config)
            load_kwargs: dict[str, Any] = {
                "config": config,
                "trust_remote_code": model_config.trust_remote_code,
            }
            if model_config.attn_implementation is not None:
                load_kwargs["attn_implementation"] = model_config.attn_implementation
            if quantization is not None and method == "hqq":
                # HQQ quantizes at load and dispatches via device_map; works on CUDA
                # (kernels) and CPU (pure PyTorch), unlike bitsandbytes.
                load_kwargs["quantization_config"] = _hqq_config(quantization)
                load_kwargs["device_map"] = device
                load_kwargs["dtype"] = dtype
            elif quantization is not None and method == "bitsandbytes" and device == "cuda":
                load_kwargs["quantization_config"] = _quant_config(quantization, dtype)
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["dtype"] = dtype

            lm = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
            if quantization is not None and method == "bitsandbytes" and device == "mps":
                lm = lm.to(device)
                lm = _quantize_mps(lm, quantization, dtype)
                self._mps_quantized = True
            elif quantization is None:
                lm = lm.to(device)
        elif isinstance(model, PreTrainedModel):
            if model_config.quantization is not None:
                raise ValueError(
                    "quantization applies to a model id the backend loads, not a "
                    "pre-loaded model; quantize it before passing it, or pass an id"
                )
            lm = model
            lm = lm.to(device)
        else:
            raise TypeError(
                "HFBackend expected a model id, a PreTrainedModel, or a factory "
                f"returning one; got {type(model).__name__}"
            )

        lm.generation_config.max_length = None
        # Make the embedding output require grad so gradients reach LoRA adapters
        # through frozen/quantized layers (also a prerequisite for checkpointing).
        lm.enable_input_require_grads()

        if model_config.lora is not None:
            lm = _apply_lora(
                lm,
                model_config.lora,
                prepare_kbit=method == "bitsandbytes" and not self._mps_quantized,
            )

        self._tokenizer = tokenizer
        self._model = lm
        self._device = device
        self._precision = model_config.precision

    def _resolve_tokenizer(self, model_config: ModelConfig, model: Model) -> Any:
        """Resolve the tokenizer spec to a loaded tokenizer.

        A factory is invoked; an already-loaded tokenizer is used as-is; a string
        id is loaded; ``None`` derives the tokenizer from a model-id string, and is
        rejected when ``model`` is a pre-loaded object (no id to derive from).
        """
        from transformers import AutoTokenizer, PreTrainedTokenizerBase

        spec = model_config.tokenizer
        if callable(spec) and not isinstance(spec, (str, PreTrainedTokenizerBase)):
            spec = spec()
        if isinstance(spec, PreTrainedTokenizerBase):
            return spec
        if spec is None:
            if not isinstance(model, str):
                raise ValueError(
                    "tokenizer must be set when model is a pre-loaded object "
                    "(there is no model id to derive it from)"
                )
            spec = model
        return AutoTokenizer.from_pretrained(spec, trust_remote_code=model_config.trust_remote_code)

    def fit(
        self,
        epoch_texts: Callable[[int], list[TrainingExample]],
        training: TrainingConfig,
        model_config: ModelConfig,
        *,
        random_state: int | None,
        callback: Callback,
        eval_examples: list[TrainingExample] | None = None,
    ) -> None:
        from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
        from transformers.trainer_callback import PrinterCallback, ProgressCallback
        from transformers.trainer_utils import get_last_checkpoint

        sched = training.lr_scheduler
        training = replace(
            training,
            lr_scheduler=replace(sched, learning_rate=sched.resolved_learning_rate(model_config)),
        )

        import torch

        # Drop the previous fit's model/tokenizer before reclaiming memory so the
        # collect/cache-clear frees them, avoiding transient double allocation on refit.
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self._load(model_config.model, model_config)
        tok, lm = self._tokenizer, self._model
        tok.padding_side = "right"
        tok.truncation_side = "right"
        seq_len = training.max_seq_length
        if seq_len is None:
            eos = tok.eos_token or ""
            measured = [*epoch_texts(0), *(eval_examples or [])]
            seq_len = resolve_max_seq_length(
                measured, lambda text: len(tok(text + eos)["input_ids"])
            )
            training = replace(training, max_seq_length=seq_len)
        self._max_seq_length = seq_len

        with tempfile.TemporaryDirectory(prefix="sklm_hf_") as tmp:
            ckpt = training.checkpoint
            output_dir = (ckpt.dir if ckpt is not None else None) or tmp
            dataset = _text_dataset(epoch_texts, tok, seq_len)
            eval_dataset = (
                _text_dataset(lambda _: eval_examples, tok, seq_len) if eval_examples else None
            )
            args = TrainingArguments(
                **_training_kwargs(
                    training,
                    self._device or "cpu",
                    self._precision,
                    output_dir,
                    len(dataset),
                    random_state,
                    has_eval=eval_dataset is not None,
                )
            )

            trainer_callbacks: list[Any] = [
                _reshuffle_callback(dataset),
                _loss_callback(callback, self._device_memory_bytes),
            ]
            if eval_dataset is not None and training.early_stopping_patience is not None:
                trainer_callbacks.append(
                    EarlyStoppingCallback(early_stopping_patience=training.early_stopping_patience)
                )

            trainer = Trainer(
                model=lm,
                args=args,
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                data_collator=_causal_collator(tok),
                callbacks=trainer_callbacks,
                optimizers=self._resolve_mps_optimizer(training),
            )
            with contextlib.suppress(ValueError):
                trainer.remove_callback(PrinterCallback)
                trainer.remove_callback(ProgressCallback)
            # When ``dir`` already holds checkpoints, resume from the most recent
            # one (weights + optimizer + scheduler + step) instead of the base
            # model; ``get_last_checkpoint`` returns None for an empty/new dir.
            resume = (
                get_last_checkpoint(output_dir)
                if ckpt is not None and ckpt.dir is not None and os.path.isdir(output_dir)
                else None
            )
            lm.train()
            trainer.train(resume_from_checkpoint=resume)
            lm.eval()

    def generate(self, prompts: Sequence[str], generation: GenerationConfig) -> list[str]:
        """Sample a continuation per prompt (greedy when ``temperature <= 0``)."""
        if not prompts:
            return []
        tok, lm = self._tokenizer, self._model
        tok.padding_side = "left"
        tok.truncation_side = "left"
        device = next(lm.parameters()).device
        # Cap the prompt at the training sequence length (one whole serialized
        # row), not ``_max_seq_length - max_new_tokens``: an estimator's prompt is
        # itself up to a full row, so reserving room for the continuation would
        # left-truncate it mid-row and strip the conditioning columns. The
        # generated tokens extend past it into the model's own context window.
        enc = tok(
            [p.rstrip() for p in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_seq_length,
        ).to(device)
        if generation.temperature > 0:
            sampling: dict[str, Any] = {
                "do_sample": True,
                "temperature": generation.temperature,
                "top_p": generation.top_p,
            }
            if generation.top_k > 0:
                sampling["top_k"] = generation.top_k
        else:
            sampling = {"do_sample": False}
        if generation.repetition_penalty is not None:
            sampling["repetition_penalty"] = generation.repetition_penalty
        with self._inference_context(device):
            out = lm.generate(
                **enc,
                max_new_tokens=resolve_max_new_tokens(generation, self._max_seq_length),
                pad_token_id=tok.pad_token_id,
                **sampling,
            )
        generated = out[:, enc["input_ids"].shape[1] :]
        return tok.batch_decode(generated, skip_special_tokens=True)

    def score(
        self,
        prompts: Sequence[str],
        continuations: Sequence[str],
        *,
        reduce: Literal["mean", "sum"] = "mean",
    ) -> list[float]:
        """Per-token log-likelihood of ``continuations[i]`` given ``prompts[i]``.

        Pairs are scored as one right-padded batch in a single forward pass. The
        continuation boundary per pair is the end of the longest token prefix
        shared with its prompt, so a BPE merge at the prompt's trailing space
        (the boundary token) is scored rather than dropped. ``reduce`` collapses
        each pair's per-token log-likelihoods to the ``"mean"`` (default) or the
        ``"sum"`` -- kept identical to :meth:`MLXBackend.score` (invariant #3).
        """
        if not prompts:
            return []
        torch, tok, lm = self._torch, self._tokenizer, self._model
        tok.padding_side = "right"
        # Right truncation keeps the prompt prefix intact so the longest-common-prefix
        # boundary below stays valid; MLX truncates the same side, so both score the
        # same span for over-length pairs (invariant: identical ranking per backend).
        tok.truncation_side = "right"
        device = next(lm.parameters()).device
        full_ids: list[list[int]] = []
        starts: list[int] = []
        for prompt, continuation in zip(prompts, continuations, strict=True):
            prompt_ids = tok(prompt)["input_ids"]
            ids = tok(prompt + continuation, truncation=True, max_length=self._max_seq_length)[
                "input_ids"
            ]
            starts.append(max(common_token_prefix(prompt_ids, ids), 1))
            full_ids.append(ids)
        enc = tok.pad({"input_ids": full_ids}, return_tensors="pt").to(device)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        scores: list[float] = []
        with self._inference_context(device):
            logits = lm(input_ids=input_ids, attention_mask=attention_mask).logits
            # Cast to fp32 before the softmax so bf16/fp16 weights rank verbalizers
            # the same as the MLX backend (which casts identically); see invariant #3.
            logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            token_logprobs = logprobs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            for row, ids in enumerate(full_ids):
                # token_logprobs[row, j] scores token j+1; right padding keeps the
                # continuation span (start..len(ids)-1) before any pad position.
                cont = token_logprobs[row, starts[row] - 1 : len(ids) - 1]
                if not cont.numel():
                    scores.append(float("-inf"))
                else:
                    scores.append(float(cont.sum() if reduce == "sum" else cont.mean()))
        return scores

    def _device_memory_bytes(self) -> int | None:
        """Current accelerator memory in use, in bytes (``None`` on CPU).

        CUDA exposes ``memory_allocated``; MPS exposes ``current_allocated_memory``
        (neither device's peak is read here -- MPS has no peak API, so the
        callback layer infers one from the running maximum of these samples)."""
        torch = self._torch
        if self._device == "cuda":
            return int(torch.cuda.memory_allocated())
        if self._device == "mps":
            return int(torch.mps.current_allocated_memory())
        return None

    def _resolve_mps_optimizer(self, training: TrainingConfig) -> tuple[Any, Any]:
        """Build an mps-bitsandbytes optimizer when an 8-bit optimizer is asked
        for on MPS (bitsandbytes has no MPS kernels). Handed to the Trainer via
        ``optimizers=(opt, None)``; ``(None, None)`` lets the Trainer build its
        own from the ``optim`` string."""
        if self._device != "mps" or training.optimizer not in _MPS_OPTIM:
            return (None, None)
        import mps_bitsandbytes as mbnb

        cls = {
            "adamw_8bit": mbnb.AdamW8bit,
            "paged_adamw_8bit": mbnb.PagedAdamW,
            "lion": mbnb.Lion8bit,
        }[training.optimizer]
        opt = cls(
            self._model.parameters(),
            lr=training.lr_scheduler.learning_rate,
            weight_decay=training.weight_decay,
        )
        return (opt, None)

    @contextlib.contextmanager
    def _inference_context(self, device: Any) -> Iterator[None]:
        """Enter no-grad, plus autocast when a reduced precision was configured."""
        torch = self._torch
        autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self._precision)
        use_autocast = autocast_dtype is not None and device.type in ("cuda", "cpu")
        if use_autocast:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
                yield
        else:
            with torch.no_grad():
                yield


def _zero_base_dropout(config: Any) -> None:
    """Zero every dropout rate on a model config, in place.

    The MLX backend's text models implement no dropout, so the HF base model's
    config dropout is silenced to keep fine-tuning regularization matched across
    backends -- GPT-2, for instance, ships ``resid_pdrop = embd_pdrop =
    attn_pdrop = 0.1``. Field names vary by architecture, so any float attribute
    whose name mentions ``dropout`` or ``pdrop`` is caught.
    """
    for attr, value in list(vars(config).items()):
        if isinstance(value, float) and ("dropout" in attr or "pdrop" in attr):
            setattr(config, attr, 0.0)


def _resolve_hf_method(quantization: QuantizationConfig) -> str:
    """Pick the HF quantization library for ``quantization`` and validate its bit
    width, raising a uniform error when no method on this backend can honor it."""
    method = quantization.method
    if method == "auto":
        method = "bitsandbytes" if quantization.bits in _HF_METHODS["bitsandbytes"] else "hqq"
    supported = _HF_METHODS.get(method)
    if supported is None:
        raise ValueError(
            f"quantization method {method!r} is not available on the HF backend; "
            f"choose from {sorted(_HF_METHODS)} or 'auto'"
        )
    if quantization.bits not in supported:
        raise ValueError(
            f"method={method!r} on the HF backend supports bits {sorted(supported)}; "
            f"got bits={quantization.bits}"
        )
    return method


def _quant_config(quantization: QuantizationConfig, dtype: Any) -> Any:
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - transformers always ships it
        raise ImportError("quantization requires transformers with BitsAndBytesConfig") from exc
    if quantization.bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def _hqq_config(quantization: QuantizationConfig) -> Any:
    """Build a transformers ``HqqConfig`` for on-the-fly HQQ quantization."""
    if importlib.util.find_spec("hqq") is None:
        raise ImportError("HQQ quantization requires the 'hqq' extra: pip install scikit-lm[hqq]")
    from transformers import HqqConfig

    kwargs: dict[str, Any] = {"nbits": quantization.bits}
    if quantization.group_size is not None:
        kwargs["group_size"] = quantization.group_size
    return HqqConfig(**kwargs)


def _quantize_mps(model: Any, quantization: QuantizationConfig, dtype: Any) -> Any:
    """In-place bitsandbytes quantization on Apple Silicon via mps-bitsandbytes."""
    try:
        from mps_bitsandbytes import BitsAndBytesConfig as MPSBitsAndBytesConfig
        from mps_bitsandbytes import quantize_model
    except ImportError as exc:
        raise ImportError(
            "MPS quantization requires the 'quant' extra: pip install scikit-lm[quant]"
        ) from exc
    if quantization.bits == 4:
        cfg = MPSBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        cfg = MPSBitsAndBytesConfig(load_in_8bit=True)
    # Keep the output head unquantized (as the CUDA path does), so the LM loss
    # stays differentiable for LoRA/QLoRA training.
    return quantize_model(
        model, quantization_config=cfg, device="mps", modules_to_not_convert=["lm_head"]
    )


def _apply_lora(model: Any, config: LoRAConfig, *, prepare_kbit: bool) -> Any:
    try:
        from peft import LoraConfig, PeftModelForCausalLM, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("LoRA requires peft: pip install scikit-lm[hf]") from exc
    # k-bit prep is a bitsandbytes-CUDA dance; mps-bitsandbytes and HQQ produce
    # ordinary trainable modules that need none of it.
    if prepare_kbit:
        model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules or "all-linear",
        bias="none",
        rank_pattern=config.rank_pattern or {},
        alpha_pattern=config.alpha_pattern or {},
        use_rslora=config.rslora,
        use_dora=config.dora,
    )
    return PeftModelForCausalLM(model, peft_config)


def _reshuffle_callback(dataset: Any) -> Any:
    """Build a ``TrainerCallback`` that re-permutes ``dataset``'s feature order at
    the start of every epoch after the first.

    The first epoch keeps the order the dataset was built with; each later epoch
    boundary calls ``dataset.reshuffle()`` to redraw the per-row serialization.
    """
    from transformers import TrainerCallback

    class _Reshuffle(TrainerCallback):
        def __init__(self) -> None:
            self._started = False

        def on_epoch_begin(self, args: object, state: object, control: object, **kw: object):
            if self._started:
                dataset.reshuffle()
            self._started = True

    return _Reshuffle()


def _loss_callback(callback: Callback, device_memory: Callable[[], int | None]) -> Any:
    """Build a ``TrainerCallback`` that forwards ``on_log`` loss and the current
    device memory to ``callback`` (both at each logging step)."""
    from transformers import TrainerCallback

    class _LossReport(TrainerCallback):
        def on_log(self, args: object, state: Any, control: object, **kw: Any) -> None:
            logs = kw.get("logs") or {}
            if "eval_loss" in logs:
                callback.on_eval_report(
                    step=state.global_step, loss=float(logs["eval_loss"]), epoch=state.epoch
                )
                return
            if "loss" not in logs:
                return
            callback.on_memory(device_memory())
            total = state.max_steps if state.max_steps and state.max_steps > 0 else None
            callback.on_train_report(
                step=state.global_step,
                total_steps=total,
                loss=float(logs["loss"]),
                epoch=state.epoch,
                learning_rate=logs.get("learning_rate"),
                grad_norm=logs.get("grad_norm"),
            )

    return _LossReport()


def _training_kwargs(
    training: TrainingConfig,
    device: str,
    precision: str,
    output_dir: str,
    n_rows: int,
    random_state: int | None,
    has_eval: bool = False,
) -> dict[str, Any]:
    optim = _OPTIM_MAP[training.optimizer]
    if optim in _BNB_OPTIM_STRINGS and device != "cuda":
        if device == "mps" and training.optimizer in _MPS_OPTIM:
            # The real optimizer is injected via Trainer's ``optimizers=``.
            optim = "adamw_torch"
        else:
            warnings.warn(
                f"optimizer={training.optimizer!r} needs bitsandbytes (CUDA) or "
                f"mps-bitsandbytes (MPS); falling back to 'adamw_torch' on {device!r}.",
                RuntimeWarning,
                stacklevel=3,
            )
            optim = "adamw_torch"

    per_step = training.batch_size * training.grad_accumulation_steps
    epoch_steps = -(-n_rows // per_step) * training.epochs  # ceil div * epochs
    max_steps = min(epoch_steps, training.max_steps) if training.max_steps is not None else -1

    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=training.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=training.batch_size,
        gradient_accumulation_steps=training.grad_accumulation_steps,
        learning_rate=training.lr_scheduler.learning_rate,
        weight_decay=training.weight_decay,
        max_grad_norm=training.max_grad_norm if training.max_grad_norm is not None else 0.0,
        optim=optim,
        label_smoothing_factor=training.label_smoothing,
        neftune_noise_alpha=training.neftune_noise_alpha,
        gradient_checkpointing=training.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if training.gradient_checkpointing else None
        ),
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        dataloader_num_workers=0,
        # sklearn convention: random_state=None means non-deterministic. HF's
        # TrainingArguments.seed defaults to a fixed 42 when unset, so draw a
        # fresh seed for None and pass the explicit int through untouched.
        seed=random_state if random_state is not None else int.from_bytes(os.urandom(4)),
    )
    _apply_lr_scheduler_kwargs(kwargs, training.lr_scheduler)
    _apply_eval_kwargs(kwargs, training, has_eval)
    return kwargs


def _apply_lr_scheduler_kwargs(kwargs: dict[str, Any], scheduler: LRScheduler) -> None:
    """Map an :class:`~sklm.LRScheduler` onto the ``TrainingArguments`` schedule keys.

    The step-based shapes set ``lr_scheduler_type`` and ``warmup_ratio``. A
    ``floor`` on cosine/linear switches the type to ``cosine_with_min_lr`` /
    ``polynomial`` (degree 1 = linear) so the decay ends at the floor instead of
    zero. :class:`~sklm.PlateauLR` sets ``reduce_lr_on_plateau`` plus the
    ``lr_scheduler_kwargs`` the ``Trainer`` forwards to
    ``torch.optim.lr_scheduler.ReduceLROnPlateau`` (minimum mode; ``threshold_mode``
    and ``eps`` left at their torch defaults to match the MLX backend).
    """
    match scheduler:
        case ConstantLR():
            kwargs["lr_scheduler_type"] = "constant"
            kwargs["warmup_ratio"] = scheduler.warmup_ratio
        case LinearLR():
            if scheduler.floor > 0.0:
                kwargs["lr_scheduler_type"] = "polynomial"
                kwargs["lr_scheduler_kwargs"] = {"lr_end": scheduler.floor, "power": 1.0}
            else:
                kwargs["lr_scheduler_type"] = "linear"
            kwargs["warmup_ratio"] = scheduler.warmup_ratio
        case CosineLR():
            if scheduler.floor > 0.0:
                kwargs["lr_scheduler_type"] = "cosine_with_min_lr"
                kwargs["lr_scheduler_kwargs"] = {"min_lr": scheduler.floor}
            else:
                kwargs["lr_scheduler_type"] = "cosine"
            kwargs["warmup_ratio"] = scheduler.warmup_ratio
        case PlateauLR():
            kwargs["lr_scheduler_type"] = "reduce_lr_on_plateau"
            kwargs["lr_scheduler_kwargs"] = {
                "mode": "min",
                "factor": scheduler.factor,
                "patience": scheduler.patience,
                "threshold": scheduler.threshold,
                "min_lr": scheduler.floor,
                "cooldown": scheduler.cooldown,
            }
        case _:
            raise ValueError(f"unknown lr_scheduler {type(scheduler).__name__}")


def _apply_eval_kwargs(kwargs: dict[str, Any], training: TrainingConfig, has_eval: bool) -> None:
    """Layer the validation / checkpoint / best-model args onto ``kwargs``.

    A :class:`~sklm.CheckpointConfig` sets the save cadence (steps or epochs),
    directory and retention. A held-out set adds evaluation on the same cadence
    and tracks the lowest-validation-loss checkpoint: the ``Trainer`` requires the
    save and eval strategies to match for ``load_best_model_at_end``, so a save
    cadence is forced on (per epoch into the tmp ``output_dir``) even when no
    explicit checkpoint config was given.
    """
    ckpt = training.checkpoint
    if ckpt is not None and ckpt.on == "step":
        save_strategy, save_steps = "steps", ckpt.each
    else:
        save_strategy, save_steps = "epoch", None

    if ckpt is not None:
        kwargs["save_strategy"] = save_strategy
        if save_steps is not None:
            kwargs["save_steps"] = save_steps
        kwargs["save_total_limit"] = ckpt.keep

    if not has_eval:
        return
    kwargs["per_device_eval_batch_size"] = training.batch_size
    if save_strategy == "steps":
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = save_steps
    else:
        kwargs["eval_strategy"] = "epoch"
    # A validation set means a "best" checkpoint exists: track and restore it
    # (HF protects it from ``save_total_limit`` pruning). Saving must be on for
    # the restore; default to per-epoch saves in the tmp dir when no checkpoint
    # config was given.
    if ckpt is None:
        kwargs["save_strategy"] = save_strategy
        kwargs["save_total_limit"] = 1
    kwargs["load_best_model_at_end"] = True
    kwargs["metric_for_best_model"] = "eval_loss"
    kwargs["greater_is_better"] = False


def _causal_collator(tokenizer: Any) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        # ``labels`` rides as a first-class column (the dataset masks the prompt
        # with -100), so it survives the Trainer's ``remove_unused_columns``. Pull
        # it out before ``tokenizer.pad`` -- which only pads input_ids/
        # attention_mask -- then right-pad it with -100 to the batch width (matches
        # the right-padded input_ids; -100 positions are ignored by the loss).
        labels = [f.pop("labels") for f in features]
        batch = tokenizer.pad(features, return_tensors="pt")
        ids = batch["input_ids"]
        padded = ids.new_full((len(labels), ids.shape[1]), -100)
        for i, lab in enumerate(labels):
            padded[i, : len(lab)] = ids.new_tensor(lab)
        batch["labels"] = padded
        return batch

    return collate


def _text_dataset(
    epoch_texts: Callable[[int], list[TrainingExample]], tokenizer: Any, max_seq_length: int
) -> Any:
    """Build a torch map-style dataset over per-epoch (re)serialized rows.

    ``reshuffle`` refreshes the buffer from ``epoch_texts`` for the next epoch
    index so the column-order permutation is redrawn at each epoch boundary
    (driven by a Trainer callback). Tokenization is lazy in ``__getitem__``, which
    also builds ``labels``: a copy of ``input_ids`` with the prompt span set to
    -100. When an example carries a non-empty ``prompt`` (loss-on-target-only),
    the masked span is the longest common token prefix between the prompt and the
    full text (robust to BPE merging the boundary); the causal collator pads
    dynamically, padding ``labels`` with -100. Emitting ``labels`` directly keeps
    the mask a first-class column that survives the Trainer's
    ``remove_unused_columns``.
    """
    from torch.utils.data import Dataset as TorchDataset

    eos = tokenizer.eos_token or ""

    class _TextDataset(TorchDataset):
        def __init__(self) -> None:
            self._epoch = 0
            self._buffer = epoch_texts(0)

        def reshuffle(self) -> None:
            self._epoch += 1
            self._buffer = epoch_texts(self._epoch)

        def __len__(self) -> int:
            return len(self._buffer)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            ex = self._buffer[idx]
            enc = tokenizer(ex.text + eos, truncation=True, max_length=max_seq_length)
            labels = list(enc["input_ids"])
            if ex.prompt:
                prompt_ids = tokenizer(ex.prompt)["input_ids"]
                n = common_token_prefix(prompt_ids, enc["input_ids"])
                # Always supervise at least one token to avoid an all-masked row.
                plen = min(n, len(enc["input_ids"]) - 1)
                labels[:plen] = [-100] * plen
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            }

    return _TextDataset()
