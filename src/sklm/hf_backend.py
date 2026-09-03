# torch/transformers come from the optional 'hf' extra and are absent from the
# type-check environment, so pyright cannot resolve them; suppress the import errors.
# When torch is installed locally its stubs also fail to re-export public names
# (torch.where, torch.arange, ...), so the private-import check is silenced too.
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pyright: reportPrivateImportUsage=false
"""Hugging Face causal-LM backend built on the ``transformers`` ``Trainer``.

Requires the ``hf`` extra (``pip install scikit-lm[hf]``). torch and
transformers are imported lazily, so they stay an optional dependency. The
:class:`HFBackend` implements the :class:`~sklm.LanguageModelBackend` protocol.
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import json
import math
import os
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from .backend import (
    DigitTokens,
    EarlyStopping,
    checkpoint_workdir,
    common_token_prefix,
    numeric_token_arrays,
    prompt_groups,
    resolve_digit_tokens,
    resolve_max_new_tokens,
    resolve_max_seq_length,
)
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
from .serialize import TrainingExample, ValueConstraint

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
        self._constraint_masks: dict[ValueConstraint, Any] = {}

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
        self._constraint_masks = {}  # per-tokenizer cache; a reload invalidates it

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
        from transformers import Trainer, TrainingArguments
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

        custom_loss = training.target_loss_weight is not None or training.numeric_loss_weight > 0
        digit_tokens = (
            resolve_digit_tokens(lambda t: tok(t, add_special_tokens=False)["input_ids"])
            if training.numeric_loss_weight > 0
            else None
        )

        with checkpoint_workdir("sklm_hf_") as tmp:
            ckpt = training.checkpoint
            output_dir = (ckpt.dir if ckpt is not None else None) or tmp
            dataset = _text_dataset(epoch_texts, tok, seq_len, training, digit_tokens)
            eval_dataset = (
                _text_dataset(lambda _: eval_examples, tok, seq_len, training, digit_tokens)
                if eval_examples
                else None
            )
            train_kwargs = _training_kwargs(
                training,
                self._device or "cpu",
                self._precision,
                output_dir,
                len(dataset),
                random_state,
                has_eval=eval_dataset is not None,
            )
            if custom_loss:
                # The custom compute_loss applies the smoothing itself; leaving the
                # factor set would also build the Trainer's unused label_smoother.
                train_kwargs["label_smoothing_factor"] = 0.0
                # The Trainer wraps the collator to drop columns outside the model's
                # forward signature; the weighted-loss columns must reach compute_loss
                # (which pops them before the forward) instead.
                train_kwargs["remove_unused_columns"] = False
            args = TrainingArguments(**train_kwargs)

            # When ``dir`` already holds checkpoints, resume from the most recent
            # one (weights + optimizer + scheduler + step) instead of the base
            # model; ``get_last_checkpoint`` returns None for an empty/new dir.
            resumable = ckpt is not None and ckpt.dir is not None
            resume = (
                get_last_checkpoint(output_dir) if resumable and os.path.isdir(output_dir) else None
            )
            best, stopping = _resume_trackers(
                output_dir,
                resume,
                training.evaluation.patience if training.evaluation is not None else None,
            )
            report = _report_callback(
                callback,
                self._device_memory_bytes,
                lm,
                stopping,
                best=best,
                ckpt_dir=output_dir if resumable else None,
            )
            trainer_cls = _make_weighted_trainer(training, digit_tokens) if custom_loss else Trainer
            trainer = trainer_cls(
                model=lm,
                args=args,
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                data_collator=_causal_collator(tok, training),
                callbacks=[_reshuffle_callback(dataset), report],
                optimizers=self._resolve_mps_optimizer(training),
            )
            with contextlib.suppress(ValueError):
                trainer.remove_callback(PrinterCallback)
                trainer.remove_callback(ProgressCallback)
            lm.train()
            trainer.train(resume_from_checkpoint=resume)
            if report.best is not None:
                _load_weights(lm, report.best)
            lm.eval()

    def _constraint_mask(self, constraint: ValueConstraint) -> Any:
        """Boolean vocab mask of the tokens ``constraint`` allows (plus EOS)."""
        mask = self._constraint_masks.get(constraint)
        if mask is None:
            tok = self._tokenizer
            allowed = [constraint.allows(tok.decode([i])) for i in range(len(tok))]
            mask = self._torch.tensor(allowed, dtype=self._torch.bool)
            if tok.eos_token_id is not None:
                mask[tok.eos_token_id] = True
            self._constraint_masks[constraint] = mask
        return mask

    def generate(
        self,
        prompts: Sequence[str],
        generation: GenerationConfig,
        *,
        constraint: ValueConstraint | None = None,
        random_state: int | None = None,
    ) -> list[str]:
        """Sample a continuation per prompt (greedy when ``temperature <= 0``)."""
        if not prompts:
            return []
        if random_state is not None:
            self._torch.manual_seed(random_state)
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
        if constraint is not None:
            from transformers import LogitsProcessorList

            allowed = self._constraint_mask(constraint).to(device)

            def _mask_logits(input_ids: Any, scores: Any) -> Any:
                n = allowed.shape[0]
                scores[:, :n] = scores[:, :n].masked_fill(~allowed, float("-inf"))
                # Model logits may be padded past the tokenizer vocab; those ids
                # decode to nothing meaningful, so they are always masked.
                if scores.shape[-1] > n:
                    scores[:, n:] = float("-inf")
                return scores

            sampling["logits_processor"] = LogitsProcessorList([_mask_logits])
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

        Pairs are scored in two phases: consecutive pairs sharing a prompt form
        a group whose common token prefix runs once (a left-padded batch with
        one row per group, primed into the model's KV cache), then every pair's
        remaining tokens are scored in one right-padded batch on top of its
        group's replicated cache -- the shared prompt is not re-forwarded per
        candidate. The continuation boundary per pair is the end of the longest
        token prefix shared with its prompt, so a BPE merge at the prompt's
        trailing space (the boundary token) is scored rather than dropped.
        ``reduce`` collapses each pair's per-token log-likelihoods to the
        ``"mean"`` (default) or the ``"sum"`` -- kept identical to
        :meth:`MLXBackend.score` (invariant #3).
        """
        if not prompts:
            return []
        torch, tok, lm = self._torch, self._tokenizer, self._model
        # Right truncation keeps the prompt prefix intact so the longest-common-prefix
        # boundary below stays valid; MLX truncates the same side, so both score the
        # same span for over-length pairs (invariant: identical ranking per backend).
        tok.truncation_side = "right"
        device = next(lm.parameters()).device
        groups = prompt_groups(prompts)
        full_ids: list[list[int]] = []
        starts: list[int] = []
        for a, b in groups:
            prompt_ids = tok(prompts[a])["input_ids"]  # once per run, not per pair
            for i in range(a, b):
                ids = tok(
                    prompts[i] + continuations[i], truncation=True, max_length=self._max_seq_length
                )["input_ids"]
                starts.append(max(common_token_prefix(prompt_ids, ids), 1))
                full_ids.append(ids)

        # Every full in a group agrees with its shared prompt on the first ``start``
        # tokens, so ``min(starts) - 1`` is a token prefix common to the whole group;
        # the ``- 1`` keeps the boundary token in the suffix, so its logit comes from
        # the suffix pass and the prefill's logits are never needed.
        prefix_lens = [max(min(starts[a:b]) - 1, 0) for a, b in groups]
        pair_prefix = [0] * len(prompts)
        for g, (a, b) in enumerate(groups):
            pair_prefix[a:b] = [prefix_lens[g]] * (b - a)
        suffixes = [ids[c : len(ids) - 1] for ids, c in zip(full_ids, pair_prefix, strict=True)]
        s_max = max(len(s) for s in suffixes)
        if s_max == 0:  # every full is a single token: nothing is scorable
            return [float("-inf")] * len(prompts)

        pad_id = tok.pad_token_id or 0
        scores: list[float] = []
        with self._inference_context(device):
            past = prefix_mask = None
            if max(prefix_lens) > 0:
                c_max = max(prefix_lens)
                prefix_ids = torch.full((len(groups), c_max), pad_id, dtype=torch.long)
                prefix_mask = torch.zeros((len(groups), c_max), dtype=torch.long)
                for g, (a, _) in enumerate(groups):
                    if prefix_lens[g]:  # left padding; a 0-length prefix row stays all-pad
                        prefix_ids[g, c_max - prefix_lens[g] :] = torch.tensor(
                            full_ids[a][: prefix_lens[g]]
                        )
                        prefix_mask[g, c_max - prefix_lens[g] :] = 1
                prefix_ids, prefix_mask = prefix_ids.to(device), prefix_mask.to(device)
                positions = (prefix_mask.cumsum(-1) - 1).clamp(min=0)
                past = lm(
                    input_ids=prefix_ids,
                    attention_mask=prefix_mask,
                    position_ids=positions,
                    use_cache=True,
                ).past_key_values
                members = torch.tensor(
                    [g for g, (a, b) in enumerate(groups) for _ in range(a, b)], device=device
                )
                past.batch_select_indices(members)  # replicate each group's row, one per pair
                prefix_mask = prefix_mask[members]
            suffix_ids = torch.full((len(suffixes), s_max), pad_id, dtype=torch.long)
            suffix_mask = torch.zeros((len(suffixes), s_max), dtype=torch.long)
            targets = torch.zeros((len(suffixes), s_max), dtype=torch.long)
            for i, (suffix, ids) in enumerate(zip(suffixes, full_ids, strict=True)):
                suffix_ids[i, : len(suffix)] = torch.tensor(suffix)
                suffix_mask[i, : len(suffix)] = 1
                targets[i, : len(suffix)] = torch.tensor(ids[pair_prefix[i] + 1 :])
            suffix_ids, suffix_mask = suffix_ids.to(device), suffix_mask.to(device)
            positions = torch.tensor(pair_prefix, device=device)[:, None] + torch.arange(
                s_max, device=device
            )
            mask = (
                suffix_mask if prefix_mask is None else torch.cat([prefix_mask, suffix_mask], dim=1)
            )
            logits = lm(
                input_ids=suffix_ids,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=past,
            ).logits
            # Cast to fp32 before the softmax so bf16/fp16 weights rank verbalizers
            # the same as the MLX backend (which casts identically); see invariant #3.
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            token_logprobs = logprobs.gather(-1, targets.to(device).unsqueeze(-1)).squeeze(-1)
            for row, ids in enumerate(full_ids):
                cont = token_logprobs[
                    row, starts[row] - 1 - pair_prefix[row] : len(ids) - 1 - pair_prefix[row]
                ]
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


type _Weights = dict[str, Any]

_BEST_FILE = "best.safetensors"
_STATE_FILE = "sklm_state.json"


def _report_callback(
    callback: Callback,
    device_memory: Callable[[], int | None],
    lm: Any,
    stopping: EarlyStopping,
    *,
    best: _Weights | None,
    ckpt_dir: str | None,
) -> Any:
    """Build a ``TrainerCallback`` that forwards train/eval loss and the current
    device memory to ``callback`` and runs early stopping.

    Every evaluation is fed to ``stopping``: an improvement snapshots the
    trainable weights into ``.best`` for the caller to restore (and, with
    ``ckpt_dir`` set, persists them to ``best.safetensors`` on the spot);
    exhausted patience stops the ``Trainer``. With ``ckpt_dir`` set, every
    checkpoint the ``Trainer`` writes gets a ``sklm_state.json`` sidecar carrying
    the no-improvement streak, so a resumed run continues the count 1:1 -- the
    mirror of the MLX backend's ``state-<step>`` sidecar. ``best`` seeds the
    snapshot from a resumed ``best.safetensors``.
    """
    from transformers import TrainerCallback

    class _Report(TrainerCallback):
        def __init__(self) -> None:
            self.best = best

        def on_log(self, args: object, state: Any, control: object, **kw: Any) -> None:
            logs = kw.get("logs") or {}
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

        def on_evaluate(self, args: object, state: Any, control: Any, **kw: Any) -> Any:
            loss = float(kw["metrics"]["eval_loss"])
            callback.on_eval_report(step=state.global_step, loss=loss, epoch=state.epoch)
            match stopping.observe(loss):
                case "improved":
                    self.best = {
                        name: p.detach().to("cpu", copy=True)
                        for name, p in lm.named_parameters()
                        if p.requires_grad
                    }
                    if ckpt_dir is not None:
                        _save_best(ckpt_dir, self.best, loss)
                case "exhausted":
                    control.should_training_stop = True
                case "no_improvement":
                    pass
            return control

        def on_save(self, args: object, state: Any, control: object, **kw: Any) -> None:
            if ckpt_dir is None:
                return
            sidecar = Path(ckpt_dir) / f"checkpoint-{state.global_step}" / _STATE_FILE
            sidecar.write_text(json.dumps({"streak": stopping.streak}))

    return _Report()


def _save_best(ckpt_dir: str, weights: _Weights, loss: float) -> None:
    """Persist the best (lowest-eval-loss) trainable weights to ``best.safetensors``.

    Written the moment the best improves (not only at the end) so an interrupted
    run can still restore it; the loss rides in the file's metadata so resume
    recovers the bar to beat."""
    from safetensors.torch import save_file

    save_file(weights, str(Path(ckpt_dir) / _BEST_FILE), metadata={"loss": repr(loss)})


def _resume_trackers(
    ckpt_dir: str, resume: str | None, patience: int | None
) -> tuple[_Weights | None, EarlyStopping]:
    """The best weights and early-stopping tracker a resumed run continues from.

    ``best.safetensors`` (when ``ckpt_dir`` holds one) seeds the best weights and
    loss; the ``sklm_state.json`` sidecar inside the ``resume`` checkpoint seeds
    the no-improvement streak. A fresh run starts both empty.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    best: _Weights | None = None
    best_loss = math.inf
    best_file = Path(ckpt_dir) / _BEST_FILE
    if best_file.exists():
        best = load_file(str(best_file))
        with safe_open(str(best_file), framework="pt") as f:
            best_loss = float(f.metadata()["loss"])
    streak = 0
    sidecar = Path(resume) / _STATE_FILE if resume is not None else None
    if sidecar is not None and sidecar.exists():
        streak = int(json.loads(sidecar.read_text())["streak"])
    return best, EarlyStopping(patience, best=best_loss, streak=streak)


def _load_weights(lm: Any, weights: _Weights) -> None:
    """Copy ``weights`` (a ``named_parameters`` subset) back into ``lm`` in place."""
    import torch

    with torch.no_grad():
        for name, p in lm.named_parameters():
            if name in weights:
                p.copy_(weights[name])


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
    steps_per_epoch = -(-n_rows // per_step)  # ceil div
    epoch_steps = steps_per_epoch * training.epochs
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
    _apply_eval_kwargs(kwargs, training, steps_per_epoch, has_eval)
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
            # The Trainer steps ReduceLROnPlateau with the metric named here.
            kwargs["metric_for_best_model"] = "eval_loss"
            kwargs["greater_is_better"] = False
        case _:
            raise ValueError(f"unknown lr_scheduler {type(scheduler).__name__}")


def _apply_eval_kwargs(
    kwargs: dict[str, Any], training: TrainingConfig, steps_per_epoch: int, has_eval: bool
) -> None:
    """Layer the checkpoint and evaluation cadences onto ``kwargs``.

    Both cadences are expressed in optimizer steps (``on="epoch"`` scales
    ``each`` by ``steps_per_epoch``), the same conversion the MLX backend
    applies, so save and eval schedules are independent of each other and
    identical across backends. Best-model tracking and early stopping live in
    :func:`_report_callback`, not in the ``Trainer``.
    """
    ckpt = training.checkpoint
    if ckpt is not None:
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = ckpt.each * (steps_per_epoch if ckpt.on == "epoch" else 1)
        kwargs["save_total_limit"] = ckpt.keep
        kwargs["save_only_model"] = not training.save_optimizer_state
    evaluation = training.evaluation
    if has_eval and evaluation is not None:
        kwargs["per_device_eval_batch_size"] = training.batch_size
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = evaluation.each * (
            steps_per_epoch if evaluation.on == "epoch" else 1
        )


def _causal_collator(
    tokenizer: Any, training: TrainingConfig
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        # ``labels`` (and the weighted-loss columns, when active) ride as
        # first-class columns, so they survive the Trainer's
        # ``remove_unused_columns``. Pull them out before ``tokenizer.pad`` --
        # which only pads input_ids/attention_mask -- then right-pad each to the
        # batch width (matching the right-padded input_ids; -100 labels and
        # zero weights/scales are ignored by the loss).
        extras: list[tuple[str, list[list[Any]], float | int, Any]] = [
            ("labels", [f.pop("labels") for f in features], -100, torch.int64)
        ]
        if training.target_loss_weight is not None:
            extras.append((
                "loss_weights",
                [f.pop("loss_weights") for f in features],
                0.0,
                torch.float32,
            ))
        # Per-number columns (one entry per numeric span, not per token) pad to
        # the batch's slot count instead of its token width.
        slotted: dict[str, list[list[float]]] = {}
        if training.numeric_loss_weight > 0:
            extras.append((
                "digit_scale",
                [f.pop("digit_scale") for f in features],
                0.0,
                torch.float32,
            ))
            extras.append((
                "digit_variant",
                [f.pop("digit_variant") for f in features],
                -1,
                torch.int64,
            ))
            extras.append(("number_id", [f.pop("number_id") for f in features], -1, torch.int64))
            slotted["numeric_targets"] = [f.pop("numeric_targets") for f in features]
            slotted["numeric_weights"] = [f.pop("numeric_weights") for f in features]
        batch = tokenizer.pad(features, return_tensors="pt")
        token_width = batch["input_ids"].shape[1]

        def right_pad(rows: list[list[Any]], fill: float | int, dtype: Any, width: int) -> Any:
            out = torch.full((len(rows), width), fill, dtype=dtype)
            for i, row in enumerate(rows):
                out[i, : len(row)] = torch.tensor(row, dtype=dtype)
            return out

        for key, rows, fill, dtype in extras:
            batch[key] = right_pad(rows, fill, dtype, token_width)
        if slotted:
            slots = max((len(t) for t in slotted["numeric_targets"]), default=0) or 1
            for key, rows in slotted.items():
                batch[key] = right_pad(rows, 0.0, torch.float32, slots)
        return batch

    return collate


def _text_dataset(
    epoch_texts: Callable[[int], list[TrainingExample]],
    tokenizer: Any,
    max_seq_length: int,
    training: TrainingConfig,
    digit_tokens: DigitTokens | None,
) -> Any:
    """Build a torch map-style dataset over per-epoch (re)serialized rows.

    ``reshuffle`` refreshes the buffer from ``epoch_texts`` for the next epoch
    index so the column-order permutation is redrawn at each epoch boundary
    (driven by a Trainer callback). Tokenization is lazy in ``__getitem__``;
    ``labels`` is a copy of ``input_ids`` (the collator pads it with -100).

    Under ``target_loss_weight`` an example's non-empty ``prompt`` becomes the
    ``loss_weights`` column (``1 - alpha`` up to the boundary, ``alpha``
    after); the boundary is the longest common token prefix between the prompt
    and the full text (robust to BPE merging the edge). Under
    ``numeric_loss_weight`` the example's ``numeric_spans`` become the
    per-token digit columns of :func:`~sklm.backend.numeric_token_arrays`.
    Both are consumed by the custom-loss Trainer.
    """
    from torch.utils.data import Dataset as TorchDataset

    eos = tokenizer.eos_token or ""
    alpha = training.target_loss_weight
    numeric = training.numeric_loss_weight > 0
    variant_of = digit_tokens.variant_of if digit_tokens is not None else {}

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
            ids = enc["input_ids"]
            item = {"input_ids": ids, "attention_mask": enc["attention_mask"], "labels": list(ids)}
            if alpha is not None:
                plen = 0
                if ex.prompt:
                    prompt_ids = tokenizer(ex.prompt)["input_ids"]
                    # Always keep at least one full-weight token: at alpha=1.0 an
                    # all-prompt row would otherwise have zero loss weight.
                    plen = min(common_token_prefix(prompt_ids, ids), len(ids) - 1)
                item["loss_weights"] = [1.0 - alpha] * plen + [alpha] * (len(ids) - plen)
            if numeric:
                arrays = numeric_token_arrays(
                    ex, ids, lambda t: tokenizer(t)["input_ids"], variant_of
                )
                item["digit_scale"] = arrays.scale
                item["digit_variant"] = arrays.variant
                item["number_id"] = arrays.number_id
                item["numeric_targets"] = arrays.targets
                item["numeric_weights"] = arrays.weights
            return item

    return _TextDataset()


def _make_weighted_trainer(training: TrainingConfig, digit_tokens: DigitTokens | None) -> type:
    """Trainer subclass computing the weighted / numeric-augmented loss.

    Active when ``target_loss_weight`` or ``numeric_loss_weight`` is set: the
    cross-entropy is computed manually (label smoothing included -- the
    ``label_smoothing_factor`` argument is zeroed by ``fit``) over shifted
    logits, weighted per token and normalized by the total weight. The numeric
    term restricts a softmax to the digit-token candidates at each digit
    position, reconstructs the expected number per ``number_id`` slot and adds
    the mean squared error against ``numeric_targets``, each slot's error
    scaled by ``numeric_weights`` (the reciprocal of its column's standard
    deviation, so unlike-magnitude columns weigh alike)
    (kept behaviorally identical to ``mlx_backend._make_loss`` -- invariant 3).
    """
    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    eps = training.label_smoothing
    numeric_weight = training.numeric_loss_weight
    candidates = digit_tokens.candidates if digit_tokens is not None else ((), ())

    class _WeightedLossTrainer(Trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # The loss is a per-micro-batch weighted mean; opt out of the
            # Trainer's num_items_in_batch scaling (see Trainer.compute_loss).
            self.model_accepts_loss_kwargs = False

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            labels = inputs.pop("labels")
            weights = inputs.pop("loss_weights", None)
            scale = inputs.pop("digit_scale", None)
            variant = inputs.pop("digit_variant", None)
            number_id = inputs.pop("number_id", None)
            targets = inputs.pop("numeric_targets", None)
            inv_scale = inputs.pop("numeric_weights", None)
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            shift_logits = logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            ce = F.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                ignore_index=-100,
                label_smoothing=eps,
                reduction="none",
            )
            if weights is not None:
                w = weights[:, 1:] * (shift_labels != -100)
            else:
                w = (shift_labels != -100).to(ce.dtype)
            loss = (ce * w).sum() / w.sum()
            if scale is not None:
                # Position t's digit token is predicted by the logits at t - 1,
                # so the per-token arrays shift exactly like the labels.
                s, v, nid = scale[:, 1:], variant[:, 1:], number_id[:, 1:]
                expected = torch.zeros_like(s)
                for var, cand in enumerate(candidates):
                    if not cand:
                        continue
                    p = torch.softmax(shift_logits[..., list(cand)], dim=-1)
                    digits = torch.arange(10, dtype=p.dtype, device=p.device)
                    expected = torch.where(v == var, (p * digits).sum(-1), expected)
                bsz, slots = targets.shape
                base = torch.arange(bsz, device=nid.device).unsqueeze(1) * slots
                # Non-digit positions land in a dump slot past the real ones.
                flat = torch.where(nid >= 0, nid + base, torch.full_like(nid, bsz * slots))
                sums = torch.zeros(bsz * slots + 1, dtype=s.dtype, device=s.device)
                sums.index_add_(0, flat.reshape(-1), (expected * s).reshape(-1))
                counts = torch.zeros_like(sums)
                counts.index_add_(0, flat.reshape(-1), (nid >= 0).reshape(-1).to(s.dtype))
                valid = counts[:-1] > 0
                # inv_scale puts the error in standard deviations of its column,
                # so columns of unlike magnitude weigh alike (invariant 3: MLX
                # computes the same product).
                err = (
                    (sums[:-1] - targets.reshape(-1).to(s.dtype))
                    * inv_scale.reshape(-1).to(s.dtype)
                ) ** 2
                loss = loss + numeric_weight * (err * valid).sum() / valid.sum().clamp(min=1)
            return (loss, outputs) if return_outputs else loss

    return _WeightedLossTrainer
