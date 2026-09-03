# mlx/mlx_lm come from the optional 'mlx' extra and are absent from the
# type-check environment, so pyright cannot resolve them; suppress the import errors.
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""MLX language-model backend (Metal on Apple Silicon, CUDA or CPU on Linux).

Mirrors :class:`~sklm.HFBackend` but runs on Apple's ``mlx`` / ``mlx-lm``
packages. Which compute backend MLX uses is fixed by the installed variant --
``scikit-lm[mlx]`` (Metal), ``[mlx-cpu]``, ``[mlx-cuda12]`` or ``[mlx-cuda13]``
-- and selected automatically at runtime. Training delegates to
``mlx_lm.tuner.trainer.train`` exactly as the HF backend delegates to the HF
``Trainer``; the per-epoch feature-order permutation is restored by a custom
``iterate_batches`` that re-draws each row's serialization at every epoch
boundary. ``mlx`` / ``mlx_lm`` are imported lazily on first use, so importing
``sklm`` never requires them.

``ModelConfig.attn_implementation`` is ignored — MLX ships its own kernels.
"""

from __future__ import annotations

import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

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
    ModelConfig,
    PlateauLR,
    QuantizationConfig,
    TrainingConfig,
)
from .serialize import TrainingExample, ValueConstraint

__all__ = ["MLXBackend"]

# MLX's native quantizer is the only library on this backend; it supports these
# bit widths. The matrix mirrors the HF backend so both validate the same way.
_MLX_METHODS: dict[str, frozenset[int]] = {"mlx": frozenset({2, 3, 4, 6, 8})}
_DTYPE_NAMES: dict[str, str] = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


class _StopTraining(Exception):
    """Raised from the training callback to stop ``mlx_lm`` training early.

    ``mlx_lm.tuner.trainer.train`` has no early-stopping hook, so the patience
    check raises this to break out of the loop; the caller catches it and
    restores the best snapshot."""


class MLXBackend:
    """A causal-LM backend built on Apple's ``mlx`` / ``mlx-lm``.

    Requires an MLX extra matching your hardware: ``scikit-lm[mlx]`` (Metal on
    Apple Silicon), ``[mlx-cpu]``, ``[mlx-cuda12]`` or ``[mlx-cuda13]`` (Linux).
    The base model is reloaded on every :meth:`fit`, so refitting starts clean.
    """

    def __init__(self) -> None:
        # mlx model/tokenizer objects have no usable stubs; typed Any at the boundary.
        self._model: Any = None
        self._tokenizer: Any = None
        self._max_seq_length: int = 256
        self._constraint_biases: dict[ValueConstraint, Any] = {}

    def _load(self, model: Model, model_config: ModelConfig) -> None:
        try:
            import mlx.core as mx
            from mlx.nn import Module
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "MLXBackend requires an MLX extra: pip install 'scikit-lm[mlx]' on "
                "macOS, or 'scikit-lm[mlx-cpu]' / 'scikit-lm[mlx-cuda12]' / "
                "'scikit-lm[mlx-cuda13]' on Linux"
            ) from exc

        device = model_config.device
        # MLX selects its GPU backend automatically (Metal on Apple Silicon, CUDA
        # on Linux/NVIDIA, per the installed mlx variant); only an explicit CPU
        # request overrides that default.
        if device == "cpu":
            mx.set_default_device(mx.Device(mx.cpu))

        # A factory defers loading to here; its product is a model id or a loaded model.
        if callable(model) and not isinstance(model, (str, Module)):
            model = model()

        lm: Any
        tokenizer: Any
        if isinstance(model, str):
            quantization = model_config.quantization
            if quantization is not None:
                _resolve_mlx_method(quantization)
            src = (
                _convert_quantized(
                    model, quantization.bits, model_config.precision, quantization.group_size
                )
                if quantization is not None
                else model
            )

            load_kwargs: dict[str, Any] = {"lazy": True}
            if model_config.trust_remote_code:
                load_kwargs["tokenizer_config"] = {"trust_remote_code": True}
                load_kwargs["model_config"] = {"trust_remote_code": True}

            try:
                loaded = load(src, **load_kwargs)
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"mlx-lm could not load {src!r}. The MLX backend needs an mlx-loadable "
                    "model -- a pre-converted 'mlx-community/*' repo or an HF repo whose "
                    "weights match mlx-lm's layout. Some HF checkpoints are not directly "
                    "loadable; use an mlx-compatible equivalent."
                ) from exc
            # load() returns (model, tokenizer); the 3-tuple form is return_config-only.
            lm, tokenizer = loaded[0], loaded[1]
            resolved = self._resolve_tokenizer(model_config)
            if resolved is not None:
                tokenizer = resolved
        elif isinstance(model, Module):
            if model_config.quantization is not None:
                raise ValueError(
                    "quantization applies to a model id the backend loads, not a "
                    "pre-loaded model; convert it before passing it, or pass an id"
                )
            lm = model
            tokenizer = self._resolve_tokenizer(model_config)
            if tokenizer is None:
                raise ValueError(
                    "tokenizer must be set when model is a pre-loaded object "
                    "(there is no model id to derive it from)"
                )
        else:
            raise TypeError(
                "MLXBackend expected a model id, an mlx model, or a factory "
                f"returning one; got {type(model).__name__}"
            )

        dtypes = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
        lm.set_dtype(dtypes[model_config.precision])
        mx.eval(lm.parameters())

        if model_config.lora is not None:
            _apply_lora(lm, model_config.lora)

        self._model = lm
        self._tokenizer = tokenizer
        self._constraint_biases = {}  # per-tokenizer cache; a reload invalidates it

    def _resolve_tokenizer(self, model_config: ModelConfig) -> Any:
        """Resolve the tokenizer spec to a loaded tokenizer, or ``None`` if unset.

        A factory is invoked; an already-loaded ``TokenizerWrapper`` is used as-is;
        a string id is loaded. ``None`` returns ``None`` so the caller decides (the
        string branch keeps the tokenizer that ``load`` already returned).
        """
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        spec = model_config.tokenizer
        if callable(spec) and not isinstance(spec, (str, TokenizerWrapper)):
            spec = spec()
        if isinstance(spec, TokenizerWrapper):
            return spec
        if isinstance(spec, str):
            return _load_tokenizer(spec, model_config.trust_remote_code)
        return None

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
        import mlx.core as mx
        from mlx.utils import tree_unflatten
        from mlx_lm.tuner.trainer import TrainingArgs
        from mlx_lm.tuner.trainer import train as mlx_train

        # Drop the previous fit's model/tokenizer before reclaiming memory so the
        # collect/cache-clear frees them, avoiding transient double allocation on refit.
        self._model = None
        self._tokenizer = None
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()

        # sklearn convention: random_state=None means non-deterministic, so only
        # seed when an explicit int is given (an int makes the fit reproducible).
        if random_state is not None:
            mx.random.seed(random_state)

        self._load(model_config.model, model_config)
        tok = self._tokenizer
        seq_len = training.max_seq_length
        if seq_len is None:
            measured = [*epoch_texts(0), *(eval_examples or [])]
            seq_len = resolve_max_seq_length(measured, lambda text: len(tok.encode(text)) + 1)
            training = replace(training, max_seq_length=seq_len)
        self._max_seq_length = seq_len
        lm = self._model

        digit_tokens = (
            resolve_digit_tokens(lambda t: tok.encode(t, add_special_tokens=False))
            if training.numeric_loss_weight > 0
            else None
        )
        dataset = _MLXTextDataset(epoch_texts, self._tokenizer, seq_len, digit_tokens)
        # mlx_lm's batcher needs a full batch per step; clamp batch_size down to the
        # dataset size so tiny inputs (down to sklearn's 1-row conformance checks)
        # still train, the way HFBackend's Trainer already does.
        if len(dataset) < training.batch_size:
            training = replace(training, batch_size=len(dataset))
        val_dataset = (
            _MLXTextDataset(lambda _: eval_examples, self._tokenizer, seq_len, digit_tokens)
            if eval_examples
            else None
        )

        # ``train`` counts micro-steps as ``iters`` and applies the optimizer
        # every ``grad_accumulation_steps``; scale so ``max_steps`` stays an
        # optimizer-step ceiling (the documented TrainingConfig semantics).
        micro_per_epoch = -(-len(dataset) // training.batch_size)
        iters = micro_per_epoch * training.epochs
        if training.max_steps is not None:
            iters = min(iters, training.max_steps * training.grad_accumulation_steps)

        # The LR schedule advances once per optimizer.update (every
        # grad_accumulation_steps micro-steps), so it is sized in optimizer steps.
        sched = training.lr_scheduler
        optimizer = _build_optimizer(
            training,
            max(1, iters // training.grad_accumulation_steps),
            sched.resolved_learning_rate(model_config),
        )
        iterate_fn = _make_iterate_batches(random_state, training.numeric_loss_weight > 0)
        loss_fn = (
            _make_loss(training, digit_tokens)
            if (
                training.label_smoothing > 0
                or training.target_loss_weight is not None
                or training.numeric_loss_weight > 0
            )
            else None
        )
        evaluation = training.evaluation
        plateau = _PlateauReducer(sched) if isinstance(sched, PlateauLR) else None

        ckpt = training.checkpoint
        resumable = ckpt is not None and ckpt.dir is not None
        with checkpoint_workdir("sklm_mlx_") as tmpdir:
            ckpt_dir = (ckpt.dir if ckpt is not None else None) or tmpdir
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            # Resume: when ``dir`` already holds snapshots, restore the most recent
            # one -- weights, optimizer state (Adam moments + step + LR schedule
            # position), the dropout/sampling RNG and the plateau/early-stop/best
            # trackers -- then continue only the steps left to the original budget,
            # 1:1 with the HF ``Trainer``'s ``resume_from_checkpoint``.
            resume = _resume_state(lm, ckpt_dir) if resumable else None
            step_offset = 0
            if resume is not None:
                step_offset = resume.resumed_it
                if resume.opt_state is not None:
                    optimizer.init(lm.trainable_parameters())
                    optimizer.state = tree_unflatten(resume.opt_state)
                    # State assigned wholesale; mark initialized so the first
                    # ``update`` does not re-init and zero the restored moments.
                    optimizer._initialized = True
                    mx.eval(optimizer.state)
                if resume.rng_state is not None:
                    # mlx-lm ships no stub for the ``mx.random.state`` global, so
                    # pyright cannot see the attribute; restoring it continues the
                    # dropout/sampling RNG 1:1 across the resume.
                    mx.random.state = resume.rng_state  # type: ignore[attr-defined]
                    mx.eval(*resume.rng_state)
                if plateau is not None and resume.plateau is not None:
                    plateau.load(*resume.plateau)
            # Cadence in micro-steps (mlx-lm counts micro-steps): an optimizer
            # step is ``grad_accumulation_steps`` micro-steps, an epoch is
            # ``micro_per_epoch``. ``None`` disables trajectory saving (no config).
            save_every = (
                None
                if ckpt is None
                else ckpt.each
                * (training.grad_accumulation_steps if ckpt.on == "step" else micro_per_epoch)
            )
            stopping = EarlyStopping(
                evaluation.patience if evaluation is not None else None,
                best=resume.best_loss if resume is not None else math.inf,
                streak=resume.no_improve if resume is not None else 0,
            )
            validation = (
                _Validation(
                    dataset=val_dataset,
                    every=evaluation.each
                    * (
                        training.grad_accumulation_steps
                        if evaluation.on == "step"
                        else micro_per_epoch
                    ),
                    batch_size=training.batch_size,
                    max_seq_length=seq_len,
                    loss=loss_fn,
                    iterate_batches=iterate_fn,
                )
                if evaluation is not None and val_dataset is not None
                else None
            )
            report = _loss_report_callback(
                callback,
                lm,
                optimizer,
                iters,
                stopping,
                validation,
                ckpt_dir=ckpt_dir,
                save_every=save_every,
                keep=ckpt.keep if ckpt is not None else None,
                plateau=plateau,
                step_offset=step_offset,
                resumable=resumable,
                save_optimizer_state=training.save_optimizer_state,
                init_best=resume.best if resume is not None else None,
            )
            remaining = iters - step_offset
            if remaining > 0:
                args = TrainingArgs(
                    batch_size=training.batch_size,
                    iters=remaining,
                    grad_accumulation_steps=training.grad_accumulation_steps,
                    grad_checkpoint=training.gradient_checkpointing,
                    max_seq_length=seq_len,
                    adapter_file=str(Path(ckpt_dir) / "adapters.safetensors"),
                    steps_per_report=1,
                    # mlx-lm's own saving and evaluation are disabled (no
                    # ``val_dataset``); the report callback writes numbered
                    # checkpoint-<step>.safetensors snapshots and evaluates itself.
                    steps_per_save=remaining + 1,
                )
                train_kwargs: dict[str, Any] = {
                    "model": lm,
                    "optimizer": optimizer,
                    "train_dataset": dataset,
                    "args": args,
                    "iterate_batches": iterate_fn,
                    "training_callback": report,
                }
                if loss_fn is not None:
                    train_kwargs["loss"] = loss_fn

                with (
                    _neftune(lm, training.neftune_noise_alpha),
                    _silenced(),
                    contextlib.suppress(_StopTraining),
                ):
                    mlx_train(**train_kwargs)

            if report.best is not None:
                lm.update(tree_unflatten(report.best))
                mx.eval(lm.parameters())
        lm.eval()

    def _constraint_bias(self, constraint: ValueConstraint) -> Any:
        """Additive logits bias: ``0`` for the tokens ``constraint`` allows (plus
        EOS), ``-inf`` for the rest -- the same mask :meth:`HFBackend.generate`
        applies (invariant #3)."""
        bias = self._constraint_biases.get(constraint)
        if bias is None:
            import mlx.core as mx

            tok = self._tokenizer
            # TokenizerWrapper has no __len__; the full vocab (added tokens
            # included) is recovered from the id mapping instead.
            vocab = max(tok.get_vocab().values()) + 1
            allowed = [constraint.allows(tok.decode([i])) for i in range(vocab)]
            if tok.eos_token_id is not None:
                allowed[tok.eos_token_id] = True
            bias = mx.where(mx.array(allowed), 0.0, float("-inf"))
            self._constraint_biases[constraint] = bias
        return bias

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
        import mlx.core as mx
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        self._model.eval()
        if random_state is not None:
            mx.random.seed(random_state)
        temp = generation.temperature if generation.temperature > 0 else 0.0
        sampler = make_sampler(temp=temp, top_p=generation.top_p, top_k=generation.top_k)
        processors = (
            make_logits_processors(repetition_penalty=generation.repetition_penalty)
            if generation.repetition_penalty is not None
            else []
        )
        if constraint is not None:
            bias = self._constraint_bias(constraint)

            def _mask_logits(_tokens: Any, logits: Any) -> Any:
                n = bias.shape[0]
                # Model logits may be padded past the tokenizer vocab; those ids
                # decode to nothing meaningful, so they are always masked.
                if logits.shape[-1] > n:
                    pad = mx.full((logits.shape[-1] - n,), float("-inf"))
                    return logits + mx.concatenate([bias, pad])
                return logits + bias

            processors = [*processors, _mask_logits]
        prompt_ids = self._truncated_prompt_ids(prompts)
        response = batch_generate(
            self._model,
            self._tokenizer,
            prompts=prompt_ids,
            max_tokens=resolve_max_new_tokens(generation, self._max_seq_length),
            sampler=sampler,
            logits_processors=processors or None,
        )
        return list(response.texts)

    def _truncated_prompt_ids(self, prompts: Sequence[str]) -> list[list[int]]:
        """Encode each prompt, keeping its last ``_max_seq_length`` tokens.

        The cap is the training sequence length (one whole serialized row), *not*
        ``_max_seq_length - max_new_tokens``: an estimator's prompt is itself up to
        a full row, so reserving room for the continuation would left-truncate it
        mid-row and strip the conditioning columns the model must read. The
        generated tokens extend past the prompt into the model's own context
        window instead.
        """
        return [self._tokenizer.encode(p.rstrip())[-self._max_seq_length :] for p in prompts]

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
        one row per group, primed into a ``BatchKVCache``), then every pair's
        remaining tokens are scored in one right-padded batch on top of its
        group's replicated cache -- the shared prompt is not re-forwarded per
        candidate. The continuation boundary per pair is the end of the longest
        token prefix shared with its prompt (so a BPE merge at the prompt's
        trailing space is scored rather than dropped) -- identical semantics to
        :meth:`HFBackend.score`, so the classifier ranks verbalizers the same way
        on both backends. ``reduce`` collapses each pair's per-token
        log-likelihoods to the ``"mean"`` (default) or the ``"sum"``.
        """
        if not prompts:
            return []
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.cache import BatchKVCache

        self._model.eval()
        groups = prompt_groups(prompts)
        fulls: list[list[int]] = []
        starts: list[int] = []
        for a, b in groups:
            prompt_ids = self._tokenizer.encode(prompts[a])  # once per run, not per pair
            for i in range(a, b):
                full = self._tokenizer.encode(prompts[i] + continuations[i])
                full = full[: self._max_seq_length]
                starts.append(max(common_token_prefix(prompt_ids, full), 1))
                fulls.append(full)

        # Every full in a group agrees with its shared prompt on the first ``start``
        # tokens, so ``min(starts) - 1`` is a token prefix common to the whole group;
        # the ``- 1`` keeps the boundary token in the suffix, so its logit comes from
        # the suffix pass and the prefill's logits are never needed.
        prefix_lens = [max(min(starts[a:b]) - 1, 0) for a, b in groups]
        pair_prefix = [0] * len(prompts)
        for g, (a, b) in enumerate(groups):
            pair_prefix[a:b] = [prefix_lens[g]] * (b - a)
        suffixes = [full[c : len(full) - 1] for full, c in zip(fulls, pair_prefix, strict=True)]
        s_max = max(len(s) for s in suffixes)
        if s_max == 0:  # every full is a single token: nothing is scorable
            return [float("-inf")] * len(prompts)

        cache = None
        if max(prefix_lens) > 0:
            c_max = max(prefix_lens)
            prefix = np.zeros((len(groups), c_max), dtype=np.int32)
            for g, (a, _) in enumerate(groups):
                if prefix_lens[g]:
                    prefix[g, c_max - prefix_lens[g] :] = fulls[a][: prefix_lens[g]]
            # One BatchKVCache per layer: it carries per-row left padding, and the
            # model builds position ids and the attention mask from it, so rows
            # with different prefix lengths batch together.
            cache = [BatchKVCache([c_max - c for c in prefix_lens]) for _ in self._model.layers]
            self._model(mx.array(prefix), cache=cache)
            members = mx.array([g for g, (a, b) in enumerate(groups) for _ in range(a, b)])
            for layer in cache:
                layer.filter(members)  # replicate each group's row, one per pair

        # Right padding with 0: causal attention keeps real tokens unaffected by
        # trailing pad positions, and per-pair slicing never reads them.
        arr = np.zeros((len(suffixes), s_max), dtype=np.int32)
        targets = np.zeros((len(suffixes), s_max), dtype=np.int32)
        for i, (suffix, full) in enumerate(zip(suffixes, fulls, strict=True)):
            arr[i, : len(suffix)] = suffix
            targets[i, : len(suffix)] = full[pair_prefix[i] + 1 :]
        logits = self._model(mx.array(arr), cache=cache).astype(mx.float32)
        logprobs = nn.log_softmax(logits, axis=-1)
        token_logprobs = mx.take_along_axis(
            logprobs, mx.array(targets)[..., None], axis=-1
        ).squeeze(-1)
        scores: list[float] = []
        for i, full in enumerate(fulls):
            lo, hi = starts[i] - 1 - pair_prefix[i], len(full) - 1 - pair_prefix[i]
            cont = token_logprobs[i, lo:hi]
            if len(full) < 2 or cont.size == 0:
                scores.append(float("-inf"))
            else:
                pooled = mx.sum(cont) if reduce == "sum" else mx.mean(cont)
                scores.append(float(pooled.item()))
        return scores


class _MLXTextDataset:
    """Map-style dataset over per-epoch (re)serialized rows.

    ``__getitem__`` returns ``(token_ids, prompt_offset)``: ``prompt_offset`` is
    the number of leading tokens the loss down-weights (``0`` unless
    ``target_loss_weight`` marks the row's context), found as the longest
    common token prefix between the example's prompt and full text. With
    ``digit_tokens`` set
    (``numeric_loss_weight``), items grow to ``(ids, offset, digit_scale,
    digit_variant, number_id, numeric_targets, numeric_weights)``
    -- the arrays of :func:`~sklm.backend.numeric_token_arrays`.
    ``reshuffle`` re-draws the buffer from ``epoch_texts`` for the next epoch
    index so the column-order permutation is redrawn at each epoch boundary;
    tokenization is lazy and memoized per epoch.
    """

    def __init__(
        self,
        epoch_texts: Callable[[int], list[TrainingExample]],
        tokenizer: Any,
        max_seq_length: int,
        digit_tokens: DigitTokens | None = None,
    ) -> None:
        self._epoch_texts = epoch_texts
        self._tok = tokenizer
        self._max = max_seq_length
        self._eos = tokenizer.eos_token_id
        self._digit_tokens = digit_tokens
        self._epoch = 0
        self._buffer = epoch_texts(0)
        self._cache: dict[int, tuple[Any, ...]] = {}

    def reshuffle(self) -> None:
        self._epoch += 1
        self._buffer = self._epoch_texts(self._epoch)
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        cached = self._cache.get(idx)
        if cached is not None:
            return cached
        ex = self._buffer[idx]
        ids = list(self._tok.encode(ex.text)[: self._max - 1])
        if self._eos is not None:
            ids.append(self._eos)
        offset = 0
        if ex.prompt:
            prompt_ids = self._tok.encode(ex.prompt)
            # Always supervise >=1 token.
            offset = min(common_token_prefix(prompt_ids, ids), len(ids) - 1)
        result: tuple[Any, ...] = (ids, offset)
        if self._digit_tokens is not None:
            arrays = numeric_token_arrays(ex, ids, self._tok.encode, self._digit_tokens.variant_of)
            result = (ids, offset, *arrays)
        self._cache[idx] = result
        return result


def _save_checkpoint(
    ckpt_dir: str, step: int, weights: Any, sidecar: dict[str, Any] | None, keep: int | None
) -> None:
    """Write a ``checkpoint-<step>`` snapshot (and its resume sidecar) and prune.

    ``weights`` is a ``tree_flatten`` list of ``(path, array)`` written to
    ``checkpoint-<step>.safetensors``. ``sidecar`` (when resuming is possible)
    holds the optimizer state, RNG and trackers needed to continue training; it
    is written to ``state-<step>.safetensors``. Both numbered files are pruned in
    lockstep (oldest first), keeping the ``keep`` most recent; ``best.safetensors``
    is named apart, so it survives. ``keep`` of ``None`` retains every snapshot.
    """
    import mlx.core as mx

    d = Path(ckpt_dir)
    mx.save_safetensors(str(d / f"checkpoint-{step}.safetensors"), dict(weights))
    if sidecar is not None:
        mx.save_safetensors(str(d / f"state-{step}.safetensors"), sidecar)
    if keep is not None:
        steps = sorted(int(p.stem.split("-")[1]) for p in d.glob("checkpoint-*.safetensors"))
        for old in steps[:-keep]:
            (d / f"checkpoint-{old}.safetensors").unlink(missing_ok=True)
            (d / f"state-{old}.safetensors").unlink(missing_ok=True)


def _save_best(ckpt_dir: str, snapshot: Any, loss: float) -> None:
    """Persist the best (lowest-val-loss) weights to ``best.safetensors``.

    Written the moment the best improves (not only at the end) so an interrupted
    run can still restore it. ``snapshot`` is a ``tree_flatten`` list; the loss is
    embedded under ``meta.best_loss`` so resume recovers the bar to beat.
    """
    import mlx.core as mx

    payload: dict[str, Any] = {f"best.{k}": v for k, v in snapshot}
    payload["meta.best_loss"] = mx.array(loss)
    mx.save_safetensors(str(Path(ckpt_dir) / "best.safetensors"), payload)


@dataclass
class _Resume:
    """State restored from a checkpoint directory to continue training 1:1.

    ``opt_state``/``rng_state``/``plateau`` are ``None`` when the directory holds
    only legacy weight snapshots (no ``state-*`` sidecar): training then warm-starts
    from the weights and reruns the full budget, the pre-resume behaviour.
    """

    resumed_it: int
    opt_state: list[tuple[str, Any]] | None
    rng_state: list[Any] | None
    no_improve: int
    plateau: tuple[float, int, int] | None
    best: list[tuple[str, Any]] | None
    best_loss: float


def _resume_state(lm: Any, ckpt_dir: str) -> _Resume | None:
    """Restore the most recent checkpoint into ``lm`` and return the resume state.

    Loads the latest ``checkpoint-<step>`` weights into ``lm`` and reads the
    matching ``state-<step>`` sidecar (optimizer state, RNG, plateau/early-stop
    trackers) plus ``best.safetensors``. Returns ``None`` when ``ckpt_dir`` holds
    no snapshot at all (a fresh run). When a weight snapshot exists without its
    sidecar (a directory written before resume tracked optimizer state), the
    optimizer fields come back ``None`` and the caller warm-starts from the
    weights.
    """
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    d = Path(ckpt_dir)
    # mx.load of a .safetensors returns a {name: array} dict; the mlx stub types
    # it as the broader array|tuple union, so narrow it where it feeds tree ops.
    best: list[tuple[str, Any]] | None = None
    best_loss = math.inf
    best_file = d / "best.safetensors"
    if best_file.exists():
        bd: dict[str, Any] = mx.load(str(best_file))  # type: ignore[assignment]
        best = [(k.removeprefix("best."), v) for k, v in bd.items() if k.startswith("best.")]
        if "meta.best_loss" in bd:
            best_loss = float(bd["meta.best_loss"].item())

    ckpts = sorted(d.glob("checkpoint-*.safetensors"), key=lambda p: int(p.stem.split("-")[1]))
    if not ckpts:
        return _Resume(0, None, None, 0, None, best, best_loss) if best is not None else None

    weights: dict[str, Any] = mx.load(str(ckpts[-1]))  # type: ignore[assignment]
    lm.update(tree_unflatten(list(weights.items())))
    mx.eval(lm.parameters())
    it = int(ckpts[-1].stem.split("-")[1])

    sidecar_file = d / f"state-{it}.safetensors"
    if not sidecar_file.exists():
        return _Resume(0, None, None, 0, None, best, best_loss)

    s: dict[str, Any] = mx.load(str(sidecar_file))  # type: ignore[assignment]
    opt_state = [(k.removeprefix("opt."), v) for k, v in s.items() if k.startswith("opt.")]
    rng_state = [s[f"rng.{i}"] for i in range(sum(1 for k in s if k.startswith("rng.")))]
    no_improve = int(s["meta.no_improve"].item())
    plateau = (
        (
            float(s["meta.plateau_best"].item()),
            int(s["meta.plateau_num_bad"].item()),
            int(s["meta.plateau_cooldown"].item()),
        )
        if "meta.plateau_best" in s
        else None
    )
    return _Resume(it, opt_state, rng_state, no_improve, plateau, best, best_loss)


class _PlateauReducer:
    """Reduce-on-plateau controller for the MLX backend.

    Reimplements ``torch.optim.lr_scheduler.ReduceLROnPlateau`` (minimum mode,
    relative threshold, ``eps=1e-8``) so the MLX backend matches the HF backend,
    which uses torch's scheduler directly. Fed the validation loss at each
    evaluation; returns the learning rate to use next.

    Parameters
    ----------
    config : PlateauLR
        Schedule parameters: factor, patience, threshold, floor, cooldown.

    Notes
    -----
    A reduction fires once the loss has failed to improve for ``patience + 1``
    consecutive evaluations (the count uses strict ``>``), matching torch.
    """

    def __init__(self, config: PlateauLR) -> None:
        self._cfg = config
        self._best = math.inf
        self._num_bad = 0
        self._cooldown = 0

    def step(self, metric: float, lr: float) -> float:
        """Return the learning rate for the next step given ``metric``.

        Parameters
        ----------
        metric : float
            Latest validation loss.
        lr : float
            Current learning rate.

        Returns
        -------
        float
            The (possibly reduced) learning rate; equal to ``lr`` unless a
            plateau just triggered a reduction.
        """
        cfg = self._cfg
        if metric < self._best * (1 - cfg.threshold):
            self._best = metric
            self._num_bad = 0
        else:
            self._num_bad += 1
        if self._cooldown > 0:
            self._cooldown -= 1
            self._num_bad = 0
        new_lr = lr
        if self._num_bad > cfg.patience:
            reduced = max(lr * cfg.factor, cfg.floor)
            if lr - reduced > 1e-8:
                new_lr = reduced
            self._cooldown = cfg.cooldown
            self._num_bad = 0
        return new_lr

    def export(self) -> tuple[float, int, int]:
        """Snapshot the controller state for checkpointing: ``(best, num_bad, cooldown)``."""
        return self._best, self._num_bad, self._cooldown

    def load(self, best: float, num_bad: int, cooldown: int) -> None:
        """Restore the controller state from an :meth:`export` snapshot."""
        self._best, self._num_bad, self._cooldown = best, num_bad, cooldown


@dataclass(frozen=True)
class _Validation:
    """The hold-out and how to evaluate it: every ``every`` micro-steps, with
    mlx-lm's ``evaluate`` over the whole ``dataset`` under the training loss."""

    dataset: Any
    every: int
    batch_size: int
    max_seq_length: int
    loss: Callable[..., Any] | None
    iterate_batches: Callable[..., Iterator[Any]]


def _loss_report_callback(
    callback: Callback,
    lm: Any,
    optimizer: Any,
    iters: int,
    stopping: EarlyStopping,
    validation: _Validation | None,
    *,
    ckpt_dir: str,
    save_every: int | None,
    keep: int | None,
    plateau: _PlateauReducer | None = None,
    step_offset: int = 0,
    resumable: bool = False,
    save_optimizer_state: bool = True,
    init_best: Any = None,
) -> Any:
    """Build an mlx-lm ``TrainingCallback`` that forwards train/val loss, the
    gradient norm and the current device memory to ``callback``.

    The gradient norm stays ``None`` when clipping is disabled
    (``max_grad_norm`` unset), since nothing computes it then.

    ``step_offset`` is added to every mlx-lm iteration so a resumed run reports
    and names checkpoints on the original absolute step axis. ``stopping`` (its
    best loss and streak seeded from the resumed checkpoint) and ``init_best``
    (the resumed best weights) let early stopping and best restoration survive
    a resume.

    Evaluation is run here rather than by mlx-lm's loop, which validates before
    a step (and always at its first and last iteration): after every
    ``validation.every``-th micro-step the hold-out is scored, so the reports
    land on the same absolute steps as the HF backend's. Each loss is fed to
    ``stopping``: an improvement snapshots the weights into ``.best`` for the
    caller to restore (and, when ``resumable``, persists them to
    ``best.safetensors`` on the spot); exhausted patience raises
    :class:`_StopTraining`.

    When ``save_every`` is set, every that-many micro-steps a numbered checkpoint
    is written under ``ckpt_dir`` and pruned to the ``keep`` most recent; when
    ``resumable`` (and ``save_optimizer_state``) it carries a ``state-<step>``
    sidecar (optimizer state, RNG, trackers) so the next ``fit`` continues 1:1.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.trainer import default_loss, evaluate

    class _LossReport(TrainingCallback):
        def __init__(self) -> None:
            # tree_flatten's result (a list of (path, array)); typed Any
            # because mlx-lm ships no stubs for it.
            self.best: Any = init_best

        def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
            callback.on_memory(int(mx.get_active_memory()))
            norm = optimizer.state.get("last_grad_norm")
            step = int(train_info["iteration"]) + step_offset
            callback.on_train_report(
                step=step,
                total_steps=iters,
                loss=float(train_info["train_loss"]),
                epoch=None,
                learning_rate=train_info.get("learning_rate"),
                grad_norm=float(norm.item()) if norm is not None else None,
            )
            if validation is not None and step % validation.every == 0:
                self._evaluate(step)
            if save_every is not None and step % save_every == 0:
                weights = tree_flatten(lm.trainable_parameters())
                sidecar = self._sidecar(step) if resumable and save_optimizer_state else None
                mx.eval([v for _, v in weights])
                _save_checkpoint(ckpt_dir, step, weights, sidecar, keep)

        def _sidecar(self, step: int) -> dict[str, Any]:
            """Optimizer state + RNG + trackers, keyed for one safetensors file."""
            opt_state = tree_flatten(optimizer.state)
            mx.eval([v for _, v in opt_state])
            sidecar: dict[str, Any] = {f"opt.{k}": v for k, v in opt_state}
            # No stub for the ``mx.random.state`` global; type it for enumerate.
            rng: list[Any] = mx.random.state  # type: ignore[attr-defined]
            sidecar |= {f"rng.{i}": a for i, a in enumerate(rng)}
            sidecar["meta.no_improve"] = mx.array(stopping.streak)
            if plateau is not None:
                best, num_bad, cooldown = plateau.export()
                sidecar["meta.plateau_best"] = mx.array(best)
                sidecar["meta.plateau_num_bad"] = mx.array(num_bad)
                sidecar["meta.plateau_cooldown"] = mx.array(cooldown)
            return sidecar

        def _evaluate(self, step: int) -> None:
            assert validation is not None
            loss = float(
                evaluate(
                    model=lm,
                    dataset=validation.dataset,
                    batch_size=validation.batch_size,
                    num_batches=-1,
                    max_seq_length=validation.max_seq_length,
                    loss=validation.loss if validation.loss is not None else default_loss,
                    iterate_batches=validation.iterate_batches,
                )
            )
            lm.train()
            callback.on_eval_report(step=step, loss=loss, epoch=None)
            verdict = stopping.observe(loss)
            if plateau is not None:
                optimizer.learning_rate = plateau.step(loss, float(optimizer.learning_rate.item()))
            match verdict:
                case "improved":
                    # mlx arrays are immutable, so snapshotting the current leaves
                    # is enough -- the optimizer rebinds new arrays on each step.
                    snapshot = tree_flatten(lm.trainable_parameters())
                    mx.eval([v for _, v in snapshot])
                    self.best = snapshot
                    if resumable:
                        _save_best(ckpt_dir, snapshot, loss)
                case "exhausted":
                    raise _StopTraining
                case "no_improvement":
                    pass

    return _LossReport()


def _make_iterate_batches(
    random_state: int | None, numeric: bool = False
) -> Callable[..., Iterator[Any]]:
    """Build the ``iterate_batches`` ``train`` will drive.

    Mirrors mlx-lm's batching (permute, pad to a multiple of 32, truncate) but
    calls ``dataset.reshuffle()`` at each wrap-around so a new column-order
    permutation is drawn per epoch. Epoch 0 uses the dataset's initial buffer;
    the first wrap fires epoch 1. Padding with 0 is harmless: the loss masks
    padded positions via ``lengths``. With ``numeric`` (the auxiliary numeric
    loss), each yielded batch grows the dataset's per-token digit arrays and
    per-number targets -- ``train`` unpacks the tuple into the custom loss.
    """

    def iterate_batches(
        dataset: Any,
        batch_size: int,
        max_seq_length: int,
        loop: bool = False,
        seed: int | None = None,
        comm_group: Any = None,
    ) -> Iterator[Any]:
        import mlx.core as mx

        rng = np.random.default_rng(random_state)
        first = True
        while True:
            if not first:
                dataset.reshuffle()
            first = False
            order = rng.permutation(len(dataset))
            batch_idx = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
            for indices in batch_idx:
                batch = [dataset[int(j)] for j in indices]
                ids_batch = [b[0] for b in batch]
                offsets = [b[1] for b in batch]
                lengths = [len(x) for x in ids_batch]
                pad_to = 32
                max_len = min(1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to), max_seq_length)
                # The last batch of an epoch may be short (len(indices) < batch_size);
                # unlike mlx-lm's default batcher, no row is dropped -- the loss
                # masks padding via lengths, so a short batch is safe.
                arr = np.zeros((len(indices), max_len), dtype=np.int32)
                trunc = []
                for j in range(len(indices)):
                    t = min(lengths[j], max_seq_length)
                    arr[j, :t] = ids_batch[j][:t]
                    trunc.append(t)
                pair = mx.array(arr), mx.array(list(zip(offsets, trunc, strict=True)))
                if not numeric:
                    yield pair
                    continue
                n = len(indices)
                slots = max(len(b[5]) for b in batch) or 1
                scale = np.zeros((n, max_len), dtype=np.float32)
                variant = np.full((n, max_len), -1, dtype=np.int32)
                number_id = np.full((n, max_len), -1, dtype=np.int32)
                targets = np.zeros((n, slots), dtype=np.float32)
                weights = np.zeros((n, slots), dtype=np.float32)
                for j, b in enumerate(batch):
                    t = trunc[j]
                    scale[j, :t] = b[2][:t]
                    variant[j, :t] = b[3][:t]
                    number_id[j, :t] = b[4][:t]
                    targets[j, : len(b[5])] = b[5]
                    weights[j, : len(b[6])] = b[6]
                yield (
                    *pair,
                    mx.array(scale),
                    mx.array(variant),
                    mx.array(number_id),
                    mx.array(targets),
                    mx.array(weights),
                )
            if not loop:
                break

    return iterate_batches


def _build_optimizer(training: TrainingConfig, steps: int, lr: float) -> Any:
    """Build the optimizer with its LR schedule sized over ``steps`` optimizer
    steps (schedules advance once per ``optimizer.update``, not per micro-step)."""
    import mlx.core as mx
    import mlx.optimizers as opt

    sched = training.lr_scheduler

    def main_schedule(n: int) -> Any:
        match sched:
            case ConstantLR():
                return opt.linear_schedule(lr, lr, max(1, n))
            case LinearLR():
                return opt.linear_schedule(lr, sched.floor, max(1, n))
            case CosineLR():
                return opt.cosine_decay(lr, max(1, n), end=sched.floor)
            case _:
                raise ValueError(f"unknown lr_scheduler {type(sched).__name__}")

    match sched:
        case PlateauLR():
            # Scalar LR (no schedule); _PlateauReducer mutates optimizer.learning_rate on plateau.
            schedule: Any = lr
        case ConstantLR() | LinearLR() | CosineLR():
            warmup_n = (
                min(round(steps * sched.warmup_ratio), steps) if sched.warmup_ratio > 0 else 0
            )
            if warmup_n > 0:
                schedule = opt.join_schedules(
                    [opt.linear_schedule(0.0, lr, warmup_n), main_schedule(steps - warmup_n)],
                    [warmup_n],
                )
            elif isinstance(sched, ConstantLR):
                schedule = lr
            else:
                schedule = main_schedule(steps)
        case _:
            raise ValueError(f"unknown lr_scheduler {type(sched).__name__}")

    match training.optimizer:
        case "adamw" | "adamw_8bit" | "paged_adamw_8bit":
            if training.optimizer != "adamw":
                warnings.warn(
                    f"optimizer={training.optimizer!r} needs bitsandbytes (CUDA); "
                    "using plain AdamW on MLX.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            optimizer: Any = opt.AdamW(learning_rate=schedule, weight_decay=training.weight_decay)
        case "adafactor":
            # With an explicit LR, disable Adafactor's relative-step scaling so
            # it honors ``learning_rate`` (matching the HF backend).
            optimizer = opt.Adafactor(
                learning_rate=schedule,
                weight_decay=training.weight_decay,
                relative_step=False,
                scale_parameter=False,
            )
        case "lion":
            optimizer = opt.Lion(learning_rate=schedule, weight_decay=training.weight_decay)
        case _:
            raise ValueError(f"unknown optimizer {training.optimizer!r}")

    if training.max_grad_norm is not None and training.max_grad_norm > 0:
        max_norm = float(training.max_grad_norm)

        # ``train``'s compiled step never clips; the optimizer's ``update`` is
        # the only hook, so clip there. The norm must land in ``optimizer.state``
        # (seeded below): only arrays inside the captured ``state`` cross mlx's
        # ``compile`` boundary -- a plain attribute is a traced array that dies
        # outside it. This mirrors how the optimizer's own ``step`` survives.
        class _ClippedOptimizer(type(optimizer)):  # type: ignore[misc]  # dynamic base class
            def update(self, model: Any, gradients: Any) -> None:
                gradients, self.state["last_grad_norm"] = opt.clip_grad_norm(gradients, max_norm)
                super().update(model, gradients)

        optimizer.__class__ = _ClippedOptimizer
        optimizer.state["last_grad_norm"] = mx.array(0.0)

    return optimizer


def _make_loss(training: TrainingConfig, digit_tokens: DigitTokens | None) -> Callable[..., Any]:
    """Cross-entropy with label smoothing, ``target_loss_weight`` weighting and
    the ``numeric_loss_weight`` auxiliary term.

    Without those knobs the masking matches ``mlx_lm.tuner.trainer.default_loss``
    (offset 0 -> loss on every token). Under ``target_loss_weight`` the binary
    prompt mask becomes ``1 - alpha`` / ``alpha`` weights normalized by their
    sum; under ``numeric_loss_weight`` a softmax restricted to the digit-token
    candidates reconstructs the expected number per ``number_id`` slot and its
    mean squared error against the targets is added, each slot's error scaled
    by its column's reciprocal standard deviation (kept behaviorally identical
    to ``hf_backend._make_weighted_trainer`` -- invariant 3).
    """
    import mlx.core as mx
    import mlx.nn as nn

    eps = training.label_smoothing
    alpha = training.target_loss_weight
    numeric_weight = training.numeric_loss_weight
    candidates = digit_tokens.candidates if digit_tokens is not None else ((), ())

    def loss(model: Any, batch: Any, lengths: Any, *numeric: Any) -> Any:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        in_row = steps <= lengths[:, 1:]
        if alpha is None:
            weights = mx.logical_and(steps >= lengths[:, 0:1], in_row).astype(mx.float32)
        else:
            weights = mx.where(steps >= lengths[:, 0:1], alpha, 1.0 - alpha) * in_row
        ce = (
            nn.losses.cross_entropy(logits, targets, label_smoothing=eps, reduction="none")
            * weights
        )
        wsum = weights.sum()
        total = ce.astype(mx.float32).sum() / wsum
        if numeric:
            # Arrays align with token positions, so their step-1.. tail aligns
            # with ``targets`` (position t is predicted by the logits at t - 1).
            scale, variant, number_id, numeric_targets, numeric_weights = numeric
            s = scale[:, 1:]
            v = variant[:, 1:]
            nid = number_id[:, 1:]
            expected = mx.zeros(s.shape)
            for var, cand in enumerate(candidates):
                if not cand:
                    continue
                p = mx.softmax(mx.take(logits, mx.array(cand), axis=-1).astype(mx.float32), axis=-1)
                digits = mx.arange(10).astype(mx.float32)
                expected = mx.where(v == var, (p * digits).sum(-1), expected)
            bsz, slots = numeric_targets.shape
            base = mx.arange(bsz)[:, None] * slots
            # Non-digit positions land in a dump slot past the real ones.
            flat = mx.where(nid >= 0, nid + base, bsz * slots).reshape(-1)
            sums = mx.zeros(bsz * slots + 1).at[flat].add((expected * s).reshape(-1))
            counts = (
                mx.zeros(bsz * slots + 1).at[flat].add((nid >= 0).reshape(-1).astype(mx.float32))
            )
            valid = counts[:-1] > 0
            # inv_scale puts the error in standard deviations of its column, so
            # columns of unlike magnitude weigh alike (invariant 3: HF computes
            # the same product).
            err = ((sums[:-1] - numeric_targets.reshape(-1)) * numeric_weights.reshape(-1)) ** 2
            aux = (err * valid).sum() / mx.maximum(valid.sum(), 1)
            total = total + numeric_weight * aux
        return total, wsum

    return loss


def _resolve_mlx_method(quantization: QuantizationConfig) -> str:
    """Validate ``quantization`` for the MLX backend (native quantizer only),
    raising the same shape of error as the HF backend's resolver."""
    method = "mlx" if quantization.method == "auto" else quantization.method
    supported = _MLX_METHODS.get(method)
    if supported is None:
        raise ValueError(
            f"the MLX backend quantizes only via method='mlx' (or 'auto'); "
            f"got method={quantization.method!r}"
        )
    if quantization.bits not in supported:
        raise ValueError(
            f"MLX quantization supports bits {sorted(supported)}; got bits={quantization.bits}"
        )
    return method


_LOCAL_REVISION = "0" * 40


def _convert_quantized(
    model_name: str, q_bits: int, precision: str, group_size: int | None = None
) -> str:
    """Convert ``model_name`` to MLX's native ``q_bits``-bit format, cached in the
    Hugging Face hub cache as the local-only repo ``sklm/<slug>-<bits>bit-<precision>``
    (visible to ``hf cache scan`` / ``hf cache delete``; the repo id resolves nowhere
    online, which is fine because the snapshot *path* is returned and loaded directly).
    Requires an mlx-convertible HF repo (safetensors) or a pre-converted
    ``mlx-community`` model.

    The conversion writes into a temporary sibling directory and is renamed into
    the snapshot path only once complete, so an interrupted convert never leaves a
    half-written cache behind (``mlx_lm`` writes ``config.json`` before the
    tokenizer files, so the files' mere presence proves nothing)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    from mlx_lm import convert

    slug = model_name.replace("/", "--")
    group = f"-g{group_size}" if group_size is not None else ""
    repo_dir = Path(HF_HUB_CACHE) / f"models--sklm--{slug}-{q_bits}bit{group}-{precision}"
    snapshot = repo_dir / "snapshots" / _LOCAL_REVISION
    if (snapshot / "config.json").exists():
        return str(snapshot)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        # A partial directory from a convert interrupted before this rename
        # scheme existed; rebuild it.
        shutil.rmtree(snapshot)
    convert_kwargs: dict[str, Any] = {
        "hf_path": model_name,
        "quantize": True,
        "q_bits": q_bits,
        "dtype": _DTYPE_NAMES[precision],
    }
    if group_size is not None:
        convert_kwargs["q_group_size"] = group_size
    staging = Path(tempfile.mkdtemp(dir=repo_dir, prefix="convert.tmp"))
    try:
        # convert refuses to write into an existing directory, hence the child.
        out = staging / "model"
        convert(mlx_path=str(out), **convert_kwargs)
        os.replace(out, snapshot)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    refs = repo_dir / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(_LOCAL_REVISION)
    return str(snapshot)


def _load_tokenizer(tokenizer_id: str, trust_remote_code: bool) -> Any:
    from mlx_lm.utils import load_tokenizer

    path = Path(tokenizer_id)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                tokenizer_id,
                allow_patterns=[
                    "*.json",
                    "*.py",
                    "tokenizer.model",
                    "*.tiktoken",
                    "tiktoken.model",
                    "*.txt",
                    "*.jinja",
                ],
            )
        )
    extra = {"trust_remote_code": True} if trust_remote_code else None
    return load_tokenizer(path, tokenizer_config_extra=extra)


def _apply_lora(model: Any, config: LoRAConfig) -> None:
    """Apply LoRA adapters via mlx-lm's ``linear_to_lora_layers``.

    mlx-lm's config dict has no ``alpha``/``rslora``/pattern fields, so the
    scale is precomputed here (``alpha / sqrt(rank)`` when ``rslora`` else
    ``alpha / rank``). Per-module rank/alpha patterns are expressed by grouping
    keys and calling ``linear_to_lora_layers`` once per group.
    """
    from mlx_lm.tuner.utils import linear_to_lora_layers

    num_layers = len(model.layers) if hasattr(model, "layers") else -1
    # linear_to_lora_layers only un-freezes the LoRA modules it creates and
    # expects the base to be frozen; without this, "LoRA" becomes a full
    # fine-tune.
    model.freeze()

    def scale(rank: int, alpha: int) -> float:
        return alpha / (rank**0.5 if config.rslora else rank)

    if not config.rank_pattern and not config.alpha_pattern:
        lora_cfg: dict[str, Any] = {
            "rank": config.rank,
            "dropout": config.dropout,
            "scale": scale(config.rank, config.alpha),
        }
        if isinstance(config.target_modules, list):
            lora_cfg["keys"] = config.target_modules
        linear_to_lora_layers(model, num_layers, lora_cfg, use_dora=config.dora)
        return

    groups: dict[tuple[int, int], list[str]] = {}
    for key in _discover_lora_keys(model, config.target_modules):
        suffix = key.rsplit(".", 1)[-1]
        rank = (config.rank_pattern or {}).get(suffix, config.rank)
        alpha = (config.alpha_pattern or {}).get(suffix, config.alpha)
        groups.setdefault((rank, alpha), []).append(key)
    for (rank, alpha), keys in groups.items():
        linear_to_lora_layers(
            model,
            num_layers,
            {"rank": rank, "dropout": config.dropout, "scale": scale(rank, alpha), "keys": keys},
            use_dora=config.dora,
        )


def _discover_lora_keys(model: Any, target_modules: str | list[str] | None) -> list[str]:
    """The module paths mlx-lm's ``linear_to_lora_layers`` would adapt by default
    (used only to expand rank/alpha patterns)."""
    if isinstance(target_modules, list):
        return list(target_modules)
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    types = (
        nn.Linear,
        nn.QuantizedLinear,
        SwitchLinear,
        QuantizedSwitchLinear,
        nn.Embedding,
        nn.QuantizedEmbedding,
    )
    keys: set[str] = set()

    def collect(path: str, module: Any) -> None:
        if hasattr(module, "to_lora") or isinstance(module, types):
            keys.add(path)

    blocks = getattr(model, "layers", [model])
    for block in blocks:
        block.apply_to_modules(collect)
    return sorted(keys)


def _locate_embed_tokens(model: Any) -> tuple[Any, str] | None:
    """Find ``(parent_module, attr_name)`` owning the input embedding, or None."""
    import mlx.nn as nn

    embed_types = (nn.Embedding, nn.QuantizedEmbedding)
    if hasattr(model, "embed_tokens") and isinstance(model.embed_tokens, embed_types):
        return model, "embed_tokens"
    inner = getattr(model, "model", None)
    if (
        inner is not None
        and hasattr(inner, "embed_tokens")
        and isinstance(inner.embed_tokens, embed_types)
    ):
        return inner, "embed_tokens"
    for name, mod in model.named_modules():
        if isinstance(mod, embed_types):
            parent_name, _, attr = name.rpartition(".")
            if not parent_name:
                return model, attr
            parent = model
            for part in parent_name.split("."):
                parent = getattr(parent, part)
            return parent, attr
    return None


@contextlib.contextmanager
def _neftune(model: Any, alpha: float | None) -> Iterator[None]:
    if alpha is None or alpha <= 0:
        yield
        return
    located = _locate_embed_tokens(model)
    if located is None:
        warnings.warn(
            "neftune_noise_alpha set but no embedding layer was found; skipping.",
            RuntimeWarning,
            stacklevel=2,
        )
        yield
        return
    import mlx.core as mx
    import mlx.nn as nn

    class _NoisyEmbedding(nn.Module):
        """Wrap an embedding to add NEFTune noise during training.

        Magnitude is ``alpha / sqrt(seq_len * dim)``, uniform in ``[-mag, mag]``.
        The wrapper inherits ``.training`` from its parent, so ``.eval()``
        disables the noise.
        """

        def __init__(self, inner: Any, alpha: float) -> None:
            super().__init__()
            self.inner = inner
            self.alpha = alpha

        def __call__(self, x: Any) -> Any:
            out = self.inner(x)
            if self.training and self.alpha > 0:
                dims = out.shape[-1] * out.shape[-2]
                mag = self.alpha / (dims**0.5)
                out = out + mx.random.uniform(low=-mag, high=mag, shape=out.shape, dtype=out.dtype)
            return out

        def as_linear(self, x: Any) -> Any:
            return self.inner.as_linear(x)

    parent, attr = located
    original = getattr(parent, attr)
    setattr(parent, attr, _NoisyEmbedding(original, alpha))
    try:
        yield
    finally:
        setattr(parent, attr, original)


@contextlib.contextmanager
def _silenced() -> Iterator[None]:
    """Suppress ``train``'s unconditional prints and tqdm progress bars."""
    import mlx_lm.tuner.trainer as trainer_mod

    # tqdm is a (private) re-export on the trainer module; patch via __dict__ so
    # neither pyright's import check nor ruff's getattr/setattr rules fire.
    patched = trainer_mod.__dict__
    original_tqdm = patched["tqdm"]
    original_stdout = sys.stdout

    def _silent_tqdm(iterable: Any = None, *args: Any, **kwargs: Any) -> Any:
        return iterable if iterable is not None else iter(())

    patched["tqdm"] = _silent_tqdm
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = original_stdout
        patched["tqdm"] = original_tqdm
