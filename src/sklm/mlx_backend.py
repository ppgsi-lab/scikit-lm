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
import os
import shutil
import sys
import tempfile
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .backend import common_token_prefix, resolve_max_new_tokens, resolve_max_seq_length
from .callbacks import Callback
from .config import GenerationConfig, LoRAConfig, ModelConfig, QuantizationConfig, TrainingConfig
from .serialize import TrainingExample

__all__ = ["MLXBackend"]

# MLX's native quantizer is the only library on this backend; it supports these
# bit widths. The matrix mirrors the HF backend so both validate the same way.
_MLX_METHODS: dict[str, frozenset[int]] = {"mlx": frozenset({2, 3, 4, 6, 8})}
_DTYPE_NAMES: dict[str, str] = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


class _StopTraining(Exception):
    """Raised from the validation callback to stop ``mlx_lm`` training early.

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

    def _load(self, model_name: str, model_config: ModelConfig) -> None:
        try:
            import mlx.core as mx
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

        quantization = model_config.quantization
        if quantization is not None:
            _resolve_mlx_method(quantization)
        src = (
            _convert_quantized(
                model_name, quantization.bits, model_config.precision, quantization.group_size
            )
            if quantization is not None
            else model_name
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
        model, tokenizer = loaded[0], loaded[1]
        if model_config.tokenizer is not None:
            tokenizer = _load_tokenizer(model_config.tokenizer, model_config.trust_remote_code)

        dtypes = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
        model.set_dtype(dtypes[model_config.precision])
        mx.eval(model.parameters())

        if model_config.lora is not None:
            _apply_lora(model, model_config.lora)

        self._model = model
        self._tokenizer = tokenizer

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

        dataset = _MLXTextDataset(epoch_texts, self._tokenizer, seq_len)
        # mlx_lm's batcher needs a full batch per step; clamp batch_size down to the
        # dataset size so tiny inputs (down to sklearn's 1-row conformance checks)
        # still train, the way HFBackend's Trainer already does.
        if len(dataset) < training.batch_size:
            training = replace(training, batch_size=len(dataset))
        val_dataset = None
        if eval_examples:
            val_dataset = _MLXTextDataset(lambda _: eval_examples, self._tokenizer, seq_len)
            if len(val_dataset) < training.batch_size:
                raise ValueError(
                    f"MLXBackend needs at least batch_size={training.batch_size} validation rows, "
                    f"got {len(val_dataset)}; lower batch_size or validation_split, "
                    "or use HFBackend."
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
        optimizer = _build_optimizer(
            training,
            max(1, iters // training.grad_accumulation_steps),
            training.resolved_learning_rate(model_config),
        )
        iterate_fn = _make_iterate_batches(random_state)
        loss_fn = _make_loss(training.label_smoothing) if training.label_smoothing > 0 else None
        patience = training.early_stopping_patience

        ckpt = training.checkpoint
        with tempfile.TemporaryDirectory(prefix="sklm_mlx_") as tmpdir:
            ckpt_dir = (ckpt.dir if ckpt is not None else None) or tmpdir
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            # Resume: when ``dir`` already holds snapshots, load the most recent
            # weights before training instead of starting from the base model.
            if ckpt is not None and ckpt.dir is not None:
                _resume_weights(lm, ckpt_dir)
            # Cadence in micro-steps (mlx-lm counts micro-steps): an optimizer
            # step is ``grad_accumulation_steps`` micro-steps, an epoch is
            # ``micro_per_epoch``. ``None`` disables trajectory saving (no config).
            save_every = (
                None
                if ckpt is None
                else ckpt.each
                * (training.grad_accumulation_steps if ckpt.on == "step" else micro_per_epoch)
            )
            report = _loss_report_callback(
                callback,
                lm,
                optimizer,
                iters,
                patience,
                ckpt_dir=ckpt_dir,
                save_every=save_every,
                keep=ckpt.keep if ckpt is not None else None,
            )
            args = TrainingArgs(
                batch_size=training.batch_size,
                iters=iters,
                grad_accumulation_steps=training.grad_accumulation_steps,
                grad_checkpoint=training.gradient_checkpointing,
                max_seq_length=seq_len,
                adapter_file=str(Path(ckpt_dir) / "adapters.safetensors"),
                steps_per_report=1,
                steps_per_eval=micro_per_epoch if val_dataset is not None else iters + 1,
                # mlx-lm's own saving is disabled; the report callback writes
                # numbered checkpoint-<step>.safetensors snapshots instead.
                steps_per_save=iters + 1,
                val_batches=-1,
            )
            train_kwargs: dict[str, Any] = {
                "model": lm,
                "optimizer": optimizer,
                "train_dataset": dataset,
                "val_dataset": val_dataset,
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
                if ckpt is not None and ckpt.dir is not None:
                    mx.save_safetensors(str(Path(ckpt_dir) / "best.safetensors"), dict(report.best))
        lm.eval()

    def generate(self, prompts: Sequence[str], generation: GenerationConfig) -> list[str]:
        """Sample a continuation per prompt (greedy when ``temperature <= 0``)."""
        if not prompts:
            return []
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        self._model.eval()
        temp = generation.temperature if generation.temperature > 0 else 0.0
        sampler = make_sampler(temp=temp, top_p=generation.top_p, top_k=generation.top_k)
        processors = (
            make_logits_processors(repetition_penalty=generation.repetition_penalty)
            if generation.repetition_penalty is not None
            else None
        )
        prompt_ids = self._truncated_prompt_ids(prompts)
        response = batch_generate(
            self._model,
            self._tokenizer,
            prompts=prompt_ids,
            max_tokens=resolve_max_new_tokens(generation, self._max_seq_length),
            sampler=sampler,
            logits_processors=processors,
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

        Pairs are scored as one right-padded batch in a single forward pass. The
        continuation boundary per pair is the end of the longest token prefix
        shared with its prompt (so a BPE merge at the prompt's trailing space is
        scored rather than dropped) -- identical semantics to
        :meth:`HFBackend.score`, so the classifier ranks verbalizers the same way
        on both backends. ``reduce`` collapses each pair's per-token
        log-likelihoods to the ``"mean"`` (default) or the ``"sum"``.
        """
        if not prompts:
            return []
        import mlx.core as mx
        import mlx.nn as nn

        self._model.eval()
        fulls: list[list[int]] = []
        starts: list[int] = []
        for prompt, continuation in zip(prompts, continuations, strict=True):
            prompt_ids = self._tokenizer.encode(prompt)
            full = self._tokenizer.encode(prompt + continuation)[: self._max_seq_length]
            starts.append(max(common_token_prefix(prompt_ids, full), 1))
            fulls.append(full)
        max_len = max(len(f) for f in fulls)
        # Right padding with 0: causal attention keeps real tokens unaffected by
        # trailing pad positions, and per-pair slicing never reads them.
        arr = np.zeros((len(fulls), max_len), dtype=np.int32)
        for i, full in enumerate(fulls):
            arr[i, : len(full)] = full
        ids = mx.array(arr)
        # No attention mask: causal attention already prevents real tokens from
        # reading trailing right-pad, and per-pair slicing below never scores pad
        # positions, so a mask would change nothing (HF passes one only because
        # its tokenizer.pad returns it for free).
        logits = self._model(ids[:, :-1]).astype(mx.float32)
        logprobs = nn.log_softmax(logits, axis=-1)
        token_logprobs = mx.take_along_axis(logprobs, ids[:, 1:][..., None], axis=-1).squeeze(-1)
        scores: list[float] = []
        for i, full in enumerate(fulls):
            cont = token_logprobs[i, starts[i] - 1 : len(full) - 1]
            if len(full) < 2 or cont.size == 0:
                scores.append(float("-inf"))
            else:
                pooled = mx.sum(cont) if reduce == "sum" else mx.mean(cont)
                scores.append(float(pooled.item()))
        return scores


class _MLXTextDataset:
    """Map-style dataset over per-epoch (re)serialized rows.

    ``__getitem__`` returns ``(token_ids, prompt_offset)``: ``prompt_offset`` is
    the number of leading tokens ``iterate_batches`` masks out of the loss
    (``0`` unless ``loss_on_target_only`` masks the row), found as the longest
    common token prefix between the example's prompt and full text. ``reshuffle``
    re-draws the buffer from ``epoch_texts`` for the next epoch index so the
    column-order permutation is redrawn at each epoch boundary; tokenization is
    lazy and memoized per epoch.
    """

    def __init__(
        self,
        epoch_texts: Callable[[int], list[TrainingExample]],
        tokenizer: Any,
        max_seq_length: int,
    ) -> None:
        self._epoch_texts = epoch_texts
        self._tok = tokenizer
        self._max = max_seq_length
        self._eos = tokenizer.eos_token_id
        self._epoch = 0
        self._buffer = epoch_texts(0)
        self._cache: dict[int, tuple[list[int], int]] = {}

    def reshuffle(self) -> None:
        self._epoch += 1
        self._buffer = self._epoch_texts(self._epoch)
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
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
        result = (ids, offset)
        self._cache[idx] = result
        return result


def _save_checkpoint(ckpt_dir: str, step: int, params: Any, keep: int | None) -> None:
    """Write a ``checkpoint-<step>.safetensors`` snapshot and prune to ``keep``.

    ``params`` is a ``tree_flatten`` list of ``(path, array)``. Only the numbered
    snapshots are pruned (oldest first); ``best.safetensors`` is named apart, so
    it survives. ``keep`` of ``None`` retains every snapshot.
    """
    import mlx.core as mx

    mx.save_safetensors(str(Path(ckpt_dir) / f"checkpoint-{step}.safetensors"), dict(params))
    if keep is not None:
        existing = sorted(
            Path(ckpt_dir).glob("checkpoint-*.safetensors"),
            key=lambda p: int(p.stem.split("-")[1]),
        )
        for old in existing[:-keep]:
            old.unlink()


def _resume_weights(lm: Any, ckpt_dir: str) -> None:
    """Load the most recent ``checkpoint-<step>.safetensors`` snapshot into ``lm``.

    No-op when ``ckpt_dir`` holds no numbered snapshots (a fresh run). Only the
    trainable weights are restored; the optimizer state and step counter are not,
    so training restarts its schedule from the loaded weights.
    """
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    existing = sorted(
        Path(ckpt_dir).glob("checkpoint-*.safetensors"),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    if not existing:
        return
    # mx.load of a .safetensors returns a {name: array} dict; the mlx stub types
    # it as the broader array|tuple union, so narrow it for tree_unflatten.
    weights: dict[str, Any] = mx.load(str(existing[-1]))  # type: ignore[assignment]
    lm.update(tree_unflatten(list(weights.items())))
    mx.eval(lm.parameters())


def _loss_report_callback(
    callback: Callback,
    lm: Any,
    optimizer: Any,
    iters: int,
    patience: int | None,
    *,
    ckpt_dir: str,
    save_every: int | None,
    keep: int | None,
) -> Any:
    """Build an mlx-lm ``TrainingCallback`` that forwards train/val loss, the
    gradient norm and the current device memory to ``callback``.

    The gradient norm stays ``None`` when clipping is disabled
    (``max_grad_norm`` unset), since nothing computes it then.

    When ``save_every`` is set, every that-many micro-steps a numbered checkpoint
    is written under ``ckpt_dir`` and pruned to the ``keep`` most recent. Whenever
    a validation set yields reports the best (lowest-loss) snapshot is tracked and
    exposed as ``.best`` for the caller to restore; with ``patience`` set it also
    raises :class:`_StopTraining` once validation stops improving for ``patience``
    reports.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.callbacks import TrainingCallback

    class _LossReport(TrainingCallback):
        def __init__(self) -> None:
            # tree_flatten's result (a list of (path, array)); typed Any
            # because mlx-lm ships no stubs for it.
            self.best: Any = None
            self.best_loss = float("inf")
            self.no_improve = 0

        def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
            callback.on_memory(int(mx.get_active_memory()))
            norm = optimizer.state.get("last_grad_norm")
            step = int(train_info["iteration"])
            callback.on_train_report(
                step=step,
                total_steps=iters,
                loss=float(train_info["train_loss"]),
                epoch=None,
                learning_rate=train_info.get("learning_rate"),
                grad_norm=float(norm.item()) if norm is not None else None,
            )
            if save_every is not None and step % save_every == 0:
                snapshot = tree_flatten(lm.trainable_parameters())
                mx.eval([v for _, v in snapshot])
                _save_checkpoint(ckpt_dir, step, snapshot, keep)

        def on_val_loss_report(self, val_info: dict[str, Any]) -> None:
            loss = float(val_info["val_loss"])
            callback.on_eval_report(step=int(val_info["iteration"]), loss=loss, epoch=None)
            if loss < self.best_loss - 1e-9:
                self.best_loss = loss
                # mlx arrays are immutable, so snapshotting the current leaves
                # is enough -- the optimizer rebinds new arrays on each step.
                snapshot = tree_flatten(lm.trainable_parameters())
                mx.eval([v for _, v in snapshot])
                self.best = snapshot
                self.no_improve = 0
            elif patience is not None:
                self.no_improve += 1
                if self.no_improve >= patience:
                    raise _StopTraining

    return _LossReport()


def _make_iterate_batches(random_state: int | None) -> Callable[..., Iterator[Any]]:
    """Build the ``iterate_batches`` ``train`` will drive.

    Mirrors mlx-lm's batching (permute, pad to a multiple of 32, truncate) but
    calls ``dataset.reshuffle()`` at each wrap-around so a new column-order
    permutation is drawn per epoch. Epoch 0 uses the dataset's initial buffer;
    the first wrap fires epoch 1. Padding with 0 is harmless: the loss masks
    padded positions via ``lengths``.
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
                yield mx.array(arr), mx.array(list(zip(offsets, trunc, strict=True)))
            if not loop:
                break

    return iterate_batches


def _build_optimizer(training: TrainingConfig, steps: int, lr: float) -> Any:
    """Build the optimizer with its LR schedule sized over ``steps`` optimizer
    steps (schedules advance once per ``optimizer.update``, not per micro-step)."""
    import mlx.core as mx
    import mlx.optimizers as opt

    def main_schedule(n: int) -> Any:
        match training.lr_scheduler:
            case "constant":
                return opt.linear_schedule(lr, lr, max(1, n))
            case "linear":
                return opt.linear_schedule(lr, 0.0, max(1, n))
            case "cosine":
                return opt.cosine_decay(lr, max(1, n))
            case _:
                raise ValueError(f"unknown lr_scheduler {training.lr_scheduler!r}")

    warmup_n = min(round(steps * training.warmup_ratio), steps) if training.warmup_ratio > 0 else 0
    if warmup_n > 0:
        schedule: Any = opt.join_schedules(
            [opt.linear_schedule(0.0, lr, warmup_n), main_schedule(steps - warmup_n)], [warmup_n]
        )
    elif training.lr_scheduler == "constant":
        schedule = lr
    else:
        schedule = main_schedule(steps)

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


def _make_loss(eps: float) -> Callable[..., Any]:
    """Full-sequence cross-entropy with label smoothing, matching the masking of
    ``mlx_lm.tuner.trainer.default_loss`` (offset 0 -> loss on every token)."""
    import mlx.core as mx
    import mlx.nn as nn

    def loss(model: Any, batch: Any, lengths: Any) -> Any:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        ce = nn.losses.cross_entropy(logits, targets, label_smoothing=eps, reduction="none") * mask
        ntoks = mask.sum()
        return ce.astype(mx.float32).sum() / ntoks, ntoks

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


def _convert_quantized(
    model_name: str, q_bits: int, precision: str, group_size: int | None = None
) -> str:
    """Convert ``model_name`` to MLX's native ``q_bits``-bit format, cached under
    ``~/.cache/sklm/mlx``. Requires an mlx-convertible HF repo (safetensors) or a
    pre-converted ``mlx-community`` model.

    The conversion writes into a temporary sibling directory and is renamed into
    the cache path only once complete, so an interrupted convert never leaves a
    half-written cache behind (``mlx_lm`` writes ``config.json`` before the
    tokenizer files, so the files' mere presence proves nothing)."""
    from mlx_lm import convert

    slug = model_name.replace("/", "--")
    group = f"-g{group_size}" if group_size is not None else ""
    cache = Path.home() / ".cache" / "sklm" / "mlx" / f"{slug}-{q_bits}bit{group}-{precision}"
    if (cache / "config.json").exists():
        return str(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        # A partial directory from a convert interrupted before this rename
        # scheme existed; rebuild it.
        shutil.rmtree(cache)
    convert_kwargs: dict[str, Any] = {
        "hf_path": model_name,
        "quantize": True,
        "q_bits": q_bits,
        "dtype": _DTYPE_NAMES[precision],
    }
    if group_size is not None:
        convert_kwargs["q_group_size"] = group_size
    staging = Path(tempfile.mkdtemp(dir=cache.parent, prefix=f"{cache.name}.tmp"))
    try:
        # convert refuses to write into an existing directory, hence the child.
        out = staging / "model"
        convert(mlx_path=str(out), **convert_kwargs)
        os.replace(out, cache)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return str(cache)


def _load_tokenizer(tokenizer_id: str, trust_remote_code: bool) -> Any:
    from mlx_lm.utils import load_tokenizer

    path = Path(tokenizer_id)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(tokenizer_id))
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
